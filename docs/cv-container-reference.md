# CV Demo Container — Complete Reference

Full technical reference for the Advantech YOLO vision demo container: how it
is built, what is inside it, how it is configured, how it is deployed to a
WEDA-managed edge device, how detection telemetry is published, and how to
operate and troubleshoot it.

For a five-minute orientation read
**[cv-container-overview.md](cv-container-overview.md)** instead. This document
is the exhaustive version.

---

## Table of contents

1. [What this container is](#1-what-this-container-is)
2. [Platform and target device](#2-platform-and-target-device)
3. [Image lineage](#3-image-lineage)
4. [Traps inside the vendor base image](#4-traps-inside-the-vendor-base-image)
5. [The demo runner](#5-the-demo-runner)
6. [MQTT telemetry contract](#6-mqtt-telemetry-contract)
7. [Configuration reference](#7-configuration-reference)
8. [Display and X11](#8-display-and-x11)
9. [Demo clips — measured survey](#9-demo-clips--measured-survey)
10. [Model selection — measured](#10-model-selection--measured)
11. [Fine-tuning pipeline](#11-fine-tuning-pipeline)
12. [Building and publishing images](#12-building-and-publishing-images)
13. [WEDA deployment](#13-weda-deployment)
14. [Operations runbook](#14-operations-runbook)
15. [Troubleshooting](#15-troubleshooting)
16. [Security posture](#16-security-posture)
17. [Limitations and known gaps](#17-limitations-and-known-gaps)

---

## 1. What this container is

An **unattended** computer-vision demo for Advantech Jetson edge devices. It:

* loops a bundled video clip forever,
* runs YOLO object detection on every frame using the device GPU,
* renders annotated output fullscreen on the device's own physical screen,
* publishes detection results to an MQTT broker running beside it.

It is deployed and lifecycle-managed through the **WEDA container-management
API**, so it can be pushed to a fleet rather than started by hand.

### How it differs from the repository's other applications

`src/advantech-yolo.py` is an **interactive** tool. It cannot serve as the demo:

| | `advantech-yolo.py` | `demo-od-loop.py` |
|:--|:--|:--|
| input | prompts via `input()` — needs a TTY | environment variables only |
| looping | `predict(stream=True)` **ends at EOF** | rewinds capture in place, runs forever |
| logging | prints per frame | interval summary only |
| shutdown | none | handles `SIGTERM` for clean `docker stop` |
| preconditions | fails late and vaguely | fails fast, naming the exact cause |

A WEDA-deployed container has no TTY attached, so `interactive_mode()` blocks
forever on the first prompt. That, plus the absence of looping, is why a
dedicated runner exists.

---

## 2. Platform and target device

| | |
|:--|:--|
| device alias | `adlk.edgedevice.2` (172.22.160.197, user `ubuntu`) |
| hardware | Advantech EPC-R7300, NVIDIA Jetson Orin |
| WEDA device id | `74fe488d5d54` |
| JetPack | 6.2 — `R36 (release), REVISION: 4.4` |
| architecture | `aarch64` |
| Docker | 27.5.1, **default runtime already `nvidia`** |
| torch | `2.5.0a0+872d972e41.nv24.08` (NVIDIA JetPack build) |
| CUDA device | `Orin` |
| RAM | 7.6 GB **shared between CPU and GPU** |
| disk | 116 GB, typically ~19 GB free |
| WEDA cluster | `weda-sit-k3s.weda.dev`, tenant `central` |
| org | `Kevin.Chien@advantech.com.tw` — `b60fee4c-3e27-43eb-9a6e-877a52bb54f5` |

Because the default Docker runtime is already `nvidia`, the compose stack does
**not** need `runtime: nvidia`. Omitting it removes a field WEDA's compose
passthrough might reject, for no loss of function.

RAM is the binding constraint. 7.6 GB shared, with roughly 3 GB free once the
platform containers are running — see §11.2 for what happens when you ignore
this.

---

## 3. Image lineage

Every image is built **natively on the device**. Building arm64 on an x86 host
under qemu was tried and abandoned: pulling the 14.5 GB base ran at 0.12 MB/s
from the build VM versus ~8 MB/s from the device — a ~65x difference.

```
edgesync.azurecr.io/advantech/advantech-yolo-vision-applications:1.6.0-Ubuntu22.04-ARM
  │  vendor base — 14.5 GB, ultralytics 8.3.220, torch 2.5.0 (JetPack)
  │
  ├─ 1.0.0        Dockerfile.demo-od         clip + weights + runner + entrypoint
  │                                          COPY-only: no RUN at all
  ├─ 1.2.0-ul84   Dockerfile.demo-od-ul84    ultralytics -> 8.4.101 (YOLO26 support)
  ├─ 1.3.0        Dockerfile.demo-od-jar     + fine-tuned jar26n.pt (jar demo)
  ├─ 1.4.0        Dockerfile.demo-od-bottle  + OD_bottle_2.mp4 + yolo26n.pt
  └─ 1.5.0        Dockerfile.demo-od-mqtt    + paho-mqtt + telemetry publisher
                                             <-- CURRENTLY DEPLOYED
```

### Why 1.0.0 has no `RUN`

Every layer is a `COPY`. Nothing executes during build, so an arm64 image can
be assembled on any architecture without qemu emulation, and the build
completes in **under a second** on the device. Only `1.2.0-ul84` and `1.5.0`
need `pip`, and those must build on the device (or another aarch64 host).

### Build-time assertions

Each derived Dockerfile ends with a `RUN python3 -c "... assert ..."` block, so
a regression **fails the build** rather than the demo. Collectively they check:

* ultralytics is 8.4.x when YOLO26 weights are present,
* `cv2.getBuildInformation()` still contains `GTK` (display works),
* `torch.__version__` still ends `nv24.08` (JetPack CUDA build not replaced),
* the YOLO26 head reports `end2end=True`,
* the fine-tuned model's class map is exactly `{0: 'jar'}`,
* the demo clip exists and is decodable,
* `paho-mqtt` imports and the publisher builds its topic layout.

These exist because every one of them was, at some point, silently wrong.

---

## 4. Traps inside the vendor base image

Two defects in the base image cause **silent** failures. Both cost significant
debugging time; both have one-line fixes.

### 4.1 `opencv-python-headless` shadows JetPack's OpenCV

The base image contains two OpenCV installations:

| package | path | GUI | GStreamer | CUDA |
|:--------|:-----|:----|:----------|:-----|
| pip `opencv-python-headless` 4.11.0.86 | `/usr/local/lib/python3.10/dist-packages` **← wins** | **NONE** | NO | no |
| apt `libopencv-python` 4.8.0 (NVIDIA) | `/usr/lib/python3.10/dist-packages` | **GTK2** | 1.20.3 | yes |

`/usr/local` precedes `/usr/lib` on `sys.path`, so the headless wheel always
wins and any `cv2.imshow` raises:

```
cv2.error: OpenCV(4.11.0) ... The function is not implemented.
Rebuild the library with Windows, GTK+ 2.x or Cocoa support.
```

**Consequence:** the `--show` flag of `src/advantech-yolo.py` has never worked
inside this container, contrary to the main README (now annotated).

**Fix — no rebuild, no uninstall:**

```bash
PYTHONPATH=/usr/lib/python3.10/dist-packages
YOLO_AUTOINSTALL=false
```

`PYTHONPATH` entries precede site-packages, so JetPack's build wins.
`YOLO_AUTOINSTALL=false` is **not optional**: ultralytics checks its
requirements at import, finds `opencv-python` missing, pip-installs it into
`/usr/local`, and silently re-shadows the fix. The resulting failure appears
unrelated to anything you changed.

### 4.2 ultralytics 8.3.x silently mis-loads YOLO26

YOLO26 uses an **end-to-end, NMS-free** detection head. Ultralytics 8.3.220
does not know that head type and coerces the checkpoint into the legacy
`Detect` head:

```
yolo11n: head=Detect end2end=False nc=80     (correct)
yolo26n: head=Detect end2end=False nc=80     (WRONG — should be end2end=True)
```

It loads **without error or warning** and produces confident nonsense —
measured at ~22 junk boxes per frame with zero true detections (2659 spurious
detections across 120 frames, versus 254 after the fix).

**Fix:** ultralytics >= 8.4 (`1.2.0-ul84` pins 8.4.101). Verify explicitly:

```python
h = YOLO('yolo26n.pt').model.model[-1]
assert getattr(h, 'end2end', False), 'wrong ultralytics for YOLO26'
```

### 4.3 Upgrading ultralytics without breaking the platform

`torch`, `torchvision`, and `numpy` in `/usr/local` are **NVIDIA JetPack
builds**. A PyPI `torch` has no Jetson CUDA support and would silently drop
inference to CPU. ultralytics 8.4 only needs `torch>=1.8.0` and
`numpy>=1.23.0`, both already satisfied, so a generated constraints file keeps
pip's resolver away from them:

```dockerfile
RUN python3 -c "import torch, torchvision, numpy; \
      open('/tmp/c.txt','w').write(f'torch=={torch.__version__}\n...')" \
 && pip3 uninstall -y ultralytics \
 && pip3 install --no-cache-dir -c /tmp/c.txt "ultralytics==8.4.101"
```

The `pip3 uninstall` matters: the base image carries ultralytics as a **source
checkout at `/ultralytics`**, not a normal site-packages install, and without
removing it the old tree keeps winning on `sys.path`.

The `torch>=2.12` constraints visible in ultralytics' metadata apply only to
optional `export-*` extras, which this image does not use.

---

## 5. The demo runner

`src/demo-od-loop.py` — environment-driven, no interactive input, never
prompts.

### 5.1 Lifecycle

```
entrypoint-demo.sh
  └─ wait for X display :0 (probe with a real cv2 window, up to X_WAIT_TIMEOUT_SEC)
       └─ exec demo-od-loop.py
            ├─ validate_environment()   fail fast, name every problem
            ├─ YOLO(MODEL_PATH)         model loaded ONCE, stays resident
            ├─ DetectionPublisher.connect()   best-effort, never fatal
            └─ loop forever:
                 read frame ── EOF? ── rewind in place (CAP_PROP_POS_FRAMES=0)
                 predict
                 imshow(result.plot())
                 publisher.publish()    rate-limited internally
                 interval log
            └─ SIGTERM/SIGINT -> publisher.close(), release, destroyAllWindows
```

### 5.2 Why the model stays resident

At EOF the capture is **rewound in place**. The obvious alternative — wrapping
the existing script in `while true` — reloads the model on every lap, stalling
the demo for seconds each time and re-warming CUDA. Reopening the file is
likewise avoided.

If the rewind itself fails, the runner reopens the capture as a fallback rather
than exiting, since a transient decode error should not end a showroom demo.

### 5.3 Fail-fast preconditions

`validate_environment()` collects **all** problems and reports them together
before the expensive model load:

* video file missing,
* model file missing,
* OpenCV without GTK (§4.1) — reported with the exact `PYTHONPATH` remedy,
* `DISPLAY` unset.

A silent failure here would leave the container "running" from WEDA's
perspective while the screen stayed blank. The log block is unmistakable:

```
[demo] PRECONDITION CHECK FAILED:
[demo]   - OpenCV 4.11.0 at /usr/local/... has no GUI support. Set PYTHONPATH=...
```

CUDA absence is a **warning**, not a failure — CPU inference is slow but the
demo still runs, which is preferable to a dark screen.

### 5.4 Interval logging

Detection counts are summarised on an interval, never per frame:

```
[demo] 11.2 FPS | 6.0 objects/frame | laps=1 | mqtt published=52 dropped=0 connected=True
```

The device root filesystem runs above 80% capacity. At ~12 FPS a per-frame log
line with Docker's default unbounded `json-file` driver would consume the
remaining space over a multi-day run and take the device down. The stack also
caps logs at `max-size: 10m, max-file: 3`.

### 5.5 Entrypoint X wait

`docker/entrypoint-demo.sh` probes the display by actually opening and
destroying a cv2 window, retrying every 3 s up to `X_WAIT_TIMEOUT_SEC`
(default 120). On a device reboot Docker can start the container before gdm has
brought up X; without this wait the demo would exit on a missing display and,
under `restart: unless-stopped`, enter a crash loop.

---

## 6. MQTT telemetry contract

An `eclipse-mosquitto:2` broker runs as a second service in the same stack.
Both services use host networking, so the broker is at **`127.0.0.1:1883`** on
the device and reachable at the device's LAN address from elsewhere.

The broker starts with `-c /mosquitto-no-auth.conf`. mosquitto 2.x refuses
remote connections under its default config; that stock file ships inside the
image and enables an anonymous listener on 1883 without needing a bind-mounted
config, which a WEDA-deployed stack cannot easily supply.

### 6.1 Topics

Base defaults to `advantech/<DEVICE_ID>/vision`, overridable with
`MQTT_TOPIC_BASE`. On the deployed device: `advantech/74fe488d5d54/vision`.

| topic | retained | QoS | published | payload |
|:------|:---------|:----|:----------|:--------|
| `<base>/status` | **yes** | 1 | on connect, and on death via LWT | `online` \| `offline` |
| `<base>/meta` | **yes** | 1 | once, on connect | JSON run configuration |
| `<base>/detections` | no | `MQTT_QOS` (0) | every `MQTT_INTERVAL_SEC` | JSON detection summary |

**Last Will and Testament.** `<base>/status` is registered as the will with
payload `offline`, QoS 1, retained. If the container dies without a clean
shutdown the broker publishes it automatically. Without an LWT, "publisher
dead" and "publisher alive but detecting nothing" are indistinguishable to a
subscriber.

**Retained status and meta** mean a subscriber connecting late immediately
learns device state and run configuration instead of waiting for the next
event.

### 6.2 `detections` payload

```jsonc
{
  "timestamp":   "2026-07-19T16:54:47.949Z",   // ISO-8601 UTC, ms precision
  "deviceId":    "74fe488d5d54",               // DEVICE_ID
  "frame":       830,                           // cumulative frames since start
  "lap":         1,                             // clip loop count
  "fps":         12.17,                          // measured render+inference rate
  "objectCount": 7,                             // total boxes this frame
  "classCounts": { "bottle": 7 },               // count per class
  "detections": [                               // omitted if MQTT_INCLUDE_BOXES=false
    {
      "class":      "bottle",
      "confidence": 0.8932,                     // 0.0-1.0, 4 dp
      "bbox":       [0.5, 116.1, 344.9, 1417.2] // [x1, y1, x2, y2]
    }
  ]
}
```

**`bbox` is in source-image pixels, not normalised.** For `OD_bottle_2.mp4` the
frame is 2560x1440. The frame dimensions are **not** carried in the payload —
a consumer needing 0-1 coordinates must take them from the clip named in
`<base>/meta`, or hard-code per clip. This is a known gap (§17).

`classCounts` is the field to chart or alarm on. `detections` dominates message
size; setting `MQTT_INCLUDE_BOXES=false` cuts the payload to roughly a tenth
when only counts matter.

### 6.3 `meta` payload

```jsonc
{
  "model":         "yolo11n.pt",       // weights actually loaded
  "source":        "OD_bottle_2.mp4",  // clip being looped
  "confThreshold": 0.4,
  "iouThreshold":  0.45,
  "demoVersion":   "1.1.0"             // runner version, NOT the image tag
}
```

### 6.4 Rate limiting

Publishing is **interval-based, not per frame**. At ~12 FPS a per-frame topic
emits 12 msg/s of near-identical payloads for no analytical gain. The rate
limit lives inside `DetectionPublisher.publish()`, so the render loop contains
no telemetry logic and cannot be broken by changing publishing behaviour.

### 6.5 Failure behaviour

**Telemetry is best-effort and never fatal.** Every public method of
`DetectionPublisher` degrades to a no-op when the broker is unreachable, so
callers need no guards and the demo keeps rendering. Losing telemetry must
never take the display down.

* `connect()` catches all exceptions, logs, and returns `False`.
* Missing `paho-mqtt` is a warning, not an error.
* The client auto-reconnects with backoff (1 s → 30 s).
* Publish failures increment a counter and log with exception context.

Failures are **logged with context, never silently swallowed**. Read the
counters in the interval line:

| observation | meaning |
|:--|:--|
| `connected=False` | broker unreachable; client is retrying |
| `dropped` rising, `connected=True` | broker accepting the connection but rejecting publishes — check broker logs and QoS |
| `published` rising, `dropped=0` | healthy |

### 6.6 Subscribing

```bash
# everything, on the device
docker run --rm --network host eclipse-mosquitto:2 \
  mosquitto_sub -h 127.0.0.1 -p 1883 -t 'advantech/#' -v

# counts only, from another host
mosquitto_sub -h 172.22.160.197 -t 'advantech/+/vision/detections' \
  | jq -c '{ts: .timestamp, n: .objectCount, cls: .classCounts}'

# liveness
mosquitto_sub -h 172.22.160.197 -t 'advantech/+/vision/status' -v
```

---

## 7. Configuration reference

Every setting is an environment variable. No config files, no CLI arguments.

### 7.1 Demo runner

| variable | default | purpose |
|:---------|:--------|:--------|
| `VIDEO_PATH` | `/advantech/data/OD_Jar.mp4` | clip to loop |
| `MODEL_PATH` | `/advantech/models/yolo11n.pt` | weights to load |
| `CONF_THRESHOLD` | `0.25` | detection confidence floor |
| `IOU_THRESHOLD` | `0.45` | NMS IoU |
| `INFER_DEVICE` | `0` | `0` = GPU, `cpu` = CPU |
| `WINDOW_NAME` | `Advantech YOLO11 - Object Detection` | X window title |
| `FULLSCREEN` | `true` | fullscreen the window |
| `STATS_INTERVAL_SEC` | `60` | interval log period |
| `X_WAIT_TIMEOUT_SEC` | `120` | entrypoint wait for X |

### 7.2 MQTT publisher

| variable | default | purpose |
|:---------|:--------|:--------|
| `MQTT_ENABLED` | `true` | `false` runs the demo with no telemetry |
| `MQTT_HOST` | `127.0.0.1` | broker address |
| `MQTT_PORT` | `1883` | broker port |
| `MQTT_QOS` | `0` | QoS for `detections` (status/meta always QoS 1) |
| `MQTT_INTERVAL_SEC` | `1.0` | publish interval |
| `MQTT_INCLUDE_BOXES` | `true` | include per-detection boxes |
| `MQTT_TOPIC_BASE` | `advantech/<DEVICE_ID>/vision` | topic prefix |
| `DEVICE_ID` | container hostname | identity in payloads |

### 7.3 Platform (set by the stack, rarely changed)

| variable | value | why |
|:---------|:------|:----|
| `PYTHONPATH` | `/usr/lib/python3.10/dist-packages` | **required** — §4.1 |
| `YOLO_AUTOINSTALL` | `false` | **required** — §4.1 |
| `DISPLAY` | `:0` | device screen |
| `XAUTHORITY` | `/tmp/.docker.xauth` | mounted gdm cookie |
| `NVIDIA_DRIVER_CAPABILITIES` | `all,compute,video,utility,graphics` | GPU + display |

---

## 8. Display and X11

X runs on `:0` under **gdm** with autologin as `ubuntu` (uid 1000).

| resource | host path | container path |
|:---------|:----------|:---------------|
| X socket | `/tmp/.X11-unix` | `/tmp/.X11-unix` |
| auth cookie | `/run/user/1000/gdm/Xauthority` | `/tmp/.docker.xauth` (ro) |
| tegra driver libs | `/usr/lib/aarch64-linux-gnu/tegra` | same (ro) |

**The cookie is at `/run/user/1000/gdm/Xauthority`, not `~/.Xauthority`.** The
home-directory file exists but is stale; using it fails authentication.

The original `docker-compose.yml` resolves `${DISPLAY}` and `${XAUTHORITY}`
from the invoking shell. A WEDA-deployed container has **no user session to
inherit from**, so both would resolve empty. The stack therefore hardcodes
`DISPLAY=:0` and mounts the gdm cookie explicitly.

None of these three mounts can be baked into the image: the socket is a live
IPC endpoint, the cookie is a per-boot credential, and the tegra libraries must
match the host driver exactly.

Screen blanking is already disabled on this device (`xset q` reports
`timeout: 0`, DPMS all `0`). On a fresh device run `xset s off -dpms` in the
desktop session or the demo goes black while the container stays healthy.

---

## 9. Demo clips — measured survey

All 12 clips in `data/cv_demo_clips/`, measured with stock `yolo11l` at
conf 0.40, no class filter. *Coverage* = share of sampled frames with at least
one box.

| clip | resolution | secs | coverage | boxes/frm | top classes |
|:-----|:-----------|-----:|---------:|----------:|:------------|
| **OD_bottle_2** | 2560x1440 | 8.0 | **100%** | **7.33** | bottle:396 |
| OD_bottle_1 | 1920x1080 | 5.3 | 100% | 5.89 | bottle:310 |
| OD_bagagge | 1920x1080 | 16.2 | 100% | 6.84 | person:193, suitcase:154 |
| Seg_traffic_2 | 1280x720 | 39.0 | 100% | 19.94 | person, car, motorcycle |
| Seg_traffic_1 | 1280x720 | 21.2 | 100% | 19.15 | car:828, bus, truck |
| Seg_traffic_4 | 2560x1440 | 21.4 | 100% | 12.17 | car:361, bus:181 |
| Seg_traffic_3 | 3840x2160 | 24.0 | 100% | 9.15 | bus:219, car:194 |
| OD_Seg_Sushi | 1280x720 | 16.5 | 73%¹ | 1.22 | person, **pizza**, **cake** |
| OD_Jar | 2560x1440 | 9.3 | 53% | 0.84 | cup, person, vase |
| OD_can | 2560x1440 | 14.8 | 21% | 0.28 | bottle:14 |
| OD_fruit | 1920x1080 | 9.2 | 16% | 0.20 | person:10 |
| 332263_medium | 1280x720 | 5.3 | **0%** | 0.00 | — nothing |

¹ Sushi is the cautionary case: measuring the first 150 of its 495 frames gave
100%; over the whole clip it is 73%. **Always measure the entire clip** — an
early sequential window can be far denser than the rest.

### Interpretation

A clip demos well when its subject is a class COCO knows. `OD_bottle_2` is
ideal: dense, high-confidence, and every detection is genuinely `bottle`.

Clips whose subject COCO does *not* know produce confidently wrong labels,
which reads worse to a viewer than no detection at all:

* `OD_Jar` — jars labelled `cup`/`vase`; the red cable loom labelled
  **`surfboard`** at 5.2 boxes/frame; the filling nozzle labelled `person`.
* `OD_Seg_Sushi` — nigiri labelled **`pizza`** (0.72), tamago labelled
  **`cake`** (0.47).

Those clips need fine-tuning (§11) or should be avoided.

---

## 10. Model selection — measured

### 10.1 On `OD_bottle_2` (deployed clip), all 478 frames

| model | conf | coverage | boxes/frame | FPS |
|:------|-----:|---------:|------------:|----:|
| **yolo11n (deployed)** | 0.40 | 100% | **6.06** | **23.8** |
| yolo26n | 0.40 | 100% | 4.97 | 22.4 |
| yolo11l | 0.40 | 100% | 7.26 | 14.3 |

Deployed class split: `bottle` x2898, 6.06/frame, max 0.95, mean 0.75,
min 0.40. Box count is stable at **5-8 per frame**, which is why the overlay
does not strobe. `yolo11n` wins on both axes here; `yolo26n` is also baked in
and switchable via `MODEL_PATH` with no rebuild.

### 10.2 On `OD_Seg_Sushi`, all 495 frames

| model | conf | coverage | boxes/frame | FPS |
|:------|-----:|---------:|------------:|----:|
| yolo26n | 0.25 | 90% | 1.98 | **29.4** |
| yolo26n | 0.40 | 73% | 1.22 | 28.9 |
| yolo11l | 0.25 | 98% | 2.95 | 16.7 |

The strongest case for YOLO26-nano on this hardware: it approaches `yolo11l`'s
quality at **1.7x the frame rate**. Labels are still wrong (§9).

### 10.3 On `OD_Jar` — why fine-tuning was needed

Stock models at conf 0.05, sampled across the clip:

| class | count | /frame | max conf | >= 0.5 |
|:------|------:|-------:|---------:|-------:|
| surfboard (cables) | 1208 | 5.21 | 0.56 | 51 |
| person (nozzle) | 158 | 0.68 | 0.51 | 1 |
| cup | 156 | 0.67 | 0.63 | 1 |
| **bottle** (the jars) | **31** | **0.13** | **0.30** | **0** |

The correct class never exceeds 0.30 while the false one exceeds 0.5 fifty-one
times. **Raising the confidence threshold makes it worse** — it removes every
jar and keeps the surfboards. No threshold or class filter rescues this; only
fine-tuning does.

---

## 11. Fine-tuning pipeline

Reusable for any clip whose subject COCO does not know. Demonstrated by turning
`yolo26n` into a single-class `jar` detector.

> **Scripts:** [`training/`](../training/) — `survey_clips.py`,
> `make_dataset.py`, `preview_labels.py`, `train.py`, `evaluate.py`.
> **Step-by-step walkthrough:** [training-guide.md](training-guide.md).
> This section is the reasoning behind them; the guide is how to run them.

### 11.1 Pseudo-labels without human annotation

`OD_Jar.mp4` is 232 frames (9.3 s at 25 fps) of 2560x1440.

**The single biggest lever is teacher resolution.** At the default 640px the
2560-wide source shrinks the jars past detectability; at 1536px the same model
finds six times as many:

| teacher imgsz | boxes/frame |
|--------------:|------------:|
| 640 | 0.43 |
| **1536** | **2.56** |

Pipeline (`make_dataset.py`):

1. **Teacher** — `yolo11l`, `imgsz=1536`, `conf=0.10`, restricted to COCO
   container classes `[39, 40, 41, 45, 75]` = bottle, wine glass, cup, bowl,
   vase. Note **`vase` is class 75**, not 46 (46 is banana).
2. **Associate** detections into tracks by greedy IoU (match ≥ 0.25, a track
   survives 12 unmatched frames). Ultralytics' BoT-SORT needs the `lap`
   extension, which is absent and requires a source build on aarch64 — a ~40
   line matcher avoids the dependency.
3. **Interpolate** linearly across gaps ≤ 25 frames. A jar on a conveyor moves
   predictably, so an interpolated box beats a teacher miss. **This is what
   lets the student exceed the teacher.**
4. **Filter** tracks shorter than 4 frames (flicker).
5. **NMS** per frame at IoU 0.55 — the teacher labels one jar as both
   `wine glass` and `cup`, producing stacked duplicates. This merged **330**
   boxes.

Result: **232 frames, 509 boxes, 68% coverage** — from a teacher that never
achieved 40% per-frame.

> **Render a sample of the labels and look at them before training.** The
> duplicate-box defect was invisible in summary statistics and obvious in a
> six-frame contact sheet.

### 11.2 Training host — not the edge device

| host | hardware | 120 epochs @ 960px |
|:-----|:---------|:-------------------|
| **`acn.air-520`** (172.16.9.80, user `kevin`) | 2x RTX 6000 Ada, 251 GB RAM, 64 cores | **15.6 minutes** |
| `adlk.edgedevice.2` | Jetson Orin, 7.6 GB shared | ~15 min **per epoch** |

Training on the Orin alongside the running demo drove load average to **74**
and left `sshd` unable to fork — the device was unreachable for several
minutes. It recovered and the demo survived, but **do not train on the edge
device.**

Both `acn.air-520` GPUs are typically ~93% occupied by an unrelated **vLLM
server** (~2.9 GB free each). A nano model at `batch=4` fits in 1.6 GB and
coexists. **Do not evict that service.**

```bash
scp jar-dataset.tgz acn.air-520:~/jar-train/
ssh acn.air-520 'docker run --rm --runtime nvidia --gpus all --shm-size 8g \
  -v ~/jar-train:/data --entrypoint python3 ultralytics/ultralytics:latest /data/train520.py'
```

Settings: `yolo26n.pt`, 120 epochs, `imgsz=960`, `batch=4`, `patience=40`,
single class `jar`.

### 11.3 A partially-trained checkpoint looks trained and detects nothing

| epochs | val mAP50 | max confidence at inference | usable |
|-------:|----------:|----------------------------:|:-------|
| 2 | 0.47 | **0.026** | no — 0 detections at any sane threshold |
| 120 | 0.972 | 0.95 | yes |

mAP is **rank-based**: box placement scores well long before the classification
head is calibrated. Never judge a run by mAP alone — check the actual
confidence distribution. Confidence calibration needs roughly 100 epochs here.

### 11.4 Result

Final `jar26n`: **mAP50 0.972, mAP50-95 0.899, P 0.934, R 0.918**. On the clip,
53% coverage at 1.94 boxes/frame versus 0.31 for stock `yolo26n` — a 6x
improvement, with labels reading `jar` instead of `surfboard`.

---

## 12. Building and publishing images

### 12.1 Build on the device

```bash
# stage context
rsync -a src/ docker/ models/ data/ adlk.edgedevice.2:/home/ubuntu/yolo-od-demo/

ssh adlk.edgedevice.2 'cd /home/ubuntu/yolo-od-demo && \
  docker build -f docker/Dockerfile.demo-od-mqtt \
    -t harbor.arfa.wise-paas.com/edge-coa/yolo-od-demo:1.5.0 .'
```

### 12.2 Publish to Harbor via the LAN

The device holds **no** `harbor.arfa.wise-paas.com` credential, and putting one
on a shared edge device is undesirable. Instead, stream the image to a
credentialled host over the LAN (~99 MB/s measured) and push from there:

```bash
ssh adlk.edgedevice.2 'docker save harbor.arfa.wise-paas.com/edge-coa/yolo-od-demo:1.5.0' \
  | docker load
docker push harbor.arfa.wise-paas.com/edge-coa/yolo-od-demo:1.5.0
```

Verify architecture after pushing — an accidental amd64 push will deploy and
then fail on the device:

```bash
docker manifest inspect -v harbor.arfa.wise-paas.com/edge-coa/yolo-od-demo:1.5.0 \
  | grep -E '"architecture"|"os"'      # expect arm64 / linux
```

---

## 13. WEDA deployment

Base URL `https://weda-sit-k3s.weda.dev/central/weda`. The SIT domain presents
a **self-signed certificate** — pin it rather than disabling verification:

```bash
openssl s_client -connect weda-sit-k3s.weda.dev:443 \
  -servername weda-sit-k3s.weda.dev -showcerts </dev/null 2>/dev/null \
  | awk '/BEGIN CERT/,/END CERT/' > weda-sit-k3s.pem
curl --cacert weda-sit-k3s.pem ...
```

Mint a JWT with the `weda-login` skill; tokens last **2 hours** and the API
returns `401 token is expired` with no other hint.

### 13.1 Object model

```
stack-config ──(PATCH)──> revision ──(:deploy to deviceIds)──> deployment ──> containers
  org-scoped              immutable          org -> device          status
```

### 13.2 Call sequence

| step | call |
|:-----|:-----|
| registry credential | `GET /api/v1/orgs/{orgId}/containers/registries` — reuse `harbor-arfa-edge-coa`, do not duplicate |
| create stack | `POST /api/v1/orgs/{orgId}/stack-configs` — compose base64 in `composeFileContent` |
| update stack | `PATCH …/stack-configs/{stackConfigId}` → new immutable revision |
| deploy | `POST …/revisions/{stackRevisionId}:deploy` `{"deviceIds":["74fe488d5d54"]}` |
| status | `GET /api/v1/devices/{deviceId}/docker/stacks` |
| remove one | `DELETE /api/v1/devices/{deviceId}/docker/stacks?stackConfigId=…` |
| clear all | `DELETE /api/v1/devices/{deviceId}/stacks/deployments` |

### 13.3 Compose field choices

The stack is deliberately conservative, because WEDA's compose passthrough is
not guaranteed to accept every key:

* **`runtime: nvidia` omitted** — the daemon's default runtime already is
  nvidia, so the GPU is injected regardless.
* **explicit `devices:` list omitted** — `privileged: true` grants the full
  `/dev` tree.
* **no `container_name`** — WEDA derives its own
  (`edge_yolo-od-demo-yolo-od-demo-1`, matching `edge_pulsar-pulsar-1`).
  Pinning one collides with existing containers and breaks WEDA's naming.

That `edge_`-prefixed compose project name is also how you **prove** a
container was created by WEDA rather than by hand.

### 13.4 A stalled deployment is indistinguishable from a healthy one

A deployment can sit at `status: deploying`, `containers: []`, `active: true`,
`error: {}` **indefinitely**. The API returns `HTTP 200` and surfaces no error.
Four consecutive deployments stalled this way before the cause was found.

**Diagnose on the device, not through the API:**

```bash
docker logs dmagent | grep -E "stack_async|composeplugin|Reported stacks"
```

If the last entry predates your deploy, the agent never received it. Outbound
telemetry (`cfgstate_handler … published config shadow state`) keeps working
throughout, which is why the device still looks healthy in WEDA.

**Confirm it is not your compose** by deploying a trivial known-good stack
(busybox) to the same device. If that stalls too, the fault is the agent.

**Fix — both steps, in order:**

1. `docker restart dmagent` — resubscribes to the cloud delta topics.
2. **Delete the stack from the device and deploy again.**

Step 2 is essential and easy to miss. dmagent subscribes to *delta* topics; a
delta published while it was disconnected is **never replayed**, so the pending
deployment stays stuck even after a successful restart. Only a fresh deploy
lands. After both steps the container appeared in under 30 seconds.

### 13.5 `awaitingCleanup` does not self-resolve

A delete-then-deploy can leave the deployment in `awaitingCleanup` with dmagent
logging `context deadline exceeded before applying desired config state`. It
does not recover on its own. Recovery:

```bash
curl -X DELETE …/api/v1/devices/{deviceId}/stacks/deployments   # clear all
curl -X POST   …/revisions/{stackRevisionId}:deploy             # fresh deploy
```

**Prefer clear-all-then-deploy for every version bump.** It has landed reliably
in ~30 s where plain delete-then-deploy wedged.

`DELETE /stacks/deployments` affects **only WEDA-managed stacks** — verified
against dmagent logs, which recorded deletes solely for the stacks in question
and never touched unrelated containers.

### 13.6 Expect a brief outage on every version bump

The old container is removed before the new one is scheduled. Budget for tens
of seconds of dark screen.

---

## 14. Operations runbook

### 14.1 Verify a deployment — all three, never just the first

```bash
# 1. WEDA agrees
curl --cacert weda-sit-k3s.pem -H "Authorization: Bearer $TOK" \
  "$BASE/api/v1/devices/74fe488d5d54/docker/stacks"
#    expect status=running, operationStatus=deployed, containers[].state=running

# 2. the demo is actually inferring
ssh adlk.edgedevice.2 'docker logs --tail 20 edge_yolo-od-demo-yolo-od-demo-1'
#    expect "OpenCV 4.8.0 (GUI enabled)", "CUDA available: Orin",
#           "12.0 FPS | 6.1 objects/frame | laps=2 | mqtt published=… dropped=0"

# 3. it is on the screen
ssh adlk.edgedevice.2 'DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority \
  scrot /tmp/s.png' && scp adlk.edgedevice.2:/tmp/s.png .

# 4. telemetry is flowing
ssh adlk.edgedevice.2 'timeout 8 docker run --rm --network host \
  eclipse-mosquitto:2 mosquitto_sub -h 127.0.0.1 -t "advantech/#" -v -W 6'
```

`status: running` alone is not proof — a container can be up with a blank
screen.

### 14.2 Switch model or clip — no rebuild

Both `yolo11n` and `yolo26n` are baked into 1.4.0+; `OD_bottle_2.mp4` and (in
1.3.0) `OD_Jar.mp4` likewise. Change `MODEL_PATH` / `VIDEO_PATH` in the stack,
`PATCH` for a new revision, clear-all, deploy.

### 14.3 Roll back

Every revision remains deployable:

| ver | revisionId | image | model | clip |
|:----|:-----------|:------|:------|:-----|
| v1 | `c0139896-8dd1-4e62-bbf3-f4cee5410eaa` | 1.0.0 | yolo11n | OD_Jar |
| v2 | `7dc22a75-a938-4f33-9286-77915fb06229` | 1.3.0 | **jar26n** | OD_Jar |
| v3 | `45685531-acef-4405-a74f-d928997c99e7` | 1.4.0 | yolo11n | OD_bottle_2 |
| v4 | `2201a204-630a-4f45-a893-70d0c9540704` | 1.5.0 | yolo11n | OD_bottle_2 + MQTT |

```bash
curl -X DELETE …/devices/74fe488d5d54/stacks/deployments
curl -X POST   …/revisions/<revisionId>:deploy -d '{"deviceIds":["74fe488d5d54"]}'
```

### 14.4 Container control without redeploying

```
POST /api/v1/devices/{deviceId}/docker/stacks/commands:{start|stop|restart|pause|resume}
     {"stackConfigId": "46a6ca7b-5e46-4c31-8f60-639417d72ad1"}
GET  /api/v1/devices/{deviceId}/docker/stacks/commands/results?stackConfigId=…
```

---

## 15. Troubleshooting

| symptom | cause | fix |
|:--------|:------|:----|
| `cv2.error: The function is not implemented … GTK+ 2.x` | headless OpenCV shadowing (§4.1) | set `PYTHONPATH` **and** `YOLO_AUTOINSTALL=false` |
| Display worked, then broke after a change | ultralytics reinstalled `opencv-python` | `YOLO_AUTOINSTALL=false` |
| YOLO26 emits ~22 junk boxes/frame, no error | ultralytics 8.3.x mis-parses the head (§4.2) | ultralytics ≥ 8.4; assert `end2end=True` |
| Deployment stuck `deploying`, `error: {}`, forever | dmagent missed the delta (§13.4) | restart dmagent **and** redeploy |
| Deployment stuck `awaitingCleanup` | apply timed out (§13.5) | clear-all, then deploy |
| Container healthy, screen blank | X auth or `DISPLAY` | check `PRECONDITION CHECK FAILED`; cookie is gdm's, not `~/.Xauthority` |
| Crash loop after device reboot | container started before X | `X_WAIT_TIMEOUT_SEC` (entrypoint already waits 120 s) |
| `401 token is expired` | JWT older than 2 h | re-mint via `weda-login` |
| `403 OrganizationNotFoundOrNoPermission` | token lacks rights to that org/device | check `resource_access` in the JWT |
| Trained model reports good mAP, detects nothing | undertrained; confidence uncalibrated (§11.3) | train ~100 epochs; inspect confidence distribution |
| `mqtt connected=False` | broker down or wrong host | check the broker container; client auto-retries |
| Device unreachable, pings but SSH hangs | memory exhaustion | do not train on the device (§11.2) |
| Model detects nothing after a clip change | fine-tuned model is clip-specific | `jar26n` only knows jars; use stock weights for other clips |

---

## 16. Security posture

**This is a lab demo configuration. Do not carry it into production.**

| area | current state | production requirement |
|:-----|:--------------|:-----------------------|
| MQTT auth | **anonymous**, no credentials | credentials + ACLs |
| MQTT transport | **plaintext** 1883 | TLS 1.2+ on 8883 |
| container privilege | `privileged: true`, host network, host IPC | drop to explicit devices + capabilities |
| X11 cookie | mounted read-only into a privileged container | unavoidable for local display; scope carefully |
| registry credential | kept off the device deliberately | keep it that way; use WEDA registry configs |
| image provenance | unsigned | sign and verify per CRA |

Anonymous MQTT on a plant network is an **open write path** into whatever
consumes these topics. IEC 62443 requires authenticated, encrypted transport
for anything carrying process data.

**Licensing.** This repository is GPL-3.0 and Ultralytics YOLO is AGPL-3.0.
Shipping this image to a customer is a distribution event with copyleft
obligations, and needs legal sign-off before any customer delivery.

---

## 17. Limitations and known gaps

* **`jar26n` is overfit to one 9-second clip by design.** Trained on 232 frames
  of `OD_Jar.mp4` from pseudo-labels; it will not generalise to another line,
  jar, camera angle, or lighting. Correct for a looping demo, wrong for
  production inspection. The deployed bottling demo does not have this problem
  — it uses stock weights and a genuine COCO class.
* **Labels are pseudo-labels**, inherited from a COCO teacher and temporally
  completed. Visually spot-checked, not verified frame by frame.
* **Coverage figures are not decomposed.** "53% coverage" mixes "no object in
  view" with "object missed"; the two have not been separated.
* **`bbox` is unnormalised** and the payload carries no frame dimensions (§6.2).
* **No TensorRT.** Everything runs PyTorch. An on-device `.engine` export
  typically gives 2-3x on Orin and would make higher `imgsz` affordable.
  Engines are hardware- and TRT-version-specific: build on the target, never
  bake into the image.
* **Device disk is tight** — the base image alone is 14.5 GB against ~19 GB
  free. Log caps are in the stack for this reason.
* **Demo clips are not version-controlled.** `data/cv_demo_clips/` is
  gitignored (246 MB). A 252 MB source archive in that folder was deleted
  during this work; the clips exist on one machine and in the deployed images.
* **Single-device.** Everything here targets one device id. Fleet rollout would
  need per-device `DEVICE_ID`/`MQTT_TOPIC_BASE` templating.
