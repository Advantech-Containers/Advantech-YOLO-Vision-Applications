# YOLO Vision Edge Demo — Build, Fine-Tune, Deploy, Publish

**Target device:** `adlk.edgedevice.2` — Advantech EPC-R7300, Jetson Orin, JetPack 6.2 (R36.4.4), WEDA device id `74fe488d5d54`
**Deployed via:** WEDA container-management API, org `Kevin.Chien@advantech.com.tw`, tenant `central`, cluster `weda-sit-k3s.weda.dev`
**Status:** live — stack `yolo-od-demo` **v4**, image `harbor.arfa.wise-paas.com/edge-coa/yolo-od-demo:1.5.0`

An unattended object-detection demo: a looping video plays fullscreen on the
device's own screen with live YOLO detections drawn over it, and detection
results are published to an on-device MQTT broker (§7).

Two demos are documented here, both deployable from the same stack config:

| demo | clip | model | fine-tuning | quality |
|:-----|:-----|:------|:------------|:--------|
| **bottling line** (deployed) | `OD_bottle_2.mp4` | stock `yolo11n` | none needed | 100% coverage, 6.06 boxes/frame |
| jar filling line | `OD_Jar.mp4` | fine-tuned `jar26n` | 232 pseudo-labelled frames | 53% coverage, 1.94 boxes/frame |

The bottling demo is the better one and needs no training — COCO already knows
`bottle`. The jar demo required fine-tuning because COCO has no `jar` class and
mislabels the product; §5 documents that pipeline, which is the reusable part
for any clip whose subject COCO does not know.

---

## 1. Results

*Coverage* = share of frames with at least one box; it is the number that
decides whether the demo looks alive. **Measure over the whole clip** — an
early sequential window can be far denser than the rest and will flatter the
result (measuring only the first 150 of 495 frames of the sushi clip reported
100% where the true figure was 73%).

**Deployed demo — `OD_bottle_2.mp4`, all 478 frames:**

| model | conf | coverage | boxes/frame | FPS |
|:------|-----:|---------:|------------:|----:|
| **`yolo11n` (deployed)** | **0.40** | **100%** | **6.06** | **23.8** |
| `yolo26n` | 0.40 | 100% | 4.97 | 22.4 |
| `yolo11l` | 0.40 | 100% | 7.26 | 14.3 |

Class split at the deployed setting: `bottle` x2898 (6.06/frame, max 0.95,
mean 0.75, min 0.40) — a single class, no false positives anywhere in the clip.
Box count is stable at 5-8 per frame, which is why the overlay does not strobe.
`yolo26n` is also baked into the image; switch via `MODEL_PATH`, no rebuild.

**Jar demo — `OD_Jar.mp4`, for comparison:**

| model | conf | coverage | boxes/frame | FPS |
|:------|-----:|---------:|------------:|----:|
| stock `yolo11n` | 0.25 | 0% | 0.00 | 24.6 |
| stock `yolo26n` | 0.20 | 21% | 0.31 | 22.6 |
| `jar26n` fine-tuned | 0.40 | 53% | 1.94 | 23.3 |

Validation of the fine-tuned model: **mAP50 0.972, mAP50-95 0.899, P 0.934, R 0.918.**
Its quality is flat between conf 0.25 and 0.50 (55% → 53% coverage), the
signature of a properly calibrated model — 0.40 buys a cleaner overlay at no
measured cost in recall.

### Why stock models failed

COCO has no `jar` class. On this footage stock `yolo11n` produced 5.2
`surfboard` detections per frame (the red cable loom) and labelled the filling
nozzle `person`, while `bottle` — the closest real class — never exceeded 0.30
confidence. Raising the confidence threshold made it *worse*: it removed every
jar and kept the surfboards.

---

## 2. Image lineage

Each image is built **on the device** (native arm64). Building on an x86 host
under qemu was abandoned: the base image pull ran at 0.12 MB/s from the VM
versus ~8 MB/s from the device, a ~65x difference.

```
edgesync.azurecr.io/advantech/advantech-yolo-vision-applications:1.6.0-Ubuntu22.04-ARM
  │  (vendor base, 14.5 GB, ultralytics 8.3.220)
  ├─ 1.0.0        docker/Dockerfile.demo-od       COPY-only: clip, weights, runner, entrypoint
  ├─ 1.2.0-ul84   docker/Dockerfile.demo-od-ul84  upgrade ultralytics -> 8.4.101 (YOLO26 support)
  └─ 1.3.0        docker/Dockerfile.demo-od-jar   + fine-tuned jar26n.pt, conf 0.40   <-- DEPLOYED
```

`1.0.0` deliberately contains **no `RUN` instruction** — every layer is a
`COPY`, so nothing executes under emulation and the image builds in under a
second. `1.2.0-ul84` is the only stage that must run `pip`.

### Two traps baked into the base image

**a) `opencv-python-headless` shadows JetPack's OpenCV.** The base image
carries two OpenCVs:

| package | path | GUI | GStreamer |
|:--------|:-----|:----|:----------|
| pip `opencv-python-headless` 4.11.0.86 | `/usr/local/lib/python3.10/dist-packages` ← wins | **NONE** | NO |
| apt `libopencv-python` 4.8.0 (JetPack) | `/usr/lib/python3.10/dist-packages` | GTK2 | 1.20.3 |

`/usr/local` precedes `/usr/lib` on `sys.path`, so `cv2.imshow` raises
*"The function is not implemented. Rebuild the library with GTK+ 2.x"*. This
means the `--show` flag of `src/advantech-yolo.py` has never worked inside this
container, contrary to the main README.

Fix, no rebuild required:

```
PYTHONPATH=/usr/lib/python3.10/dist-packages
YOLO_AUTOINSTALL=false
```

The second variable is **not optional** — without it ultralytics pip-installs
`opencv-python` at import time and silently re-shadows the fix.

**b) ultralytics 8.3.x silently mis-loads YOLO26.** A YOLO26 checkpoint loaded
under 8.3.220 is coerced into the legacy `Detect` head with `end2end=False`.
It loads *without error* and emits ~22 junk boxes per frame with zero true
detections. Verify explicitly:

```python
h = YOLO('yolo26n.pt').model.model[-1]
assert getattr(h, 'end2end', False), 'wrong ultralytics for YOLO26'
```

Both Dockerfiles assert these conditions at build time, so a regression fails
the build rather than the demo.

---

## 3. Building

```bash
# On the device (native arm64, no qemu, base image already local)
cd /home/ubuntu/yolo-od-demo
docker build -f docker/Dockerfile.demo-od      -t harbor.arfa.wise-paas.com/edge-coa/yolo-od-demo:1.0.0 .
docker build -f docker/Dockerfile.demo-od-ul84 -t harbor.arfa.wise-paas.com/edge-coa/yolo-od-demo:1.2.0-ul84 .
docker build -f docker/Dockerfile.demo-od-jar  -t harbor.arfa.wise-paas.com/edge-coa/yolo-od-demo:1.3.0 .
```

### Publishing to Harbor

The device holds no `harbor.arfa.wise-paas.com` credential and pulls the base
image from a slow link, so the image is streamed to a credentialled host over
the LAN (~99 MB/s) and pushed from there:

```bash
ssh adlk.edgedevice.2 'docker save harbor.arfa.wise-paas.com/edge-coa/yolo-od-demo:1.3.0' | docker load
docker push harbor.arfa.wise-paas.com/edge-coa/yolo-od-demo:1.3.0
```

This avoids copying a registry credential onto a shared edge device.

---

## 4. The demo runner

`src/demo-od-loop.py` exists because `src/advantech-yolo.py` cannot run
unattended:

* it blocks on `input()` with no TTY attached, and
* its predict loop uses `stream=True`, which **terminates at video EOF** — there is no looping.

The runner keeps the model resident and rewinds the capture in place
(`cap.set(CAP_PROP_POS_FRAMES, 0)`); reopening the file or reloading the model
would stall the demo for seconds on every lap. It also:

* logs an FPS/detection summary on an interval, never per frame — the device
  root filesystem sits above 80% and unbounded `json-file` logs would fill it;
* fails loudly on missing display, model, video, or a headless OpenCV, so a
  broken deployment cannot masquerade as a healthy container;
* handles `SIGTERM` so `docker stop` is graceful.

`docker/entrypoint-demo.sh` waits up to 120s for the X display before starting,
so a device reboot does not put the container into a crash loop under
`restart: unless-stopped`.

### Reaching the device screen

X runs on `:0` under gdm. The live cookie is at `/run/user/1000/gdm/Xauthority`
— **not** `~/.Xauthority`, which is stale. The stack therefore hardcodes
`DISPLAY=:0` and mounts both the socket and the gdm cookie; a WEDA-deployed
container has no user session to inherit `$DISPLAY`/`$XAUTHORITY` from.

---

## 5. Fine-tuning

### 5.1 Pseudo-labels without human annotation

`OD_Jar.mp4` is 232 frames (9.3 s at 25 fps) of 2560x1440. The demo loops that
one file forever, so a model specialised to it is exactly what is wanted.

The single biggest lever was **teacher resolution**: at the default 640px the
2560-wide source shrinks the jars past detectability (0.43 boxes/frame); at
1536px the same model yields 2.56.

Pipeline (`make_dataset.py`):

1. **Teacher** — `yolo11l`, `imgsz=1536`, `conf=0.10`, restricted to COCO
   container classes `[39, 40, 41, 45, 75]` (bottle, wine glass, cup, bowl, vase).
   Note `vase` is id **75**, not 46.
2. **Associate** detections into tracks by greedy IoU (threshold 0.25, a track
   survives 12 unmatched frames). Ultralytics' BoT-SORT needs the `lap`
   extension, absent here and a source build on aarch64.
3. **Interpolate** linearly across gaps up to 25 frames. Jars move predictably
   on a conveyor, so an interpolated box beats a teacher miss. This is what
   lets the student exceed the teacher.
4. **Filter** tracks shorter than 4 frames (flicker).
5. **NMS** per frame at IoU 0.55 — the teacher labels one jar as both
   `wine glass` and `cup`, producing stacked duplicate boxes.

Result: **232 frames, 509 boxes, 68% coverage** — from a teacher that never
achieved 40% per-frame.

> **Always render a sample of the labels and look at them before training.**
> The duplicate-box defect was invisible in the summary statistics and obvious
> in a six-frame contact sheet.

### 5.2 Training

**Train on `acn.air-520` (172.16.9.80). Do not train on the Orin.**

| host | hardware | 120 epochs @ 960px |
|:-----|:---------|:-------------------|
| `acn.air-520` | 2x RTX 6000 Ada, 251 GB RAM, 64 cores | **15.6 minutes** |
| `adlk.edgedevice.2` | Jetson Orin, 7 GB shared RAM | ~15 min **per epoch**; device became unreachable |

Training on the Orin with the demo running drove load average to 74 and left
`sshd` unable to fork — the device was unreachable for several minutes. It
recovered, but the edge device is the wrong machine for this.

Both air-520 GPUs are typically ~93% occupied by an unrelated vLLM server
(~2.9 GB free each). A nano model at `batch=4` fits in 1.6 GB and coexists —
**do not evict that service.**

```bash
scp jar-dataset.tgz acn.air-520:~/jar-train/
ssh acn.air-520 'docker run --rm --runtime nvidia --gpus all --shm-size 8g \
  -v ~/jar-train:/data --entrypoint python3 ultralytics/ultralytics:latest /data/train520.py'
```

Settings: `yolo26n.pt`, 120 epochs, `imgsz=960`, `batch=4`, `patience=40`,
single class `jar`.

> **A partially-trained checkpoint looks trained and detects nothing.** At 2
> epochs the model reported mAP50 0.47 but its maximum confidence was 0.026 —
> mAP is rank-based, so box placement scores well long before the
> classification head is calibrated. Confidence needs roughly 100 epochs.
> Never judge a run by mAP alone; check the actual confidence distribution.

---

## 6. Deployment via the WEDA container-management API

Base URL `https://weda-sit-k3s.weda.dev/central/weda` (self-signed cert — pin it
with `--cacert` rather than using `-k`). Mint a JWT with the `weda-login`
skill; tokens last 2 hours.

| step | call |
|:-----|:-----|
| registry credential | already existed as `harbor-arfa-edge-coa` — reuse, don't duplicate |
| create stack | `POST /api/v1/orgs/{orgId}/stack-configs` (compose base64 in `composeFileContent`) |
| update stack | `PATCH …/stack-configs/{stackConfigId}` → new immutable revision |
| deploy | `POST …/revisions/{stackRevisionId}:deploy` `{"deviceIds":["74fe488d5d54"]}` |
| status | `GET /api/v1/devices/{deviceId}/docker/stacks` |

The compose file (`docker/weda-stack-od-demo.yml`) is deliberately conservative:
`runtime: nvidia` is omitted (the daemon's default runtime already is nvidia)
and the explicit `devices:` list is omitted (`privileged: true` covers it), to
minimise fields WEDA's passthrough might reject. No `container_name` is set —
WEDA derives its own (`edge_yolo-od-demo-yolo-od-demo-1`), and pinning one
collides with existing containers.

### 6.1 A stalled deployment is indistinguishable from a healthy one

A deployment can sit at `status: deploying`, `containers: []`, `active: true`,
`error: {}` **forever**. The API returns `HTTP 200` and surfaces no error.
Four deployments stalled this way before the cause was found.

**Diagnose on the device**, not through the API:

```bash
docker logs dmagent | grep -E "stack_async|composeplugin|Reported stacks"
```

If the last entry predates the deploy, the agent never received it. Outbound
telemetry (`cfgstate_handler … published config shadow state`) keeps working
throughout, which is why the device still looks healthy in WEDA.

Confirm it is not your compose by deploying a trivial known-good stack to the
same device. If that stalls too, the fault is the agent.

**Fix — both steps, in order:**

1. `docker restart dmagent` — resubscribes to the cloud delta topics.
2. **Delete the stack from the device and deploy again.**

Step 2 is essential and easy to miss: dmagent subscribes to *delta* topics, and
a delta published while it was disconnected is never replayed. The pending
deployment stays stuck even after a successful restart — only a fresh deploy
lands. After both steps the container appeared in under 30 seconds.

### 6.2 `awaitingCleanup` does not self-resolve

During the v1.3.0 rollout, delete-then-deploy left the deployment in
`awaitingCleanup` with dmagent logging `context deadline exceeded before
applying desired config state`. It sat there for 10+ minutes with the demo
down. Recovery:

```bash
curl -X DELETE …/api/v1/devices/{deviceId}/stacks/deployments   # clear all
curl -X POST   …/revisions/{stackRevisionId}:deploy            # fresh deploy
```

The container came up 30 seconds later. Budget for a short outage on any
version bump — the old container is removed before the new one is scheduled.

---

## 7. MQTT detection telemetry

The stack runs an `eclipse-mosquitto:2` broker beside the demo and publishes
detection results to it. Both services use host networking, so the broker is
reachable on the device at **`127.0.0.1:1883`** (and on the device's LAN
address from elsewhere).

`mosquitto 2.x` refuses remote connections under its default config, so the
service starts with `-c /mosquitto-no-auth.conf` — a stock file inside the
image that enables an anonymous listener on 1883. This avoids bind-mounting a
config file, which a WEDA-deployed stack cannot easily supply.

### 7.1 Topics

Base defaults to `advantech/<DEVICE_ID>/vision`, overridable via
`MQTT_TOPIC_BASE`. On the deployed device that resolves to
`advantech/74fe488d5d54/vision`.

| topic | retained | QoS | when | payload |
|:------|:---------|:----|:-----|:--------|
| `<base>/status` | yes | 1 | on connect / on death | `online` \| `offline` |
| `<base>/meta` | yes | 1 | once, on connect | JSON, run configuration |
| `<base>/detections` | no | `MQTT_QOS` (0) | every `MQTT_INTERVAL_SEC` (1 s) | JSON, detection summary |

`<base>/status` carries a **Last Will and Testament**: the broker publishes
`offline` automatically if the container dies without a clean shutdown. A
subscriber can therefore distinguish a dead publisher from one that simply has
nothing to report — with a plain `online` flag those two states look identical.

Both retained topics mean a subscriber that connects late immediately learns
the device state and run configuration without waiting for the next event.

### 7.2 `detections` message schema

```jsonc
{
  "timestamp":   "2026-07-19T16:54:47.949Z",  // ISO-8601 UTC, millisecond precision
  "deviceId":    "74fe488d5d54",              // WEDA device id
  "frame":       830,                          // cumulative frame counter since start
  "lap":         1,                            // how many times the clip has looped
  "fps":         12.17,                         // measured render+inference rate
  "objectCount": 7,                            // total boxes in this frame
  "classCounts": { "bottle": 7 },              // count per class
  "detections": [                              // omitted when MQTT_INCLUDE_BOXES=false
    {
      "class":      "bottle",
      "confidence": 0.8932,                    // 0.0-1.0, rounded to 4 dp
      "bbox":       [0.5, 116.1, 344.9, 1417.2] // [x1, y1, x2, y2]
    }
  ]
}
```

**`bbox` is in source-image pixels, not normalised** — for `OD_bottle_2.mp4`
the frame is 2560x1440, so coordinates range over that. Consumers expecting
0-1 must divide by the frame dimensions, which are not currently in the
payload; take them from `<base>/meta`'s `source` or hard-code per clip.

`classCounts` is the field to chart or alarm on. `detections` is the heavy part
of the payload — set `MQTT_INCLUDE_BOXES=false` to drop it and cut the message
to roughly a tenth of its size when only counts matter.

### 7.3 `meta` message schema

```jsonc
{
  "model":         "yolo11n.pt",       // weights actually loaded
  "source":        "OD_bottle_2.mp4",  // clip being looped
  "confThreshold": 0.4,
  "iouThreshold":  0.45,
  "demoVersion":   "1.1.0"             // runner version, not image tag
}
```

### 7.4 Configuration

| variable | default | purpose |
|:---------|:--------|:--------|
| `MQTT_ENABLED` | `true` | set `false` to run the demo with no telemetry |
| `MQTT_HOST` | `127.0.0.1` | broker address |
| `MQTT_PORT` | `1883` | broker port |
| `MQTT_QOS` | `0` | QoS for `detections` (status/meta are always QoS 1) |
| `MQTT_INTERVAL_SEC` | `1.0` | publish interval |
| `MQTT_INCLUDE_BOXES` | `true` | include per-detection boxes |
| `MQTT_TOPIC_BASE` | `advantech/<DEVICE_ID>/vision` | topic prefix |
| `DEVICE_ID` | container hostname | identity in payloads and default topic |

Detections are published on an **interval, not per frame**: at ~12 FPS a
per-frame topic emits 12 msg/s of near-identical payloads for no analytical
gain. Rate limiting lives inside the publisher so the render loop stays free of
telemetry concerns.

### 7.5 Failure behaviour

**Telemetry is best-effort and never fatal.** Every publisher method degrades
to a no-op when the broker is unreachable, so the demo keeps rendering to the
screen — losing telemetry must not take the display down. The client
auto-reconnects with backoff (1 s to 30 s). Failures are logged with context,
never silently swallowed, and the per-interval demo log carries a counter:

```
[demo] 11.2 FPS | 6.0 objects/frame | laps=1 | mqtt published=52 dropped=0 connected=True
```

`dropped` incrementing while `connected=True` means the broker is accepting the
connection but rejecting or dropping publishes — check broker logs and QoS.

### 7.6 Subscribing

```bash
# from the device (or anywhere with a route to it)
docker run --rm --network host eclipse-mosquitto:2 \
  mosquitto_sub -h 127.0.0.1 -p 1883 -t 'advantech/#' -v

# counts only, formatted
mosquitto_sub -h <device-ip> -t 'advantech/+/vision/detections' \
  | jq -c '{ts: .timestamp, n: .objectCount, cls: .classCounts}'
```

> **Security.** The broker is **unauthenticated and unencrypted**. That is
> acceptable for a host-networked demo on a lab device and nothing else. Any
> deployment carrying real process data needs TLS and credentials per
> IEC 62443 — anonymous MQTT on a plant network is an open write path into
> whatever consumes these topics.

---

## 8. Verifying a deployment

Never trust `status: running` alone; the container can be up with a blank
screen. Check all three:

```bash
# 1. WEDA agrees
curl …/api/v1/devices/74fe488d5d54/docker/stacks     # status=running, containers[].state=running

# 2. the demo is actually inferring
ssh adlk.edgedevice.2 'docker logs --tail 20 edge_yolo-od-demo-yolo-od-demo-1'
#    expect: "OpenCV 4.8.0 (GUI enabled)", "CUDA available: Orin",
#            "11.4 FPS | 2.3 objects/frame | laps=2"

# 3. it is on the screen
ssh adlk.edgedevice.2 'DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority scrot /tmp/s.png'
scp adlk.edgedevice.2:/tmp/s.png .
```

A `PRECONDITION CHECK FAILED` block in the logs names the exact cause — the
runner fails fast rather than leaving a dark screen under a healthy container.

---

## 9. Limitations

* **The `jar26n` model is overfit to one 9-second clip by design.** Trained on
  232 frames of `OD_Jar.mp4` from pseudo-labels, it will **not** generalise to
  another line, jar, camera angle, or lighting. Correct for a looping demo;
  wrong for production inspection. Any product claim needs real labelled data
  across varied conditions. (The deployed bottling demo does not have this
  problem — it uses stock weights and a real COCO class.)
* **Labels are pseudo-labels**, inherited from a COCO teacher and temporally
  completed. They were visually spot-checked, not verified frame by frame.
* **53% coverage is not 100%.** Part of that is genuine — stretches of the clip
  have no jar near the camera — but the figure has not been separated into
  "no jar present" versus "jar missed".
* **The MQTT broker is unauthenticated and unencrypted** (§7). Lab demo only.
* **`bbox` values are source-image pixels, not normalised**, and the frame
  dimensions are not carried in the payload.
* **No TensorRT.** Everything runs PyTorch. An on-device `.engine` export
  typically gives 2-3x on Orin. Engines are hardware- and TRT-version-specific,
  so they must be built on the target and cannot be baked into the image.
* **Device disk is tight** — the base image alone is 14.5 GB against ~19 GB
  free. Log caps (`max-size: 10m`, `max-file: 3`) are in the stack for this
  reason.
* **Licensing.** This repository is GPL-3.0 and Ultralytics YOLO is AGPL-3.0.
  Shipping this image to a customer is a distribution event with copyleft
  obligations and needs legal sign-off.

---

## 10. File map

| path | purpose |
|:-----|:--------|
| `src/demo-od-loop.py` | unattended looping detection runner |
| `src/mqtt_publisher.py` | MQTT detection publisher (topics, schema, LWT) |
| `docker/Dockerfile.demo-od` | 1.0.0 — COPY-only demo image |
| `docker/Dockerfile.demo-od-ul84` | 1.2.0-ul84 — ultralytics 8.4 for YOLO26 |
| `docker/Dockerfile.demo-od-jar` | 1.3.0 — fine-tuned `jar26n` (jar demo) |
| `docker/Dockerfile.demo-od-bottle` | 1.4.0 — `OD_bottle_2` on stock weights |
| `docker/Dockerfile.demo-od-mqtt` | 1.5.0 — adds MQTT telemetry **(deployed)** |
| `docker/entrypoint-demo.sh` | waits for X, then starts the runner |
| `docker/weda-stack-od-demo.yml` | WEDA compose stack — broker + demo |
| `docs/jar-detection-demo.md` | this document |

Training scripts (`make_dataset.py`, `train520.py`, evaluation sweeps) live on
the device under `/home/ubuntu/yolo-od-demo/` and on `acn.air-520` under
`~/jar-train/`; they are not yet in the repository.
