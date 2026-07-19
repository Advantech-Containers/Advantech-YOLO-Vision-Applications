# CV Demo Container — Overview

A five-minute orientation. For depth, go to:

| document | covers |
|:---------|:-------|
| **[cv-container-reference.md](cv-container-reference.md)** | Everything: images, config, MQTT contract, WEDA API, runbook, troubleshooting |
| **[training-guide.md](training-guide.md)** | Fine-tuning a detector for a clip COCO cannot handle |

---

## What it is

An unattended computer-vision demo for Advantech Jetson edge devices. It loops
a video clip forever, runs YOLO detection on the device GPU, renders annotated
output fullscreen on the device's own screen, and publishes detection results
to an MQTT broker running beside it. It is deployed and managed through the
**WEDA container-management API**.

```
+-- WEDA stack "yolo-od-demo" ---------------------------------+
|                                                              |
|   mqtt-broker             yolo-od-demo                       |
|   eclipse-mosquitto:2     yolo-od-demo:1.5.0                 |
|     :1883  <------------- publishes detections               |
|                             |                                |
|                             +--> X11 :0 --> device screen    |
+--------------------------------------------------------------+
              deployed to adlk.edgedevice.2 (74fe488d5d54)
```

## Current state

| | |
|:--|:--|
| device | `adlk.edgedevice.2` — EPC-R7300, Jetson Orin, JetPack 6.2 |
| stack | `yolo-od-demo` **v4** (`2201a204-630a-4f45-a893-70d0c9540704`) |
| image | `harbor.arfa.wise-paas.com/edge-coa/yolo-od-demo:1.5.0` |
| clip | `OD_bottle_2.mp4` — bottling line, 8 s, 2560x1440 |
| model | stock `yolo11n` @ conf 0.40 |
| measured | **100% frame coverage, 6.06 boxes/frame, ~12 FPS live** |
| telemetry | `advantech/74fe488d5d54/vision/#` |

Every detection in the deployed clip is `bottle` — one class, no false
positives anywhere across its 478 frames.

## Two demos

| demo | clip | model | training | coverage | boxes/frame |
|:-----|:-----|:------|:---------|---------:|------------:|
| **bottling** (deployed) | `OD_bottle_2` | stock `yolo11n` | none | **100%** | **6.06** |
| jar filling | `OD_Jar` | fine-tuned `jar26n` | 232 frames | 53% | 1.94 |

The bottling demo is better *and* free — COCO already knows `bottle`. The jar
demo needed fine-tuning because COCO has no `jar` class and labels the product
`cup`, the cable loom `surfboard`, and the filling nozzle `person`. Both
revisions remain deployable; see the rollback table in
[reference §14.3](cv-container-reference.md#143-roll-back).

## MQTT at a glance

Base `advantech/<DEVICE_ID>/vision`:

| topic | retained | payload |
|:------|:---------|:--------|
| `/status` | yes | `online` / `offline` (Last Will and Testament) |
| `/meta` | yes | model, clip, thresholds |
| `/detections` | no | every 1 s: counts, confidences, boxes |

```bash
mosquitto_sub -h 172.22.160.197 -t 'advantech/#' -v
```

Telemetry is best-effort: if the broker dies the demo keeps rendering. Full
schema in [reference §6](cv-container-reference.md#6-mqtt-telemetry-contract).

## Three things that will bite you

**1. The base image ships a headless OpenCV that shadows JetPack's.**
`cv2.imshow` raises *"rebuild the library with GTK+ 2.x"*. The fix is two
environment variables — `PYTHONPATH=/usr/lib/python3.10/dist-packages` **and**
`YOLO_AUTOINSTALL=false`. The second is not optional: without it ultralytics
reinstalls `opencv-python` at import and re-breaks the display.
[§4.1](cv-container-reference.md#41-opencv-python-headless-shadows-jetpacks-opencv)

**2. ultralytics 8.3.x silently mis-loads YOLO26.** No error — just ~22 junk
boxes per frame and zero true detections. Needs >= 8.4.
[§4.2](cv-container-reference.md#42-ultralytics-83x-silently-mis-loads-yolo26)

**3. A stalled WEDA deployment looks exactly like a healthy one.**
`status: deploying`, `error: {}`, forever. Diagnose from `dmagent` logs on the
device, not from the API. The fix is *two* steps — restart dmagent **and**
redeploy, because a delta missed while disconnected is never replayed.
[§13.4](cv-container-reference.md#134-a-stalled-deployment-is-indistinguishable-from-a-healthy-one)

## Layout

```
src/demo-od-loop.py             unattended looping runner
src/mqtt_publisher.py           MQTT telemetry publisher
docker/Dockerfile.demo-od*      image lineage 1.0.0 -> 1.5.0
docker/weda-stack-od-demo.yml   WEDA compose stack (broker + demo)
training/                       fine-tuning pipeline
docs/                           this document, reference, training guide
```

## Quick verification

```bash
# 1. WEDA agrees            -> status=running, containers[].state=running
curl --cacert weda-sit-k3s.pem -H "Authorization: Bearer $TOK" \
  "$BASE/api/v1/devices/74fe488d5d54/docker/stacks"

# 2. the demo is inferring  -> "12.0 FPS | 6.1 objects/frame | mqtt published=..."
ssh adlk.edgedevice.2 'docker logs --tail 20 edge_yolo-od-demo-yolo-od-demo-1'

# 3. it is on the screen
ssh adlk.edgedevice.2 'DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority scrot /tmp/s.png'

# 4. telemetry is flowing
ssh adlk.edgedevice.2 'timeout 8 docker run --rm --network host eclipse-mosquitto:2 \
  mosquitto_sub -h 127.0.0.1 -t "advantech/#" -v -W 6'
```

`status: running` alone proves nothing — a container can be up with a blank
screen. Check all four.

## Scope and honesty

The deployed bottling demo uses stock weights and a genuine COCO class, so it
generalises to other bottling footage. The `jar26n` model does **not** — it was
trained on 232 frames of one 9-second clip from pseudo-labels and is
deliberately specialised to it. Correct for a looping demo, wrong for
production inspection.

The MQTT broker is **unauthenticated and unencrypted**: acceptable for a
host-networked lab device and nothing else. Anonymous MQTT on a plant network
is an open write path into whatever consumes these topics.
[§16](cv-container-reference.md#16-security-posture)

This repository is GPL-3.0 and Ultralytics YOLO is AGPL-3.0 — shipping this
image to a customer is a distribution event with copyleft obligations.
