# SOP — Build a YOLO container image on the edge device and push it

**Applies to:** `harbor.arfa.wise-paas.com/edge-coa/yolo-od-demo` and any image in
the same lineage.
**Audience:** anyone shipping a new demo image tag to a WEDA-managed Jetson.
**Companion:** [sop-run-on-edge-device.md](sop-run-on-edge-device.md) (deploying
and running it), [cv-container-reference.md](cv-container-reference.md) §3
(lineage), §12 (build/publish), §13 (WEDA deployment).

> **Every command in this SOP runs in a shell on the edge device** — sit at it,
> or `ssh adlk.edgedevice.2` (172.22.160.197, user `ubuntu`) first.

---

## 0. Why the build happens on the device

The vendor base image is **14.5 GB of arm64**. Pulling it into an x86 build VM
measured **0.12 MB/s** against **~8 MB/s** on the device — a ~65x difference —
and any `RUN` step would then execute under qemu emulation. Cross-building was
tried and abandoned. Every image in this lineage is built **natively on the
Jetson**, where a COPY-only layer completes in under a second.

The device is also the only place where the build-time assertions mean
anything: they check JetPack's CUDA torch, the GTK OpenCV, and the GPU — none of
which exist off the device.

---

## 1. Prerequisites

| Requirement | Check |
|:---|:---|
| Architecture | `docker info --format '{{.Architecture}}'` → `aarch64` |
| Docker daemon | `docker ps` succeeds without sudo |
| **Vendor base present** | `docker images edgesync.azurecr.io/advantech/advantech-yolo-vision-applications` — it is 14.5 GB and is what `Dockerfile.demo-od` builds `FROM`. If absent, pull it here (it is already the parent of every demo image on this device) |
| Existing demo tags (if you start from one) | `docker images harbor.arfa.wise-paas.com/edge-coa/yolo-od-demo` |
| Disk headroom | `df -h /` — root sits at ~88 %. Below ~5 % free, `docker image prune` before you start |
| Registry reachable | `curl -sI https://harbor.arfa.wise-paas.com/v2/ \| head -1` |

Set the target once, in the shell you will build from:

```bash
export REGISTRY=harbor.arfa.wise-paas.com
export PROJECT=edge-coa
export IMAGE=yolo-od-demo
export TAG=1.7.0                     # the NEW tag you are building
export REF="$REGISTRY/$PROJECT/$IMAGE:$TAG"
export CTX=/home/ubuntu/yolo-od-demo # build context on this device
```

To publish somewhere else, change `REGISTRY`/`PROJECT` and log in to that
registry instead. Nothing else in the procedure changes.

---

## 2. Choose your starting point

`docker/Dockerfile.demo-od` is the worked example this SOP follows. It is the
`1.0.0` layer: it sits directly on the vendor base and adds nothing but your
weights, your clip, your runner and an entrypoint. **That is the template for
building your own image.**

```
edgesync.azurecr.io/advantech/advantech-yolo-vision-applications:1.6.0-Ubuntu22.04-ARM
  │  vendor base — 14.5 GB, ultralytics 8.3.220, torch 2.5.0 (JetPack)
  ├─ 1.0.0        Dockerfile.demo-od         clip + weights + runner + entrypoint (COPY-only)
  │                                          <-- YOUR IMAGE STARTS HERE
  ├─ 1.2.0-ul84   Dockerfile.demo-od-ul84    ultralytics -> 8.4.101 (YOLO26 support)
  ├─ 1.3.0        Dockerfile.demo-od-jar     + fine-tuned jar26n.pt
  ├─ 1.4.0        Dockerfile.demo-od-bottle  + OD_bottle_2.mp4 + yolo26n.pt
  ├─ 1.5.0        Dockerfile.demo-od-mqtt    + paho-mqtt + telemetry publisher
  └─ 1.6.0        Dockerfile.demo-od-window  windowed UI    <-- CURRENTLY DEPLOYED
```

| You want | Start from | Template |
|:---|:---|:---|
| Your own clip + your own weights, self-contained | the **vendor base** | `Dockerfile.demo-od` (§4) |
| The same, but with a YOLO26 model | `1.2.0-ul84` | `Dockerfile.demo-od-bottle` |
| Your own clip + weights **and** MQTT telemetry | `1.5.0` | `Dockerfile.demo-od-bottle` |
| A tweak to the shipped demo (a knob, a code fix) | `1.6.0` | `Dockerfile.demo-od-window` |

**Two things decide this for you:**

* **Is your model YOLO26?** The vendor base ships **ultralytics 8.3.220**, which
  silently coerces a YOLO26 checkpoint into the legacy `Detect` head — it loads
  without error and emits ~22 junk boxes per frame. If your weights are YOLO26,
  start from `1.2.0-ul84` or later, or add the 8.4 upgrade `RUN` from
  `Dockerfile.demo-od-ul84` to your own file. YOLO11 weights are fine on the base.
* **Do you need MQTT?** `paho-mqtt` and `src/mqtt_publisher.py` arrive at `1.5.0`.
  Starting from the vendor base gives you the screen output only.

**Before writing anything, check whether you need a new image at all.**
`MODEL_PATH`, `VIDEO_PATH`, `CONF_THRESHOLD`, `FULLSCREEN`, `WINDOW_*` and every
`MQTT_*` knob are environment variables read at startup
(`src/demo-od-loop.py:39-54`). Swapping the model or the clip for one already
baked into an existing image is a **stack edit, not a rebuild** — see
[sop-run-on-edge-device.md](sop-run-on-edge-device.md) §5.1.

---

## 3. Assemble the build context

```bash
git clone https://github.com/Advantech-EdgeSync-Containers/Advantech-YOLO-Vision-Applications.git "$CTX"
cd "$CTX"
mkdir -p models data/cv_demo_clips
```

**The clone is not a complete build context.** Model weights (`models/`, `*.pt`)
and demo clips (`data/cv_demo_clips/`) are in `.gitignore` — they are large
binaries that must not enter git history — so the `COPY` lines cannot be
satisfied from git alone.

**Your own model and clip:** copy them into `models/` and `data/cv_demo_clips/`
on this device by whatever means you have (`scp`, a USB stick, the training
host). These are the two files that make the image *yours*.

**Advantech's stock weights and clips**, if you want them as a starting point or
a fallback, are already baked into the published images — pull them out rather
than re-downloading:

```bash
docker create --name ctx-src "$REGISTRY/$PROJECT/$IMAGE:1.6.0"
docker cp ctx-src:/advantech/models/yolo11n.pt         models/
docker cp ctx-src:/advantech/models/yolo26n.pt         models/
docker cp ctx-src:/advantech/data/OD_bottle_2.mp4      data/cv_demo_clips/
docker rm ctx-src
```

Confirm every file your Dockerfile `COPY`s is present before building. A missing
weight fails the build late, after the context has been read:

```bash
ls -la models data/cv_demo_clips src docker
```

---

## 4. Write your Dockerfile, using `docker/Dockerfile.demo-od` as the template

Copy it and edit the four marked lines:

```bash
cp docker/Dockerfile.demo-od docker/Dockerfile.my-demo
```

```dockerfile
FROM edgesync.azurecr.io/advantech/advantech-yolo-vision-applications:1.6.0-Ubuntu22.04-ARM

LABEL maintainer="Advantech Center of Excellence" \
      vendor="Advantech" \
      version="1.0.0" \                    # <-- CHANGE: match your tag
      description="YOLO11 looping object-detection demo for WEDA-managed Jetson devices"

ENV PYTHONPATH=/usr/lib/python3.10/dist-packages \   # keep — see below
    PYTHONUNBUFFERED=1 \
    YOLO_AUTOINSTALL=false \                          # keep — see below
    YOLO_OFFLINE=true \
    VIDEO_PATH=/advantech/data/OD_Jar.mp4 \           # <-- CHANGE: your clip
    MODEL_PATH=/advantech/models/yolo11n.pt \         # <-- CHANGE: your weights
    CONF_THRESHOLD=0.25 \
    INFER_DEVICE=0 \
    FULLSCREEN=true \
    STATS_INTERVAL_SEC=60

WORKDIR /advantech

COPY models/yolo11n.pt              /advantech/models/yolo11n.pt        # <-- CHANGE
COPY data/cv_demo_clips/OD_Jar.mp4  /advantech/data/OD_Jar.mp4          # <-- CHANGE
COPY src/demo-od-loop.py            /advantech/src/demo-od-loop.py
COPY docker/entrypoint-demo.sh      /advantech/entrypoint-demo.sh

ENTRYPOINT ["/advantech/entrypoint-demo.sh"]
```

Four edits: the `version` label, `VIDEO_PATH`, `MODEL_PATH`, and the two `COPY`
lines that place them. The `COPY` destination must match the `ENV` path exactly —
a mismatch produces `PRECONDITION CHECK FAILED: model not found` at runtime, not
at build time.

### Why each convention is there

| Line | Why it is not optional |
|:---|:---|
| `PYTHONPATH=/usr/lib/python3.10/dist-packages` | The base ships `opencv-python-headless` in `/usr/local`, which shadows JetPack's GTK-enabled OpenCV and makes `cv2.imshow` raise. This puts the JetPack build first without uninstalling anything. |
| `YOLO_AUTOINSTALL=false` | Without it ultralytics reinstalls `opencv-python` at import and re-shadows the fix. Both variables, or neither works. |
| `YOLO_OFFLINE=true` | The device has no reliable internet egress; ultralytics must never try to fetch weights at runtime. |
| `PYTHONUNBUFFERED=1` | Otherwise `docker logs` shows nothing until the buffer flushes — and the log is how you verify the demo (§6). |
| Weights and clip **baked in**, not mounted | A WEDA-deployed container has no host paths to bind; anything not in the image is not there. |
| `ENTRYPOINT` is `entrypoint-demo.sh`, not python | It waits up to 120 s for gdm to bring X up. Without the wait, a container started at boot exits on a missing `DISPLAY` and, under `restart: unless-stopped`, crash-loops. |

### Deliberately no `RUN`

Every layer in this file is a `COPY`. Nothing executes during the build, so it
completes in under a second and could in principle be assembled anywhere. Keep
it that way if you can.

If you do need one — a pip package, an ultralytics upgrade — add it, and add
these two safeguards.

**1. Constrain pip.** It will happily resolve a PyPI `torch` over JetPack's CUDA
build and silently drop the demo to CPU:

```dockerfile
RUN python3 -c "import torch, torchvision, numpy; \
      open('/tmp/c.txt','w').write(f'torch=={torch.__version__}\ntorchvision=={torchvision.__version__}\nnumpy=={numpy.__version__}\n')" \
 && pip3 install --no-cache-dir -c /tmp/c.txt "<package>==<version>" \
 && rm -f /tmp/c.txt
```

**2. Assert, so a regression fails the build rather than the demo.** Every
derived Dockerfile in `docker/` ends with a block like this:

```dockerfile
RUN python3 -c "\
import torch, cv2; \
assert torch.__version__.endswith('nv24.08'), 'JetPack torch was replaced'; \
assert 'GTK' in cv2.getBuildInformation(), 'OpenCV lost GUI support'; \
print('CHECKS PASSED')"
```

Add to it according to what your layer touches:

| If your layer touches… | Assert |
|:---|:---|
| a YOLO26 checkpoint | `getattr(model.model.model[-1], 'end2end', False)` is `True` |
| a fine-tuned model | the exact class map, e.g. `m.names == {0: 'jar'}` |
| a demo clip | `cv2.VideoCapture(path)` reports `CAP_PROP_FRAME_COUNT > 0` |
| ultralytics | `ultralytics.__version__.startswith('8.4')` |
| MQTT | `from mqtt_publisher import DetectionPublisher, MQTT_AVAILABLE` |
| the window UI | `'resizeWindow' in src and 'moveWindow' in src` |

Commit your Dockerfile back to the repo once the build passes — the lineage is
only reproducible if the file that produced each tag is in git.

---

## 5. Build

```bash
cd "$CTX"
docker build -f docker/Dockerfile.my-demo -t "$REF" .
```

A COPY-only image completes in **under a second** — the 14.5 GB base is already
on the device and nothing executes. A layer with `RUN pip` takes a few minutes.
If you added assertions, the build must end with your `CHECKS PASSED` line.


---

## 6. Smoke-test before pushing

A green build proves the image is *internally* consistent. This proves it runs.

```bash
docker run --rm --runtime nvidia "$REF" python3 -c "
import torch, cv2, os
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('cv2', cv2.__file__)
print('MODEL_PATH', os.environ.get('MODEL_PATH'))
assert torch.cuda.is_available(), 'no GPU in container'"
```

Expect `cuda True` and a `cv2` resolved under `/usr/lib/python3.10/dist-packages`
(JetPack's GTK build) — **not** `/usr/local`. If it resolves to `/usr/local`,
`PYTHONPATH` was lost and `cv2.imshow` will fail at runtime with the GTK error.

Confirm the architecture:

```bash
docker image inspect "$REF" --format '{{.Os}}/{{.Architecture}}'   # linux/arm64
```

---

## 7. Push to the registry

### 7.1 Log in, push, log out

```bash
docker login "$REGISTRY"          # prompts for username / password
docker push "$REF"
docker logout "$REGISTRY"         # do not leave the credential behind
```

> **Log out when you are done.** `docker login` writes the credential
> **base64-encoded, not encrypted**, to `~/.docker/config.json`, where it
> survives reboots. On a shared edge device that is a standing registry
> credential for anyone with shell or root on the box.

### 7.2 Verify what landed in the registry — do not skip this

An amd64 image deploys cleanly through WEDA and then fails on the device with
`exec format error`:

```bash
docker manifest inspect -v "$REF" | grep -E '"architecture"|"os"'
# expect: "architecture": "arm64",  "os": "linux"
```

### 7.3 Never re-push an existing tag

WEDA stack revisions pin a tag, not a digest. Overwriting a tag changes what
already-deployed devices pull on their next restart, with no revision history to
roll back to. Always allocate a new tag.

---

## 8. Record and deploy

1. **Add the tag to the lineage** in
   [cv-container-reference.md](cv-container-reference.md) §3 and move the
   `<-- CURRENTLY DEPLOYED` marker.
2. **Bump the image** in `docker/weda-stack-od-demo.yml` (line 42) and any env
   knobs the new layer expects.
3. **Deploy and verify** — see
   [sop-run-on-edge-device.md](sop-run-on-edge-device.md) §2 and §4. Clear all
   deployments first, then deploy; and check all four verification steps, not
   just WEDA's `status: running`.

### Rollback

Point the stack back at the previous tag and redeploy. The old image is still in
the registry and still in this device's local cache, so rollback costs no pull.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|:---|:---|:---|
| `COPY failed: file not found` | `models/` and `data/cv_demo_clips/` are gitignored, so the clone lacks them | extract them from the parent image (§3) |
| `docker build` fails on `pip` with a network error | no egress to PyPI at that moment | retry, or build a COPY-only layer instead (§2) |
| `no space left on device` mid-build | root fs at ~88 % | `docker image prune`, then rebuild |
| Demo runs on CPU after a pip layer | PyPI torch replaced JetPack's | use the constraints file (§4) |
| `The function is not implemented. Rebuild the library with GTK+ 2.x support` | `opencv-python-headless` in `/usr/local` shadows JetPack's OpenCV | keep `PYTHONPATH=/usr/lib/python3.10/dist-packages` **and** `YOLO_AUTOINSTALL=false` — ultralytics reinstalls `opencv-python` at import without the second |
| ~22 confident boxes/frame, all wrong | YOLO26 weights under ultralytics 8.3.x — coerced to the legacy head, **no error raised** | build from `1.2.0-ul84` or later; assert `end2end` (§4) |
| `denied: requested access to the resource is denied` | not logged in, or the account lacks push rights on the project | `docker login "$REGISTRY"` (§7.1) |
| `exec format error` after deploy | an amd64 image reached the registry | rebuild here, re-verify §6 and §7.2 |

---

Advantech Corporation — Center of Excellence
