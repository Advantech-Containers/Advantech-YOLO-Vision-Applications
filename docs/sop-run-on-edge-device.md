# SOP — Run the YOLO demo on an edge device

**Applies to:** `adlk.edgedevice.2` (Jetson Orin, JetPack 6.2 / R36.4.4,
deviceId `74fe488d5d54`) and any equivalently-provisioned WEDA device.
**Audience:** whoever brings the showroom/trade-show display up and keeps it up.
**Companion:** [sop-build-push-image.md](sop-build-push-image.md) (getting an
image into the registry), [cv-container-reference.md](cv-container-reference.md)
§8 (X11), §13 (WEDA), §14 (runbook).

Two ways to run it. **Path A (WEDA)** is how the demo is actually deployed and
survives reboots. **Path B (compose on the device)** is for a local test when
WEDA is unavailable or you are bisecting a problem.

---

## 1. Pre-flight — run these before touching a deployment

```bash
export DEVICE=<your_device_IP>
export DEVICE_ID=<your_deive_mac_str> #74fe488d5d54
```

| # | Check | Command | Expect |
|:--|:------|:--------|:-------|
| 1 | Device reachable | `ssh "$DEVICE" true` | exit 0 |
| 2 | X is up | `ssh "$DEVICE" 'DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority xset q \| head -3'` | no error |
| 3 | X cookie present | `ssh "$DEVICE" 'ls -l /run/user/1000/gdm/Xauthority'` | file exists |
| 4 | Screen blanking off | `ssh "$DEVICE" 'DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority xset q \| grep -A2 "Screen Saver"'` | `timeout: 0`, DPMS all `0` |
| 5 | Disk headroom | `ssh "$DEVICE" 'df -h /'` | root fs sits at ~88%; below ~5% free, stop and clean up |
| 6 | Agent alive | `ssh "$DEVICE" 'docker ps --filter name=dmagent --format "{{.Status}}"'` | `Up …` (Path A only) |

**The X cookie is gdm's, at `/run/user/1000/gdm/Xauthority` — not
`~/.Xauthority`.** The home-directory file exists but is stale and fails
authentication. This is the single most common cause of "container running,
screen blank".

If check 4 fails on a freshly imaged device, run in the desktop session:

```bash
ssh "$DEVICE" 'DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority xset s off -dpms'
```

Otherwise the panel blanks while the container stays perfectly healthy.

---

## 2. Path A — deploy through WEDA (the normal way)

### 2.1 Get a token

Base URL `https://weda-sit-k3s.weda.dev/central/weda`. The SIT domain presents a
**self-signed certificate** — pin it, do not disable verification:

```bash
openssl s_client -connect weda-sit-k3s.weda.dev:443 \
  -servername weda-sit-k3s.weda.dev -showcerts </dev/null 2>/dev/null \
  | awk '/BEGIN CERT/,/END CERT/' > weda-sit-k3s.pem
```

Mint the JWT with the `weda-login` skill. **Tokens last 2 hours**; expiry
surfaces only as `401 token is expired`.

### 2.2 Deploy

Use `docker/weda-stack-od-demo.yml` as the compose body (base64 in
`composeFileContent`). Call sequence:

| step | call |
|:-----|:-----|
| reuse the registry credential | `GET /api/v1/orgs/{orgId}/containers/registries` — reuse `harbor-arfa-edge-coa`, **do not create a duplicate** |
| create or update the stack | `POST /api/v1/orgs/{orgId}/stack-configs`, or `PATCH …/stack-configs/{stackConfigId}` → new immutable revision |
| **clear existing deployments** | `DELETE /api/v1/devices/{DEVICE_ID}/stacks/deployments` |
| deploy | `POST …/revisions/{stackRevisionId}:deploy` `{"deviceIds":["74fe488d5d54"]}` |

**Always clear-all before deploying, including on a version bump.** Plain
delete-then-deploy has wedged; clear-all-then-deploy has landed in ~30 s every
time.

Expect **tens of seconds of dark screen** on every bump — the old container is
removed before the new one is scheduled.

### 2.3 Known revisions (rollback targets)

| ver | revisionId | image | model | clip |
|:----|:-----------|:------|:------|:-----|
| v1 | `c0139896-8dd1-4e62-bbf3-f4cee5410eaa` | 1.0.0 | yolo11n | OD_Jar |
| v2 | `7dc22a75-a938-4f33-9286-77915fb06229` | 1.3.0 | jar26n | OD_Jar |
| v3 | `45685531-acef-4405-a74f-d928997c99e7` | 1.4.0 | yolo11n | OD_bottle_2 |
| v4 | `2201a204-630a-4f45-a893-70d0c9540704` | 1.5.0 | yolo11n | + MQTT, fullscreen |
| v5 | `de070d1b-4e9c-4f8a-9ca4-78fccbfce2b6` | 1.6.0 | yolo11n | + MQTT, windowed |
| v8 | `5842934c-a7c6-41bb-88a7-caa2cff48411` | 1.6.0 | **yolo26n** | + MQTT, windowed — **current** |

Rollback = clear-all, then deploy the older `revisionId`. No rebuild, no pull.

---

## 3. Path B — run it directly on the device with compose

For a local test. Requires the image already present or pullable on the device.

```bash
rsync -a docker/weda-stack-od-demo.yml "$DEVICE:/home/ubuntu/yolo-od-demo/"
ssh "$DEVICE" 'cd /home/ubuntu/yolo-od-demo && docker compose -f weda-stack-od-demo.yml up -d'
ssh "$DEVICE" 'cd /home/ubuntu/yolo-od-demo && docker compose -f weda-stack-od-demo.yml logs -f'
```

Tear down with `docker compose -f weda-stack-od-demo.yml down`.

Two things to know before you do this:

* The compose file is written for WEDA and **hardcodes `DISPLAY=:0` and the gdm
  cookie path** rather than inheriting `${DISPLAY}` from your shell — which is
  correct here, because an SSH session has no X session to inherit from either.
* Containers started this way are **not** `edge_`-prefixed. That prefix is how
  you tell a WEDA-managed container from a hand-started one — useful when both
  exist and the screen shows the wrong thing.

---

## 4. Verify — all four, never just the first

`status: running` is **not** proof. A container can be up, healthy, and showing
a blank screen.

```bash
# 1. WEDA agrees (Path A only)
curl --cacert weda-sit-k3s.pem -H "Authorization: Bearer $TOK" \
  "$BASE/api/v1/devices/$DEVICE_ID/docker/stacks"
#    expect status=running, operationStatus=deployed, containers[].state=running

# 2. the demo is actually inferring
ssh "$DEVICE" 'docker logs --tail 20 edge_yolo-od-demo-yolo-od-demo-1'
```

A healthy log opens with these and then repeats a stats line every 60 s:

```
[demo] OpenCV 4.8.0 (GUI enabled)
[demo] CUDA available: Orin
[demo] window: 1280x720 at +40+40
[demo] 12.0 FPS | 6.1 objects/frame | laps=2 | mqtt published=… dropped=0
```

`[demo] PRECONDITION CHECK FAILED:` means the runner refused to start and named
the reason — missing clip, missing model, headless OpenCV, or unset `DISPLAY`.
Read the bullets under it; each states its own remedy.

```bash
# 3. it is on the screen
ssh "$DEVICE" 'DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority scrot /tmp/s.png' \
  && scp "$DEVICE:/tmp/s.png" .

# 4. telemetry is flowing
ssh "$DEVICE" 'timeout 8 docker run --rm --network host \
  eclipse-mosquitto:2 mosquitto_sub -h 127.0.0.1 -t "advantech/#" -v -W 6'
```

Topics under `advantech/74fe488d5d54/vision`: retained `/status`
(`online`/`offline`, the LWT), retained `/meta`, and `/detections` every 1 s.
Telemetry is **best-effort by design** — if the broker dies the demo keeps
rendering, which is the correct trade for a display.

---

## 5. Day-to-day operations

### 5.1 Change model, clip, window size or thresholds — no rebuild

Every knob is an environment variable read at startup. `yolo11n` and `yolo26n`
are both baked into 1.4.0+, as is `OD_bottle_2.mp4`.

| variable | current | alternatives |
|:---|:---|:---|
| `MODEL_PATH` | `/advantech/models/yolo26n.pt` | `…/yolo11n.pt` (6.06 boxes/frame, 23.8 FPS vs 4.97 @ 22.4) |
| `VIDEO_PATH` | `/advantech/data/OD_bottle_2.mp4` | `…/OD_Jar.mp4` (1.3.0 images only) |
| `CONF_THRESHOLD` | `0.40` | flat quality 0.25–0.50 |
| `FULLSCREEN` | `false` | `true` |
| `WINDOW_WIDTH/HEIGHT` | `1280x720` | must be set explicitly when not fullscreen — the clip is 2560x1440 and would overflow the 1920x1080 panel |
| `MQTT_INTERVAL_SEC` | `1.0` | per-frame publishing is 12 msg/s of near-identical payloads for no gain |

Edit the stack → `PATCH` for a new revision → clear-all → deploy.

**A fine-tuned model is clip-specific.** `jar26n` knows exactly one class,
`jar`, trained on 232 frames of one clip. Pointing it at another clip detects
nothing — that is the model working as designed, not a fault.

### 5.2 Start / stop / restart without redeploying

```
POST /api/v1/devices/{DEVICE_ID}/docker/stacks/commands:{start|stop|restart|pause|resume}
     {"stackConfigId": "46a6ca7b-5e46-4c31-8f60-639417d72ad1"}
GET  /api/v1/devices/{DEVICE_ID}/docker/stacks/commands/results?stackConfigId=…
```

### 5.3 After a device reboot

Nothing to do. `restart: unless-stopped` brings both containers back, and
`entrypoint-demo.sh` waits up to `X_WAIT_TIMEOUT_SEC` (120 s) for gdm to bring X
up before starting — without that wait the container would crash-loop on a
missing display. If the device is unusually slow to reach the desktop, raise
that variable rather than adding a sleep.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|:---|:---|:---|
| Deployment stuck at `deploying`, `containers: []`, `error: {}`, forever | dmagent missed the delta; API returns 200 and shows nothing wrong | **both, in order:** `docker restart dmagent`, then delete the stack and **deploy again**. A delta published while the agent was disconnected is never replayed, so restart alone leaves it stuck |
| Deployment stuck at `awaitingCleanup` | apply timed out; does not self-resolve | clear-all, then fresh deploy |
| Container `running`, screen blank | X auth or `DISPLAY` | check the log for `PRECONDITION CHECK FAILED`; the cookie is gdm's, not `~/.Xauthority` |
| Screen went black after hours of running | screen blanking re-enabled | `xset s off -dpms` (§1) |
| `cv2.error: The function is not implemented … GTK+ 2.x` | `opencv-python-headless` in `/usr/local` shadows JetPack's OpenCV | `PYTHONPATH=/usr/lib/python3.10/dist-packages` **and** `YOLO_AUTOINSTALL=false` — ultralytics reinstalls `opencv-python` at import without the second |
| ~22 confident boxes/frame, all wrong, no error | YOLO26 weights under ultralytics 8.3.x — coerced to the legacy head | run an image built from `1.2.0-ul84` or later |
| Window runs off the edge of the panel | `FULLSCREEN=false` without an explicit size | set `WINDOW_WIDTH/HEIGHT` |
| Crash loop right after reboot | started before X was up | expected; entrypoint waits 120 s. Raise `X_WAIT_TIMEOUT_SEC` |
| `mqtt connected=False` in the stats line | broker down or wrong host | check the `mqtt-broker` container; the client auto-retries and the display is unaffected |
| `401 token is expired` | JWT older than 2 h | re-mint via `weda-login` |
| `403 OrganizationNotFoundOrNoPermission` | token lacks rights to that org/device | check `resource_access` in the JWT |
| Device pings but SSH hangs | memory exhaustion | never train on the device — 7.6 GB shared |

**Is it your compose, or the agent?** Deploy a trivial busybox stack to the same
device. If that stalls too, the fault is dmagent, not your stack. Outbound
telemetry (`cfgstate_handler … published config shadow state`) keeps working
throughout a stall, which is exactly why the device still looks healthy in WEDA.

```bash
ssh "$DEVICE" 'docker logs dmagent | grep -E "stack_async|composeplugin|Reported stacks"'
# if the last entry predates your deploy, the agent never received it
```

---

## 7. Running the interactive apps on the device instead

The demo runner is unattended by design. To drive detection/segmentation/
classification by hand on the same device, use the toolkit container — it is a
different image and a different workflow (`build.sh`, then
`python3 src/advantech-yolo.py`). Two things it needs that the demo stack
handles for you:

```bash
xhost +local:docker                                    # on the device's desktop
export PYTHONPATH=/usr/lib/python3.10/dist-packages    # inside the container
export YOLO_AUTOINSTALL=false                          # or ultralytics undoes it
```

Without both variables, `--show` fails with the GTK error above. See the
[README](../README.md) for the full interactive workflow.

---

Advantech Corporation — Center of Excellence
