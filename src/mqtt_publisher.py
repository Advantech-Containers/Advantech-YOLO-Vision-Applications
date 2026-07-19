#!/usr/bin/env python3
"""
MQTT publisher for YOLO detection results
=========================================
Version:      1.0.0
Created:      July 19, 2026
Description:  Publishes detection summaries from the looping vision demo to an
              MQTT broker.

Design notes:
  * The broker is a convenience, not a dependency.  If it is unreachable the
    demo keeps rendering to the screen -- losing telemetry must never take the
    display down.  Failures are logged with context, never swallowed silently.
  * A Last Will and Testament marks the device offline if the container dies,
    so a subscriber can distinguish "no detections" from "publisher gone".
  * Detections are published on an interval, not per frame.  At ~12 FPS a
    per-frame topic would emit 12 msg/s of near-identical payloads for no
    analytical gain.

Copyright (c) 2026 Advantech Corporation. All rights reserved.
"""

import json
import os
import socket
import time
from datetime import datetime, timezone

__version__ = "1.0.0"

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:                                   # pragma: no cover
    MQTT_AVAILABLE = False


def _log(message):
    print(f"[mqtt] {message}", flush=True)


class DetectionPublisher:
    """Publishes detection summaries to MQTT.

    All failures are non-fatal: every public method degrades to a no-op if the
    broker is unavailable, so the caller never needs to guard its calls.

    Topics (base defaults to ``advantech/<device>/vision``):
        <base>/status       retained, "online" / "offline" (LWT)
        <base>/detections   detection summary, published every interval
        <base>/meta         retained, model/source/config, published once
    """

    def __init__(self):
        self.enabled = os.getenv("MQTT_ENABLED", "true").lower() == "true"
        self.host = os.getenv("MQTT_HOST", "127.0.0.1")
        self.port = int(os.getenv("MQTT_PORT", "1883"))
        self.qos = int(os.getenv("MQTT_QOS", "0"))
        self.interval = float(os.getenv("MQTT_INTERVAL_SEC", "1.0"))
        self.include_boxes = os.getenv("MQTT_INCLUDE_BOXES", "true").lower() == "true"
        self.device_id = os.getenv("DEVICE_ID", socket.gethostname())

        base = os.getenv("MQTT_TOPIC_BASE", f"advantech/{self.device_id}/vision")
        self.topic_status = f"{base}/status"
        self.topic_detections = f"{base}/detections"
        self.topic_meta = f"{base}/meta"

        self._client = None
        self._connected = False
        self._last_publish = 0.0
        self._published = 0
        self._dropped = 0

    # -- lifecycle ---------------------------------------------------------

    def connect(self, meta=None):
        """Connect and start the network loop. Never raises."""
        if not self.enabled:
            _log("disabled via MQTT_ENABLED=false")
            return False
        if not MQTT_AVAILABLE:
            _log("WARNING: paho-mqtt not installed -- telemetry disabled, "
                 "demo continues")
            return False

        try:
            self._client = mqtt.Client(
                client_id=f"yolo-demo-{self.device_id}-{int(time.time())}",
                clean_session=True,
            )
            # LWT: if this container dies, subscribers see 'offline' rather
            # than a stale 'online' that never changes.
            self._client.will_set(self.topic_status, "offline",
                                  qos=1, retain=True)
            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._client.reconnect_delay_set(min_delay=1, max_delay=30)
            self._client.connect_async(self.host, self.port, keepalive=30)
            self._client.loop_start()
            _log(f"connecting to {self.host}:{self.port} "
                 f"(qos={self.qos}, interval={self.interval}s)")
            _log(f"topics: {self.topic_detections} | {self.topic_status}")

            if meta:
                self._meta = meta
            return True
        except Exception as exc:
            _log(f"WARNING: connect failed ({type(exc).__name__}: {exc}) -- "
                 "telemetry disabled, demo continues")
            self._client = None
            return False

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            _log(f"connected to {self.host}:{self.port}")
            try:
                client.publish(self.topic_status, "online", qos=1, retain=True)
                meta = getattr(self, "_meta", None)
                if meta:
                    client.publish(self.topic_meta, json.dumps(meta),
                                   qos=1, retain=True)
            except Exception as exc:
                _log(f"WARNING: status publish failed: {exc}")
        else:
            self._connected = False
            _log(f"WARNING: connect refused (rc={rc}) -- will retry")

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        if rc != 0:
            _log(f"WARNING: unexpected disconnect (rc={rc}) -- auto-reconnecting")

    # -- publishing --------------------------------------------------------

    def publish(self, result, frame_index, lap, fps):
        """Publish a detection summary if the interval has elapsed.

        Rate limiting lives here rather than in the caller so the render loop
        stays free of telemetry concerns.
        """
        if not self._client:
            return
        now = time.monotonic()
        if now - self._last_publish < self.interval:
            return
        self._last_publish = now

        try:
            payload = self._build_payload(result, frame_index, lap, fps)
            info = self._client.publish(self.topic_detections,
                                        json.dumps(payload), qos=self.qos)
            if info.rc == mqtt.MQTT_ERR_SUCCESS:
                self._published += 1
            else:
                self._dropped += 1
        except Exception as exc:
            self._dropped += 1
            _log(f"WARNING: publish failed ({type(exc).__name__}: {exc})")

    def _build_payload(self, result, frame_index, lap, fps):
        detections = []
        classes = {}
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            names = result.names
            for box in boxes:
                label = names[int(box.cls)]
                conf = round(float(box.conf), 4)
                classes[label] = classes.get(label, 0) + 1
                if self.include_boxes:
                    x1, y1, x2, y2 = (round(float(v), 1)
                                      for v in box.xyxy[0].tolist())
                    detections.append({"class": label, "confidence": conf,
                                       "bbox": [x1, y1, x2, y2]})

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(
                timespec="milliseconds").replace("+00:00", "Z"),
            "deviceId": self.device_id,
            "frame": frame_index,
            "lap": lap,
            "fps": round(fps, 2),
            "objectCount": sum(classes.values()),
            "classCounts": classes,
        }
        if self.include_boxes:
            payload["detections"] = detections
        return payload

    def stats(self):
        return f"published={self._published} dropped={self._dropped} " \
               f"connected={self._connected}"

    def close(self):
        """Mark offline and shut the client down cleanly. Never raises."""
        if not self._client:
            return
        try:
            self._client.publish(self.topic_status, "offline",
                                 qos=1, retain=True)
            time.sleep(0.2)          # let the retained status leave the buffer
            self._client.loop_stop()
            self._client.disconnect()
            _log(f"closed ({self.stats()})")
        except Exception as exc:
            _log(f"WARNING: shutdown error ({type(exc).__name__}: {exc})")
