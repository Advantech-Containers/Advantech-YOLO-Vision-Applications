# Training Guide — Fine-Tuning a Detector for a Demo Clip

How to turn a clip that stock COCO weights cannot detect into one with a
working detector, without labelling anything by hand.

Scripts live in [`training/`](../training/). Full context for the container
they feed is in [cv-container-reference.md](cv-container-reference.md).

---

## When you need this

Run [`survey_clips.py`](../training/survey_clips.py) first. It tells you which
of three situations you are in:

| survey result | meaning | action |
|:--------------|:--------|:-------|
| high coverage, **correct** labels | COCO already knows the subject | use stock weights — **no training needed** |
| high coverage, **wrong** labels | COCO substitutes its nearest class | fine-tune (this guide) |
| low coverage | subject is invisible to the teacher too | pick a different clip |

```bash
python3 training/survey_clips.py --dir data/cv_demo_clips --conf 0.40
```

Real output from this repository's clips:

```
clip                    resolution   secs  coverage  boxes/frm  top classes
OD_bottle_2              2560x1440    8.0      100%       7.33  bottle:396
OD_Jar                   2560x1440    9.3       53%       0.84  cup, person, vase
OD_Seg_Sushi              1280x720   16.5       73%       1.22  person, pizza, cake
332263_medium             1280x720    5.3        0%       0.00  -- nothing --
```

`OD_bottle_2` needs nothing — COCO knows `bottle`. `OD_Jar` and `OD_Seg_Sushi`
are the fine-tuning cases: the model detects *something* but calls jars
`surfboard` and sushi `pizza`. `332263_medium` is unusable.

**Do not try to fix wrong labels with a confidence threshold.** On `OD_Jar` the
correct class (`bottle`) never exceeded 0.30 confidence while the wrong one
(`surfboard`) exceeded 0.50 fifty-one times. Raising the threshold removes
every jar and keeps the surfboards.

---

## The pipeline

```
survey_clips.py      which clip, and does it need training at all
      │
make_dataset.py      teacher + tracking + interpolation -> labelled dataset
      │
preview_labels.py    LOOK AT THE LABELS  <-- do not skip
      │
train.py             on a GPU workstation, NOT the edge device
      │
evaluate.py          verify confidence distribution, not just mAP
```

---

## Step 1 — Build the dataset

```bash
python3 training/make_dataset.py \
    --video data/cv_demo_clips/OD_Jar.mp4 \
    --out   /tmp/jar-dataset \
    --class-name jar \
    --teacher yolo11l.pt \
    --teacher-imgsz 1536
```

### How it works

A COCO teacher has poor per-frame recall on objects it was never trained on.
Associating its detections into **tracks** and **interpolating the gaps**
recovers the frames it misses — an object on a conveyor moves predictably, so
a box interpolated between two confident detections is usually closer to truth
than the teacher's own miss. That temporal completion is what lets the student
exceed the teacher.

1. **Teacher** at high resolution, low confidence, restricted to plausible COCO
   classes.
2. **Associate** across frames by greedy IoU. (Ultralytics' BoT-SORT needs the
   `lap` extension, absent on aarch64 and requiring a source build; a small
   matcher avoids the dependency.)
3. **Interpolate** linearly across gaps up to `--max-gap`.
4. **Filter** tracks shorter than `--min-track-len` — these are flicker.
5. **NMS** per frame — the teacher labels one object under two classes
   (`wine glass` *and* `cup`), producing stacked duplicates. On `OD_Jar` this
   merged **330** boxes.

### Teacher resolution is the biggest lever

More important than teacher model size:

| `--teacher-imgsz` | boxes/frame on OD_Jar |
|------------------:|----------------------:|
| 640 (default) | 0.43 |
| **1536** | **2.56** |

The source is 2560x1440; 640 shrinks the jars past detectability. If your clip
is high-resolution, raise this first.

**Result on `OD_Jar`:** 232 frames, 509 boxes, **68% coverage** — from a
teacher that never managed 40% per frame.

---

## Step 2 — Look at the labels

```bash
python3 training/preview_labels.py --dataset /tmp/jar-dataset --out preview.jpg
```

**Do not skip this.** Summary statistics cannot tell you whether boxes are on
the right objects. This step immediately exposed the duplicate-box defect that
the counts reported as a healthy 68% coverage.

Check three things:

- boxes are on the **intended object**, not the background or machinery
- no **stacked duplicates** on a single object
- boxes are not **systematically offset** or oversized

Training on bad labels is worse than not training.

---

## Step 3 — Train, on the right machine

```bash
scp -r /tmp/jar-dataset gpu-host:~/train/
ssh gpu-host 'docker run --rm --runtime nvidia --gpus all --shm-size 8g \
  -v ~/train:/data --entrypoint python3 ultralytics/ultralytics:latest \
  /data/train.py --data /data/jar-dataset/data.yaml --model yolo26n.pt --epochs 120'
```

### Not on the edge device

| host | hardware | 120 epochs @ 960px |
|:-----|:---------|:-------------------|
| `acn.air-520` | 2x RTX 6000 Ada, 251 GB RAM | **15.6 minutes** |
| Jetson Orin | 7.6 GB shared with the running demo | ~15 min **per epoch** |

Training on the Orin drove load average to **74** and left `sshd` unable to
fork — the device was unreachable for minutes. It recovered and the demo
survived, but do not do this.

### Share the GPU politely

Both `acn.air-520` cards are typically ~93% occupied by an unrelated **vLLM
server** (~2.9 GB free each). A nano model at `--batch 4` fits in 1.6 GB and
coexists. `train.py` prints free GPU memory and warns below 2 GB. **Do not
evict other people's services.**

### YOLO26 needs ultralytics ≥ 8.4

Under 8.3.x a YOLO26 checkpoint is silently coerced into the legacy head and
trains incorrectly, with no error. `train.py` refuses to start in that
combination.

---

## Step 4 — Verify, and do not trust mAP alone

```bash
python3 training/evaluate.py \
    --video data/cv_demo_clips/OD_Jar.mp4 \
    --models runs/finetune/weights/best.pt \
    --conf 0.25 0.40 0.50 --classes-report
```

### The trap this catches

| epochs | val mAP50 | max confidence at inference | usable? |
|-------:|----------:|----------------------------:|:--------|
| 2 | 0.47 | **0.026** | **no** — zero detections at any sane threshold |
| 120 | 0.972 | 0.95 | yes |

**mAP is rank-based.** Box placement scores well long before the classification
head is calibrated, so a badly undertrained model reports a respectable mAP
while detecting nothing in practice. If `evaluate.py` shows 0% coverage but
your training log looked fine, re-run with `--conf 0.001` — if boxes appear
there in a narrow band around 0.02, the model simply needs more epochs.

### Measure the whole clip

`--max-frames 0` (the default) reads every frame. Measuring only an early
window overstates coverage badly — the first 150 of the sushi clip's 495 frames
reported 100% where the true figure was **73%**.

### What good looks like

```
model                  imgsz  conf  coverage  boxes/frm  min  max     FPS
jar26n.pt                640  0.25       55%       2.09    0    4    22.0
jar26n.pt                640  0.40       53%       1.94    0    4    23.3
jar26n.pt                640  0.50       53%       1.86    0    4    21.9
```

Quality **flat across thresholds** is the signature of a calibrated model —
detections are genuinely confident, not scraping past a low bar. A model whose
coverage collapses as the threshold rises is undertrained.

Final `jar26n`: **mAP50 0.972, mAP50-95 0.899, P 0.934, R 0.918** — and on the
clip, 6x the detections of stock `yolo26n`, labelled `jar` instead of
`surfboard`.

---

## Step 5 — Ship it

Bake the weights into an image and deploy. See
[cv-container-reference.md §12–13](cv-container-reference.md#12-building-and-publishing-images).

```dockerfile
FROM harbor.arfa.wise-paas.com/edge-coa/yolo-od-demo:1.2.0-ul84
COPY models/jar26n.pt /advantech/models/jar26n.pt
ENV MODEL_PATH=/advantech/models/jar26n.pt CONF_THRESHOLD=0.40
```

A YOLO26-derived model **must** use the `1.2.0-ul84` base or later.

---

## Scope and honesty

A model trained this way is **specialised to the clip it was trained on**.
`jar26n` was built from 232 frames of one 9-second video and will not
generalise to another line, jar, camera angle, or lighting. For a looping demo
that is the correct trade. For production inspection it is not: that needs real
labelled data across varied conditions, and the pseudo-labels here were
visually spot-checked rather than verified frame by frame.

Say which one you are building before anyone shows it to a customer.
