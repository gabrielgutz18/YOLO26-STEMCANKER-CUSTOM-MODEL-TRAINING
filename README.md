# Dragon-fruit stem canker detection: YOLO26-small

Training, evaluation and inference code for detecting **stem canker** lesions on dragon-fruit
stems with **Ultralytics YOLO26-small**. The main goal is **higher confidence scores**,
measured honestly.

```
configs/
  dataset.yaml            original Roboflow split (YAOWI.yolov12/)
  dataset_clean.yaml      leak-free re-split (generated)
  dataset_v2.yaml         leak-free split + corrected labels (generated)
  label_fixes_v2.csv      612 reviewed lesions the original labels missed
  train.yaml              confidence-tuned hyper-parameters, from COCO weights (round 1)
  train_v2.yaml           fine-tune of the round-1 best.pt on dataset_v2   <- current config
  train_baseline.yaml     plain Ultralytics defaults, for an A/B comparison
scripts/
  check_dataset.py        label / leakage audit
  make_clean_split.py     group near-duplicate photos, re-split without leakage
  apply_label_fixes.py    clean split + label_fixes_v2.csv -> stemcanker_v2
  train.py                train (+ automatic test-set confidence report)
  evaluate.py             mAP + confidence report, model comparison, error export
  predict.py              inference on images / folders / video
runs/                     all outputs (train/<name>/weights/best.pt, eval_test/, predict/)
  label_review_v2/        the crop sheets the label fixes were decided on
weights/                  pretrained yolo26s.pt (auto-downloaded)

%USERPROFILE%\datasets\avx-dragonfruit\stemcanker_clean
                          training copy in the clean split, on the internal SSD
%USERPROFILE%\datasets\avx-dragonfruit\stemcanker_v2
                          same split with corrected labels (images hard-linked, no extra space)
```

## Round 2: corrected labels (current)

Round 1 (`yolo26s_stemcanker-2`, `train.yaml`) reached test mAP50 0.303. The diagnosis was the
labels, not the settings: the model scored AP50 0.81 on the photo source with consistent labels
and 0.25–0.30 on the rest, and 37% of its confident detections had no label at all.

What was changed (`scripts/apply_label_fixes.py`):

* Every detection of the round-1 model with conf ≥ 0.3 that touched no label (867) was reviewed
  on crop sheets (`runs/label_review_v2/`). **612 real lesions were added** (train 443, val 105,
  test 64). Rejected: areoles, insects, debris, glare, padding edges, boxes spanning two lesions,
  and faint white spots that couldn't be identified with confidence.
* 242 polygon labels were converted to boxes (no more "Box and segment counts" warning), and 1
  zero-size box and 1 duplicate box were removed.
* The split is unchanged, so old and new models are compared on the same test photos.

Round-1 model on the **corrected** test labels (`runs/train/yolo26s_stemcanker-2/eval_test_v2labels`):
mAP50 **0.376**, mAP50-95 **0.198**, best F1 0.460. This is the number round 2 has to beat.
It's higher than the 0.303 against the old labels because many of the round-1 model's "false
alarms" were real lesions. The added test labels came from that model's own proposals, so this
comparison slightly favours it.

```powershell
cd "D:\0. GXJ PROJECTS\0. AVX-DRAGONFRUIT"
& "$env:USERPROFILE\.venvs\avx-dragonfruit\Scripts\Activate.ps1"
python scripts/train.py --config configs/train_v2.yaml
```

This fine-tunes from `runs/train/yolo26s_stemcanker-2/weights/best.pt` (AdamW, lr 0.001, 100
epochs, patience 30): about 1 h. The result goes to `runs/train/yolo26s_stemcanker_v2`. Compare:

```powershell
python scripts/evaluate.py runs/train/yolo26s_stemcanker-2/weights/best.pt runs/train/yolo26s_stemcanker_v2/weights/best.pt --data configs/dataset_v2.yaml
```

Still not fixed: lesions the round-1 model never detected, e.g. the large grey necrotic cankers
in photo `169`. Repeating the review with the v2 model will surface more of them.

## Run the round-1 training (PowerShell, from this folder)

The environment and the clean split are already set up on this PC, so only this is needed:

```powershell
cd "D:\0. GXJ PROJECTS\0. AVX-DRAGONFRUIT"
& "$env:USERPROFILE\.venvs\avx-dragonfruit\Scripts\Activate.ps1"
python scripts/train.py
```

You'll see one progress bar per epoch (`box_loss`, `cls_loss`, GPU memory, speed) followed by a
validation line (`P  R  mAP50  mAP50-95`). Expect about 4 it/s and **2–2.5 h for 200 epochs**.
Training stops earlier if validation doesn't improve for 50 epochs. At the end, the
test-set confidence report prints automatically. `Ctrl+C` stops safely; continue with
`python scripts/train.py --resume runs/train/yolo26s_stemcanker/weights/last.pt`.
Curves are saved live to `runs/train/yolo26s_stemcanker/results.png`.

## All steps (fresh machine / after a new Roboflow export)

```powershell
# 1. environment (once). Creates the venv on the internal SSD; see "Performance" below
powershell -ExecutionPolicy Bypass -File setup_env.ps1
& "$env:USERPROFILE\.venvs\avx-dragonfruit\Scripts\Activate.ps1"

# 2. audit + leak-free split (once, and again after every Roboflow re-export)
python scripts/check_dataset.py --imgsz 800
python scripts/make_clean_split.py

# 3. train
python scripts/train.py
#    any Ultralytics argument can be overridden:
python scripts/train.py imgsz=960 batch=8 name=yolo26s_960
#    A/B baseline with plain Ultralytics defaults:
python scripts/train.py --config configs/train_baseline.yaml

# 4. evaluate / compare
python scripts/evaluate.py runs/train/yolo26s_stemcanker/weights/best.pt --export-errors
python scripts/evaluate.py runs/train/yolo26s_baseline/weights/best.pt runs/train/yolo26s_stemcanker/weights/best.pt

# 5. predict (uses the best-F1 threshold from step 4 automatically)
python scripts/predict.py --source path\to\new\photos
```

## How YOLO26 produces its confidence score, and what this project changes

YOLO26 trains its score with a task-aligned target (`ultralytics/utils/tal.py`). For every
labelled lesion, the one-to-one (NMS-free) head gets **one** positive prediction whose target
score is the **IoU between the predicted box and your label box**. A lesion that the model boxes
at IoU 0.6 is trained toward a score of about 0.6. So:

> **Confidence ≈ how tightly the model can reproduce your boxes.** Tighter, more consistent
> labels and better box regression are what raise it.

| Setting (`configs/train.yaml`) | Default | Here | Why it raises confidence |
|---|---|---|---|
| `data` | original split | leak-free split | scores measure real photos, not memorised copies; `best.pt` is chosen on unseen data |
| `imgsz` | 640 | **800** | 22% of lesions are < 16 px at 640; small boxes get low IoU, which means low score |
| `epochs` / `patience` | 100 / 100 | **200 / 50** | small dataset; YOLO26 shifts loss weight to the inference (one-to-one) head over the run |
| `cls` | 0.5 | **1.0** | more gradient to the score branch so it actually reaches its IoU target |
| `close_mosaic` | 10 | **20** | longer final phase on real, un-mosaicked images |
| `cos_lr` | off | **on** | smoother convergence at the end |
| `flipud` | 0.0 | **0.5** | stems have no canonical "up"; flips keep boxes exact |
| `degrees`, `shear`, `perspective` | 0 | **kept 0** | rotation turns tight boxes into loose ones, which lowers the IoU target |
| `mixup`, `cutmix` | 0 | **kept 0** | blending images blends labels and teaches the model to be less sure |

YOLO26 has no NMS and **does not support test-time augmentation** (Ultralytics silently falls
back to single-scale), so there is no `--tta` option.

## Reading the confidence report

`evaluate.py` writes `runs/train/<name>/eval_test/confidence_report.{json,png}`. It prints:

| Metric | Meaning | Goal |
|---|---|---|
| **lesion confidence** | mean over *all* labelled lesions of the score they were detected with (0 if missed). Recall and confidence in one number | ↑ |
| mean / median conf, correct dets | score of correct detections (IoU ≥ 0.5) | ↑ |
| recall @ conf ≥ 0.5 / 0.7 | share of lesions found *confidently* | ↑ |
| precision @ conf ≥ 0.5 / 0.7 | share of confident detections that are real | ↑ |
| false alarms/img @ 0.5 | confident mistakes | ↓ |
| conf threshold at best F1 | the `conf` to deploy with; `predict.py --conf auto` uses it | |

A real improvement moves the correct-detection numbers up **without** pushing false alarms up.
Any model can be made to print bigger numbers; that isn't the goal.

## Findings about this dataset (from `check_dataset.py`)

* **Near-duplicate leakage.** The export contains the same photos several times (resized,
  re-encoded, flipped, pre-augmented), split at random. 210 photo groups spanned more than one
  split, and about 40% of the original test images had a near-copy in train.
  `make_clean_split.py` keeps every copy of a photo inside one split (627 groups, same
  1488/213/170 sizes, verified 0 near-duplicate pairs across splits). The list is in
  `split_manifest.csv` inside the dataset copy.
  **Don't re-run it while training** (it rewrites the folder, and the script refuses unless
  `--force`). After a re-split, start a fresh run rather than `--resume`, because the
  val/test sets have changed.
* **140 label files use 4-point polygons** (242 lines) instead of `x y w h`. All of them are
  axis-aligned rectangles, i.e. ordinary boxes in another format, and they line up with the
  lesions once EXIF rotation is applied. Ultralytics converts them correctly, so there's
  nothing to fix.
* **59 photos carry an EXIF rotation tag.** Labels refer to the *rotated* image, which is how
  Ultralytics loads them. Some photos exist both EXIF-rotated and physically rotated under
  different names (`Stem_Canker (12)` vs `Stem_Canker-12-`); the split script hashes after
  applying EXIF so those are grouped too.
* **About 230 pre-augmented copies (~12%).** 177 are named as augmentations (`aug3_…` prefix 95,
  `__flipv` 31, `__fliph` 26, `__trans` 11, `__noise` 10, `__rot` 10). About 60 more are
  rotated/sheared copies on black or grey padding (detected by border; a few are false
  positives, e.g. a photo with plain sky at the top). Ultralytics augments during training
  anyway, so these mostly add duplicates.
* **Labelling style differs between sources.** Some photos box the large necrotic cankers,
  others box only small spots and leave the large lesion unlabelled. This caps confidence;
  see "Raising confidence further".
* 1 zero-size / out-of-range box (`402__fliph_jpg…`), 1 duplicated box (`Stem_Canker_269_jpg…`),
  1 background image. Ultralytics drops the invalid box by itself.
* Mixed image sizes: 512×512 and 640×360 (about 70%) plus 3000–4000 px phone photos.

## Raising confidence further (in order of impact)

1. **Fix the labels.** `evaluate.py … --export-errors` saves the images the model disagrees with
   most to `eval_test/errors/`. **Red** boxes are confident detections with no label, often a
   lesion nobody boxed. **Yellow** boxes are labelled lesions the model misses, often a
   loose, ambiguous or wrong box. Fix them in Roboflow, re-export, rerun
   `make_clean_split.py` and retrain. Inconsistent labels cap confidence more than any
   hyper-parameter can.
2. **Tight, consistent boxes.** Because the score target is IoU, a dataset where some annotators
   box the lesion only and others include the halo will never score high.
3. **More resolution** for small lesions: `imgsz=960 batch=8` (predict at 960 too).
4. **Bigger model** if speed allows: `model=weights/yolo26m.pt`.
5. **More real (not copied) photos**, especially of the situations in the error export.

## Performance notes (this PC)

* `D:` is a USB hard drive (5–30 MB/s, under 2 MB/s for many small files). A first test run
  from it managed only 3–6 s per training step with the GPU mostly idle, and importing PyTorch
  from a venv there took more than 10 minutes. So the venv (`setup_env.ps1`) and the training
  copy of the images (`make_clean_split.py`) both live on the internal SSD. Photos larger than
  1600 px are downscaled once in that copy, so they aren't re-decoded at full size every epoch.
  The code, configs and `runs/` stay on `D:`.
* RTX 5050 (Blackwell) needs a CUDA 12.8+ PyTorch build (`cu128` index; handled by the setup
  script). `batch=16` at `imgsz=800` used 7.3 of 8 GB, so the config uses `batch=12` (5.6 GB).
  If you get out-of-memory errors, use `batch=8`.
* `workers: 2`. On Windows every dataloader worker is a full PyTorch process (about 1.3 GB
  of committed memory), and validation starts twice as many. With 6 workers the run
  crashed with `error code: <1455>` ("paging file is too small"). With 2 it ran stably and
  faster (4 it/s instead of 1.2). Peak memory commit was still about 32 GB, because your other
  apps use about 15 GB. **Close heavy apps (browsers, Discord, VS Code) while training.** If you
  still hit error 1455, increase the Windows pagefile (System → Advanced system settings →
  Performance → Virtual memory) or use `workers=1`.
* Always predict at the training resolution (the scripts default to it). A different `imgsz`
  changes the apparent lesion size and lowers the scores.
