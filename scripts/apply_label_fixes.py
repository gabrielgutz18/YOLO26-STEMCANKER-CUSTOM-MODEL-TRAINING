"""Build the label-corrected dataset (v2) from the leak-free split.

The labels miss many lesions: the first model (yolo26s_stemcanker-2) made 867 confident
(conf >= 0.3) detections that touched no label at all. Each one was reviewed by eye on crop sheets
(`runs/label_review_v2/`). The 612 that are real lesions are listed in `configs/label_fixes_v2.csv`
and get added here. Rejected: areoles/spines, insects, debris, glare, padding edges, boxes that merge
two lesions, and faint white spots that could not be told apart from non-disease marks.

On top of that, every label file is normalised:
  * 4-point polygons -> plain `x y w h` boxes (removes the "Box and segment counts" warning)
  * boxes clipped to the image, zero-size boxes dropped
  * duplicated boxes (IoU > 0.95) dropped

Images are hard-linked from the clean split (same drive, no extra space), and the split is unchanged,
so the same photos stay in train / val / test and old and new models can be compared on the test set.

    python scripts/apply_label_fixes.py
    python scripts/train.py --config configs/train_v2.yaml

Round 3 adds a second review on top of round 2 (see scripts/label_review.py):

    python scripts/apply_label_fixes.py --fixes configs/label_fixes_v2.csv configs/label_fixes_v3.csv
        --out ~/datasets/avx-dragonfruit/stemcanker_v3 --out-yaml configs/dataset_v3.yaml   (one line)
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from common import IMG_EXTS, ROOT, project_path, read_yolo_labels, row_to_xywhn
from make_clean_split import link_or_copy, running_training

SPLIT_DIRS = {"train": "train", "val": "valid", "test": "test"}  # yaml key -> folder name


def xyxy(b: np.ndarray) -> np.ndarray:
    return np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = xyxy(a), xyxy(b)
    lt, rb = np.maximum(a[:, None, :2], b[None, :, :2]), np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-12)


def clean_boxes(rows: list[list[float]], stats: dict) -> list[list[float]]:
    """[[cls, xc, yc, w, h] | polygon rows] -> clipped, de-duplicated [[cls, xc, yc, w, h]]."""
    out = []
    for row in rows:
        if len(row) != 5:
            stats["polygons converted"] += 1
        box = row_to_xywhn(row)
        if box is None:
            stats["malformed dropped"] += 1
            continue
        c, xc, yc, w, h = box
        x1, y1, x2, y2 = max(xc - w / 2, 0), max(yc - h / 2, 0), min(xc + w / 2, 1), min(yc + h / 2, 1)
        if x2 - x1 <= 1e-4 or y2 - y1 <= 1e-4:
            stats["zero-size dropped"] += 1
            continue
        cand = [int(c), (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
        if out and iou(np.array([cand[1:]]), np.array([b[1:] for b in out])).max() > 0.95:
            stats["duplicates dropped"] += 1
            continue
        out.append(cand)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="~/datasets/avx-dragonfruit/stemcanker_clean", help="output of make_clean_split.py")
    ap.add_argument("--fixes", nargs="+", default=["configs/label_fixes_v2.csv"], help="reviewed boxes to add (one or more csv)")
    ap.add_argument("--out", default="~/datasets/avx-dragonfruit/stemcanker_v2")
    ap.add_argument("--out-yaml", default="configs/dataset_v2.yaml")
    ap.add_argument("--force", action="store_true", help="run even if a training process is alive")
    args = ap.parse_args()

    if not args.force and (live := running_training()):
        raise SystemExit("Training is running - stop it first (or pass --force):\n  " + "\n  ".join(live))

    src, out = Path(args.src).expanduser(), Path(args.out).expanduser()
    if not src.exists():
        raise SystemExit(f"{src} not found - run `python scripts/make_clean_split.py` first")

    fixes = defaultdict(list)
    for fixes_csv in args.fixes:
        with open(project_path(fixes_csv), newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                fixes[(r["split"], r["image"])].append([0, *(float(r[k]) for k in ("xc", "yc", "w", "h"))])

    if out.exists():
        shutil.rmtree(out)
    print(f"writing {out} ...")
    applied = 0
    for folder in SPLIT_DIRS.values():
        stats = defaultdict(int)
        for img in sorted((src / folder / "images").iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            link_or_copy(img, out / folder / "images" / img.name)
            rows = read_yolo_labels(src / folder / "labels" / f"{img.stem}.txt")
            stats["boxes before"] += len(rows)
            boxes = clean_boxes(rows, stats)
            added = fixes.pop((folder, img.name), [])
            boxes += added
            stats["reviewed boxes added"] += len(added)
            stats["boxes after"] += len(boxes)
            lbl = out / folder / "labels" / f"{img.stem}.txt"
            lbl.parent.mkdir(parents=True, exist_ok=True)
            lbl.write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in boxes), encoding="utf-8")
        applied += stats["reviewed boxes added"]
        print(f"  {folder:6s} " + ", ".join(f"{k} {v}" for k, v in stats.items()))
    if fixes:
        print(f"WARNING {sum(map(len, fixes.values()))} fixes matched no image (dataset re-split since the review?):")
        for split, name in list(fixes)[:5]:
            print(f"    {split}/{name}")
    if (src / "split_manifest.csv").exists():
        shutil.copy2(src / "split_manifest.csv", out / "split_manifest.csv")

    out_yaml = project_path(args.out_yaml)
    cfg = {"path": str(out).replace("\\", "/"), **{k: f"{v}/images" for k, v in SPLIT_DIRS.items()}, "names": {0: "stemcanker"}}
    out_yaml.write_text(
        "# Label-corrected leak-free split generated by scripts/apply_label_fixes.py - do not edit by hand.\n"
        + yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8",
    )
    print(f"{applied} reviewed boxes added; wrote {out}  and  {out_yaml.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
