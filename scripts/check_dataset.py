"""Audit the dataset before training.

Low confidence scores are very often a *label* problem, not a model problem:
duplicate boxes, inconsistent box tightness, near-invisible boxes, or the same
image appearing in train and test. This script surfaces those issues.

    python scripts/check_dataset.py
    python scripts/check_dataset.py --imgsz 800
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from common import IMG_EXTS, load_yaml, read_yolo_labels, resolve_data_yaml, row_to_xywhn

SPLITS = ("train", "val", "test")


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between two sets of xywh (normalized) boxes -> (len(a), len(b))."""
    a1, a2 = a[:, :2] - a[:, 2:] / 2, a[:, :2] + a[:, 2:] / 2
    b1, b2 = b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2
    lt = np.maximum(a1[:, None], b1[None])
    rb = np.minimum(a2[:, None], b2[None])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a, area_b = a[:, 2:].prod(-1), b[:, 2:].prod(-1)
    return inter / (area_a[:, None] + area_b[None] - inter + 1e-9)


def dhash(im: Image.Image, size: int = 8) -> int:
    """64-bit difference hash; robust to resizing / re-compression."""
    g = np.asarray(im.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: np.uint64, b: np.ndarray) -> np.ndarray:
    x = np.bitwise_xor(b, a)
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)


def pct(values, q):
    return float(np.percentile(values, q)) if len(values) else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="configs/dataset.yaml")
    ap.add_argument("--imgsz", type=int, default=640, help="training size used to express box sizes in pixels")
    ap.add_argument("--dup-iou", type=float, default=0.9, help="IoU above which two GT boxes count as duplicates")
    ap.add_argument("--near-dup-bits", type=int, default=4, help="max dHash distance (of 64) to call two images near-duplicates")
    args = ap.parse_args()

    data = load_yaml(resolve_data_yaml(args.data))
    root, nc = Path(data["path"]), len(data["names"])
    hashes: dict[str, list[str]] = defaultdict(list)
    dhashes: list[tuple[int, str]] = []
    problems: Counter = Counter()
    info: Counter = Counter()
    examples: dict[str, list[str]] = defaultdict(list)

    def flag(kind: str, where: str):
        problems[kind] += 1
        if len(examples[kind]) < 5:
            examples[kind].append(where)

    for split in SPLITS:
        if split not in data:
            continue
        img_dir = root / data[split]
        lbl_dir = img_dir.parent / "labels"
        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        n_boxes, min_side_px, sizes = [], [], Counter()

        for img in images:
            where = f"{split}/{img.name}"
            hashes[hashlib.md5(img.read_bytes()).hexdigest()].append(where)
            try:
                with Image.open(img) as im:
                    im = ImageOps.exif_transpose(im)  # labels refer to the EXIF-rotated image, as in Ultralytics
                    w, h = im.size
                    dhashes.append((dhash(im), where))
            except Exception:
                flag("corrupt image", where)
                continue
            sizes[(w, h)] += 1

            lbl = lbl_dir / f"{img.stem}.txt"
            if not lbl.exists():
                flag("missing label file", where)
                n_boxes.append(0)
                continue
            rows = read_yolo_labels(lbl)
            if not rows:
                flag("empty label file (background image)", where)
            # Ultralytics treats a file as polygons if ANY line is a polygon; plain bbox lines in such a
            # file are then mis-read as 2-point polygons and become wrong boxes.
            if any(len(r) > 6 for r in rows) and any(len(r) == 5 for r in rows):
                flag("MIXED bbox + polygon lines in one file (bboxes will be corrupted)", where)
            boxes = []
            for raw in rows:
                r = row_to_xywhn(raw)
                if r is None:
                    flag("malformed label line", where)
                    continue
                if len(raw) > 6:
                    info["polygon label lines (Ultralytics converts them to boxes)"] += 1
                c, xc, yc, bw, bh = r
                if not (0 <= c < nc and c == int(c)):
                    flag("bad class id", where)
                if bw <= 0 or bh <= 0 or not all(0 <= v <= 1 for v in (xc, yc, bw, bh)):
                    flag("box out of [0,1] or zero size", where)
                    continue
                boxes.append([xc, yc, bw, bh])
                scale = args.imgsz / max(w, h)
                min_side_px.append(min(bw * w, bh * h) * scale)
            n_boxes.append(len(boxes))

            if len(boxes) > 1:
                b = np.array(boxes)
                iou = np.triu(box_iou(b, b), k=1)
                for _ in range(int((iou > args.dup_iou).sum())):
                    flag(f"duplicate GT boxes (IoU>{args.dup_iou})", where)

        ms = np.array(min_side_px)
        print(f"\n=== {split}: {len(images)} images, {int(np.sum(n_boxes))} boxes ===")
        print(f"  boxes/image   mean {np.mean(n_boxes):.2f} | median {np.median(n_boxes):.0f} | max {np.max(n_boxes)}")
        print(f"  box min-side @{args.imgsz}px  p10 {pct(ms, 10):.1f} | p50 {pct(ms, 50):.1f} | p90 {pct(ms, 90):.1f}")
        if len(ms):
            print(f"  tiny boxes    <8px {np.mean(ms < 8):.1%} | <16px {np.mean(ms < 16):.1%}")
        print(f"  image sizes   {', '.join(f'{w}x{h} ({n})' for (w, h), n in sizes.most_common(5))}")

    for group in hashes.values():
        if len(group) > 1:
            kind = "identical image in >1 split (leakage)" if len({g.split('/')[0] for g in group}) > 1 else "identical image within a split"
            flag(kind, " == ".join(group))

    # Near-duplicates (same photo re-encoded / resized) across splits inflate val/test scores.
    codes = np.array([c for c, _ in dhashes], dtype=np.uint64)
    names = [n for _, n in dhashes]
    for i in range(len(codes)):
        dist = hamming(codes[i], codes[i + 1 :])
        for j in np.nonzero(dist <= args.near_dup_bits)[0] + i + 1:
            if names[i].split("/")[0] != names[j].split("/")[0]:
                flag("near-duplicate image across splits (possible leakage)", f"{names[i]} ~ {names[j]}")

    print("\n=== issues ===")
    if not problems:
        print("  none found")
    for kind, n in problems.most_common():
        print(f"  {kind}: {n}")
        for e in examples[kind]:
            print(f"      e.g. {e}")
    for kind, n in info.items():
        print(f"  (info) {kind}: {n}")

    print(
        "\nHints:\n"
        "  * Many boxes <16px at the training size -> train/predict at a larger --imgsz (800-960).\n"
        "  * Duplicate GT boxes and leakage should be fixed at the source (Roboflow) and re-exported.\n"
        "  * Inconsistent labelling (some lesions boxed, similar ones not) caps confidence more than any\n"
        "    hyper-parameter. Use `evaluate.py --export-errors` after training to find those images."
    )


if __name__ == "__main__":
    main()
