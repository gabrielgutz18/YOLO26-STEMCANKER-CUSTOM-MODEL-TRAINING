"""Audit the frozen test labels with v2 + v3 proposals, then rescore both models on the audited labels.

    python runs/test_audit/test_audit.py mine     # candidates.csv + blind crop sheets
    #  ... review sheet_XX.jpg, write real-lesion ids into accepted.txt ...
    python runs/test_audit/test_audit.py score    # audited test labels + v2/v3 on old and audited labels

Fairness: candidates are the union of BOTH models' unlabelled detections, merged where they overlap,
and the sheets show only an id (no model, no confidence), so the review cannot favour either model.
Candidate = touches no label (same rule as scripts/label_review.py) and
            conf >= 0.3, or conf >= 0.1 with the longer side >= 8% of the image (round-3 "big" pass).
The frozen dataset is not touched: audited labels go to a separate copy of the test split.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from common import read_yolo_labels, row_to_xywhn, xywhn_to_xyxy  # noqa: E402
from evaluate import evaluate, iou_matrix, match, print_table  # noqa: E402
from label_review import touches  # noqa: E402

OUT = Path(__file__).resolve().parent
SRC = Path.home() / "datasets/avx-dragonfruit/stemcanker_v3"
AUDIT = Path.home() / "datasets/avx-dragonfruit/stemcanker_testaudit"
MODELS = {
    "v2": ROOT / "runs/train/yolo26s_stemcanker_v2/weights/best.pt",
    "v3": ROOT / "runs/train/yolo26s_stemcanker_v3/weights/best.pt",
}
IMGSZ, CONF, BIG_CONF, BIG_SIZE, MERGE_IOU = 800, 0.3, 0.1, 0.08, 0.3
TILE, COLS, ROWS = 256, 6, 4


def load_gt(img: Path, w: int, h: int) -> np.ndarray:
    rows = [r for r in map(row_to_xywhn, read_yolo_labels(SRC / "test/labels" / f"{img.stem}.txt")) if r]
    return np.array([xywhn_to_xyxy(*r[1:5], w, h) for r in rows], float).reshape(-1, 4)


def tile(img: np.ndarray, box: np.ndarray, gts: np.ndarray, cid: int) -> np.ndarray:
    """Like label_review.tile, but blind: id only. RED = candidate, GREEN = existing labels."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) * 3 + 48
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    l, t = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
    r, b = int(min(cx + side / 2, w)), int(min(cy + side / 2, h))
    s = TILE / max(r - l, b - t)
    crop = cv2.resize(img[t:b, l:r], (max(int((r - l) * s), 1), max(int((b - t) * s), 1)))
    out = np.full((TILE, TILE, 3), 40, np.uint8)
    out[: crop.shape[0], : crop.shape[1]] = crop
    to_tile = lambda bb: tuple(int(v) for v in ((bb[0] - l) * s, (bb[1] - t) * s, (bb[2] - l) * s, (bb[3] - t) * s))
    for g in gts:
        gx1, gy1, gx2, gy2 = to_tile(g)
        cv2.rectangle(out, (gx1, gy1), (gx2, gy2), (0, 200, 0), 1)
    px1, py1, px2, py2 = to_tile(box)
    cv2.rectangle(out, (px1, py1), (px2, py2), (0, 0, 255), 2)
    cv2.putText(out, str(cid), (3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(out, str(cid), (3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    return out


def mine():
    preds = {}  # image name -> list of (model, xyxy, conf)
    breakdown = {}
    for name, weights in MODELS.items():
        model = YOLO(weights)
        counts = {t: dict(tp=0, fp_touching=0, fp_unlabelled=0) for t in (0.3, 0.5)}
        for r in model.predict(source=str(SRC / "test/images"), imgsz=IMGSZ, conf=BIG_CONF, stream=True, verbose=False):
            h, w = r.orig_shape
            img = Path(r.path)
            gts = load_gt(img, w, h)
            xyxy, conf = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
            tp, _ = match(xyxy, conf, np.zeros(len(conf), int), gts, np.zeros(len(gts), int), 0.5)
            for t, c in counts.items():
                keep = conf >= t
                c["tp"] += int((tp & keep).sum())
                for box in xyxy[~tp & keep]:
                    c["fp_touching" if touches(box, gts) else "fp_unlabelled"] += 1
            for box, cf in zip(xyxy, conf):
                big = max(box[2] - box[0], box[3] - box[1]) >= BIG_SIZE * max(w, h)
                if touches(box, gts) or not (cf >= CONF or big):
                    continue
                preds.setdefault(img.name, []).append((name, box, float(cf)))
        breakdown[name] = counts
        print(name, json.dumps(counts))

    # merge overlapping proposals (across and within models) into one candidate per lesion
    cands, tiles = [], []
    for img_name in sorted(preds):
        groups = []
        for name, box, cf in sorted(preds[img_name], key=lambda p: -p[2]):
            for g in groups:
                if iou_matrix(box[None], np.array([m[1] for m in g]))[0].max() >= MERGE_IOU:
                    g.append((name, box, cf))
                    break
            else:
                groups.append([(name, box, cf)])
        im = cv2.imdecode(np.fromfile(str(SRC / "test/images" / img_name), np.uint8), cv2.IMREAD_COLOR)
        h, w = im.shape[:2]
        gts = load_gt(Path(img_name), w, h)
        for g in groups:
            top = {}  # best box per model, then average across models
            for name, box, cf in g:
                if name not in top or cf > top[name][1]:
                    top[name] = (box, cf)
            box = np.mean([b for b, _ in top.values()], 0)
            cid = len(cands)
            x1, y1, x2, y2 = box / [w, h, w, h]
            cands.append(dict(id=cid, image=img_name, x1=f"{x1:.6f}", y1=f"{y1:.6f}", x2=f"{x2:.6f}", y2=f"{y2:.6f}",
                              models="+".join(sorted(top)), **{f"conf_{m}": f"{top[m][1]:.4f}" if m in top else "" for m in MODELS}))
            tiles.append(tile(im, box, gts, cid))

    with open(OUT / "candidates.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(cands[0]))
        wr.writeheader()
        wr.writerows(cands)
    (OUT / "fp_breakdown.json").write_text(json.dumps(breakdown, indent=2), encoding="utf-8")
    per = COLS * ROWS
    for i in range(0, len(tiles), per):
        chunk = tiles[i : i + per] + [np.zeros((TILE, TILE, 3), np.uint8)] * (per - len(tiles[i : i + per]))
        sheet = np.vstack([np.hstack(chunk[r * COLS : (r + 1) * COLS]) for r in range(ROWS)])
        cv2.imwrite(str(OUT / f"sheet_{i // per:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    by = {k: sum(c["models"] == k for c in cands) for k in ("v2", "v3", "v2+v3")}
    print(f"{len(cands)} candidates ({by}) on {(len(tiles) + per - 1) // per} sheets in {OUT}")


def score():
    text = (OUT / "accepted.txt").read_text(encoding="utf-8")
    ids = {int(t) for line in text.splitlines() for t in re.findall(r"\d+", line.split("#", 1)[0])}
    with open(OUT / "candidates.csv", newline="", encoding="utf-8") as f:
        cands = list(csv.DictReader(f))
    added = {}
    for c in cands:
        if int(c["id"]) in ids:
            x1, y1, x2, y2 = (float(c[k]) for k in ("x1", "y1", "x2", "y2"))
            added.setdefault(c["image"], []).append(f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}\n")
    acc = [c for c in cands if int(c["id"]) in ids]
    print(f"accepted {len(acc)} of {len(cands)}: " + ", ".join(f"{k} {sum(c['models'] == k for c in acc)}/{sum(c['models'] == k for c in cands)}" for k in ("v2", "v3", "v2+v3")))

    for sub in ("images", "labels"):
        (AUDIT / "test" / sub).mkdir(parents=True, exist_ok=True)
    n_old = n_new = 0
    for img in sorted((SRC / "test/images").iterdir()):
        dst = AUDIT / "test/images" / img.name
        if not dst.exists():
            os.link(img, dst)
        old = (SRC / "test/labels" / f"{img.stem}.txt")
        lines = old.read_text(encoding="utf-8").splitlines(keepends=True) if old.exists() else []
        lines = [ln if ln.endswith("\n") else ln + "\n" for ln in lines if ln.strip()]
        n_old += len(lines)
        lines += added.get(img.name, [])
        n_new += len(lines)
        (AUDIT / "test/labels" / f"{img.stem}.txt").write_text("".join(lines), encoding="utf-8")
    cfg = {"path": str(AUDIT).replace("\\", "/"), "train": str(SRC / "train/images").replace("\\", "/"),
           "val": str(SRC / "valid/images").replace("\\", "/"), "test": "test/images", "names": {0: "stemcanker"}}
    data = OUT / "dataset_testaudit.yaml"
    data.write_text("# v3 test split + accepted test-audit boxes (runs/test_audit). Generated - do not edit.\n" + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"test labels {n_old} -> {n_new}  ({AUDIT})")

    results = {}
    for labels, d in (("frozen", ROOT / "configs/dataset_v3.yaml"), ("audited", data)):
        for name, weights in MODELS.items():
            s = evaluate(weights, d, "test", IMGSZ, out_dir=OUT / f"eval_{name}_{labels}")
            s["weights"] = f"x/{name}_{labels}/weights/best.pt"  # print_table names columns by run folder
            results[f"{name}_{labels}"] = s
    (OUT / "scores.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n=== v2 vs v3, frozen vs audited test labels ===")
    print_table(list(results.values()))


if __name__ == "__main__":
    {"mine": mine, "score": score}[sys.argv[1]]()
