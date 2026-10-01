"""Find lesions the labels miss, using a trained model's confident detections, and review them by eye.

    python scripts/label_review.py mine runs/train/yolo26s_stemcanker_v2/weights/best.pt --out runs/label_review_v3
    #  ... look at runs/label_review_v3/sheet_XX.jpg, write the real-lesion ids into accepted.txt ...
    python scripts/label_review.py accept runs/label_review_v3 --fixes configs/label_fixes_v3.csv

    # big lesions the model is unsure about (it rarely scores large cankers >= 0.3):
    python scripts/label_review.py mine best.pt --out runs/label_review_v3_big --conf 0.1 --min-size 0.08

`mine` runs the model on the chosen splits. Every detection with conf >= --conf that touches no label
(IoU < 0.1, less than half of either box inside the other) becomes a candidate. Candidates go to
candidates.csv and onto crop sheets. RED = candidate, GREEN = existing labels, yellow = candidate id.

`accept` turns the ids listed in accepted.txt (whitespace separated, `#` comments allowed) into a
fixes csv for scripts/apply_label_fixes.py.

The test split is left out by default. If test labels are fixed with the model's own proposals, the
test score rewards agreeing with that model, and runs can no longer be compared on the same test set.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from common import project_path, read_yolo_labels, row_to_xywhn, split_dirs, xywhn_to_xyxy

FOLDERS = {"train": "train", "val": "valid", "test": "test"}  # split name -> dataset folder name
TILE, COLS, ROWS = 224, 8, 5


def touches(cand: np.ndarray, gts: np.ndarray) -> bool:
    """True if candidate xyxy overlaps any label: IoU >= 0.1 or > 50% of either box inside the other."""
    if not len(gts):
        return False
    lt, rb = np.maximum(cand[:2], gts[:, :2]), np.minimum(cand[2:], gts[:, 2:])
    inter = np.clip(rb - lt, 0, None).prod(1)
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    a, g = area(cand), area(gts)
    iou = inter / (a + g - inter + 1e-9)
    return bool(((iou >= 0.1) | (inter > 0.5 * a) | (inter > 0.5 * g)).any())


def tile(img: np.ndarray, box: np.ndarray, gts: np.ndarray, cid: int, conf: float) -> np.ndarray:
    """Crop around a candidate with context; draw candidate red, labels green, id yellow."""
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
    cv2.putText(out, f"{cid} {conf:.2f}", (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(out, f"{cid} {conf:.2f}", (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return out


def mine(args):
    out = project_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(project_path(args.weights))
    data = args.data or model.ckpt.get("train_args", {}).get("data")
    cands, tiles = [], []
    for split in args.splits:
        img_dir, lbl_dir = split_dirs(data, split)
        for r in model.predict(source=str(img_dir), imgsz=args.imgsz, conf=args.conf, stream=True, verbose=False):
            h, w = r.orig_shape
            rows = [x for x in map(row_to_xywhn, read_yolo_labels(lbl_dir / f"{Path(r.path).stem}.txt")) if x]
            gts = np.array([xywhn_to_xyxy(*x[1:5], w, h) for x in rows], float).reshape(-1, 4)
            for box, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                if touches(box, gts) or max(box[2] - box[0], box[3] - box[1]) < args.min_size * max(w, h):
                    continue
                cid = len(cands)
                x1, y1, x2, y2 = box / [w, h, w, h]
                cands.append(dict(id=cid, split=split, image=Path(r.path).name, x1=f"{x1:.6f}", y1=f"{y1:.6f}",
                                  x2=f"{x2:.6f}", y2=f"{y2:.6f}", conf=f"{conf:.4f}"))
                tiles.append(tile(r.orig_img, box, gts, cid, float(conf)))
        print(f"  {split}: {sum(c['split'] == split for c in cands)} candidates")

    with open(out / "candidates.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(cands[0]) if cands else ["id"])
        wr.writeheader()
        wr.writerows(cands)
    per = COLS * ROWS
    for i in range(0, len(tiles), per):
        chunk = tiles[i : i + per] + [np.zeros((TILE, TILE, 3), np.uint8)] * (per - len(tiles[i : i + per]))
        sheet = np.vstack([np.hstack(chunk[r * COLS : (r + 1) * COLS]) for r in range(ROWS)])
        cv2.imwrite(str(out / f"sheet_{i // per:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"{len(cands)} candidates on {(len(tiles) + per - 1) // per} sheets in {out}")


def accept(args):
    review = project_path(args.review_dir)
    text = (review / "accepted.txt").read_text(encoding="utf-8")
    ids = {int(t) for line in text.splitlines() for t in re.findall(r"\d+", line.split("#", 1)[0])}
    with open(review / "candidates.csv", newline="", encoding="utf-8") as f:
        cands = [c for c in csv.DictReader(f) if int(c["id"]) in ids]
    fixes = project_path(args.fixes)
    with open(fixes, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["split", "image", "xc", "yc", "w", "h", "model_conf", "review_id"])
        for c in cands:
            x1, y1, x2, y2 = (float(c[k]) for k in ("x1", "y1", "x2", "y2"))
            wr.writerow([FOLDERS[c["split"]], c["image"], f"{(x1 + x2) / 2:.6f}", f"{(y1 + y2) / 2:.6f}",
                         f"{x2 - x1:.6f}", f"{y2 - y1:.6f}", c["conf"], c["id"]])
    print(f"{len(cands)} accepted boxes -> {fixes}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mine", help="make candidates.csv and crop sheets")
    m.add_argument("weights")
    m.add_argument("--out", required=True)
    m.add_argument("--data", help="default: the dataset the weights were trained on")
    m.add_argument("--splits", nargs="+", default=["train", "val"], choices=list(FOLDERS))
    m.add_argument("--conf", type=float, default=0.3)
    m.add_argument("--imgsz", type=int, default=800)
    m.add_argument("--min-size", type=float, default=0.0, help="skip boxes whose longer side is below this share of the image")
    a = sub.add_parser("accept", help="accepted.txt -> fixes csv")
    a.add_argument("review_dir")
    a.add_argument("--fixes", required=True)
    args = ap.parse_args()
    mine(args) if args.cmd == "mine" else accept(args)


if __name__ == "__main__":
    main()
