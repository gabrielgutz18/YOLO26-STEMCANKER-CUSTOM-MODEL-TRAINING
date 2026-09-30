"""Accuracy + confidence report for trained weights.

    python scripts/evaluate.py runs/train/yolo26s_stemcanker/weights/best.pt
    python scripts/evaluate.py runs/train/baseline/weights/best.pt runs/train/yolo26s_stemcanker/weights/best.pt
    python scripts/evaluate.py best.pt --export-errors     # dump worst images for label review

Why not just "average confidence"?
----------------------------------
Any model can be made to output bigger numbers. What you want is: real lesions get high scores
AND non-lesions stay low. So this reports both sides (IoU >= 0.5 to count as a correct detection):

  lesion_conf       mean over ALL ground-truth lesions of the score of its detection (0 if
                    missed). The headline number: recall and confidence in one.       higher = better
  tp_conf_mean      mean score of correct detections                                  higher = better
  recall@T          share of lesions detected with score >= T                         higher = better
  precision@T       share of detections with score >= T that are real lesions         higher = better
  fp_per_img@0.5    confident false alarms per image                                  lower  = better
  best_f1_conf      the score threshold that maximises F1 -> use it as `conf` when deploying
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from common import IMG_EXTS, project_path, read_yolo_labels, resolve_data_yaml, row_to_xywhn, split_dirs, xywhn_to_xyxy

THRESHOLDS = (0.25, 0.5, 0.7)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between xyxy boxes a (N,4) and b (M,4) -> (N, M)."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def match(p_xyxy, p_conf, p_cls, g_xyxy, g_cls, iou_thr):
    """Greedy, confidence-ordered one-to-one matching (same rule as COCO / Ultralytics).

    Returns (tp mask per prediction, score of the matched prediction per GT or 0 if missed).
    Because matching is confidence-ordered, raising the threshold later never changes which
    GT a surviving prediction is matched to, so P/R/F1 at any threshold follow from these arrays.
    """
    tp = np.zeros(len(p_conf), bool)
    g_conf = np.zeros(len(g_xyxy))
    if len(p_conf) and len(g_xyxy):
        iou = iou_matrix(p_xyxy, g_xyxy)
        iou[p_cls[:, None] != g_cls[None]] = 0
        taken = np.zeros(len(g_xyxy), bool)
        for i in np.argsort(-p_conf):
            cand = np.nonzero(~taken & (iou[i] >= iou_thr))[0]
            if len(cand):
                j = cand[np.argmax(iou[i, cand])]
                taken[j], tp[i], g_conf[j] = True, True, p_conf[i]
    return tp, g_conf


def load_gt(label_file: Path, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    rows = [r for r in map(row_to_xywhn, read_yolo_labels(label_file)) if r is not None]
    xyxy = np.array([xywhn_to_xyxy(*r[1:5], w, h) for r in rows], dtype=float).reshape(-1, 4)
    return xyxy, np.array([int(r[0]) for r in rows], dtype=int)


def collect(model: YOLO, img_dir: Path, lbl_dir: Path, imgsz: int, iou_thr: float, conf_floor: float):
    """Run inference over a split and match every prediction to ground truth."""
    per_image = []
    for r in model.predict(source=str(img_dir), imgsz=imgsz, conf=conf_floor, stream=True, verbose=False):
        h, w = r.orig_shape
        g_xyxy, g_cls = load_gt(lbl_dir / f"{Path(r.path).stem}.txt", w, h)
        b = r.boxes
        p_xyxy, p_conf, p_cls = b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)
        tp, g_conf = match(p_xyxy, p_conf, p_cls, g_xyxy, g_cls, iou_thr)
        per_image.append(dict(path=r.path, gt=g_xyxy, pred=p_xyxy, conf=p_conf, tp=tp, g_conf=g_conf))
    return per_image


def confidence_stats(per_image: list[dict]) -> tuple[dict, dict]:
    conf = np.concatenate([d["conf"] for d in per_image])
    tp = np.concatenate([d["tp"] for d in per_image])
    g_conf = np.concatenate([d["g_conf"] for d in per_image])
    n_gt, n_img = len(g_conf), len(per_image)

    grid = np.linspace(0.01, 0.99, 99)
    n_pred = np.array([(conf >= t).sum() for t in grid])
    prec = np.array([(tp & (conf >= t)).sum() for t in grid]) / np.maximum(n_pred, 1)
    rec = np.array([(g_conf >= t).sum() for t in grid]) / max(n_gt, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    best = int(np.argmax(f1))

    s = {
        "images": n_img,
        "gt_lesions": n_gt,
        "lesion_conf": float(g_conf.mean()) if n_gt else 0.0,
        "tp_conf_mean": float(conf[tp].mean()) if tp.any() else 0.0,
        "tp_conf_median": float(np.median(conf[tp])) if tp.any() else 0.0,
        "best_f1": float(f1[best]),
        "best_f1_conf": float(grid[best]),
    }
    for t in THRESHOLDS:
        s[f"recall@{t}"] = float((g_conf >= t).sum() / max(n_gt, 1))
        s[f"precision@{t}"] = float((tp & (conf >= t)).sum() / max((conf >= t).sum(), 1))
    s["fp_per_img@0.5"] = float((~tp & (conf >= 0.5)).sum() / max(n_img, 1))
    curves = dict(grid=grid, precision=prec, recall=rec, f1=f1, tp_conf=conf[tp], fp_conf=conf[~tp])
    return s, curves


def plot_report(curves: dict, stats: dict, out: Path, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.linspace(0, 1, 41)
    a1.hist(curves["tp_conf"], bins=bins, alpha=0.7, label=f"correct (n={len(curves['tp_conf'])})", color="#2a9d8f")
    a1.hist(curves["fp_conf"], bins=bins, alpha=0.6, label=f"false alarm (n={len(curves['fp_conf'])})", color="#e76f51")
    a1.set_yscale("log")
    a1.set(xlabel="confidence", ylabel="detections (log)", title="Score distribution")
    a1.legend()
    for k, c in (("precision", "#264653"), ("recall", "#2a9d8f"), ("f1", "#e9c46a")):
        a2.plot(curves["grid"], curves[k], label=k, color=c, lw=2)
    a2.axvline(stats["best_f1_conf"], ls="--", color="gray")
    a2.set(xlabel="confidence threshold", ylim=(0, 1.02), title=f"best F1 {stats['best_f1']:.3f} @ conf {stats['best_f1_conf']:.2f}")
    a2.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def export_errors(per_image: list[dict], out_dir: Path, top: int, show_conf: float = 0.25):
    """Save the images the model disagrees with most, for label review.

    Red   = confident prediction with no matching label -> often an UNLABELLED lesion
    Yellow= labelled lesion the model missed / scored < show_conf -> wrong/ambiguous label or hard case
    Blue  = correct detection, Green = label
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    scored = []
    for d in per_image:
        fp = ~d["tp"] & (d["conf"] >= show_conf)
        missed = d["g_conf"] < show_conf
        scored.append((float(d["conf"][fp].sum() + missed.sum()), int(fp.sum()), int(missed.sum()), d))
    scored.sort(key=lambda x: -x[0])

    with open(out_dir / "errors.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["rank", "score", "false_alarms", "missed_lesions", "gt_lesions", "image"])
        for rank, (score, n_fp, n_miss, d) in enumerate(scored[:top], 1):
            wr.writerow([rank, f"{score:.2f}", n_fp, n_miss, len(d["gt"]), d["path"]])
            im = cv2.imdecode(np.fromfile(d["path"], np.uint8), cv2.IMREAD_COLOR)
            t = max(1, round(max(im.shape[:2]) / 400))
            for (x1, y1, x2, y2), gc in zip(d["gt"].astype(int), d["g_conf"]):
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 255) if gc < show_conf else (0, 200, 0), t * (2 if gc < show_conf else 1))
            for (x1, y1, x2, y2), c, ok in zip(d["pred"].astype(int), d["conf"], d["tp"]):
                if c < show_conf:
                    continue
                color = (255, 128, 0) if ok else (0, 0, 255)
                cv2.rectangle(im, (x1, y1), (x2, y2), color, t)
                cv2.putText(im, f"{c:.2f}", (x1, max(y1 - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * t, color, t)
            _, buf = cv2.imencode(".jpg", im)
            buf.tofile(str(out_dir / f"{rank:03d}_{Path(d['path']).name}"))
    print(f"  wrote {min(top, len(scored))} review images -> {out_dir}")


def evaluate(
    weights: str | Path,
    data: str | Path | None = None,
    split: str = "test",
    imgsz: int | None = None,
    iou: float = 0.5,
    conf_floor: float = 0.01,
    errors: int = 0,
    out_dir: str | Path | None = None,
) -> dict:
    weights = project_path(weights)
    model = YOLO(weights)
    imgsz = imgsz or model.overrides.get("imgsz", 640)  # evaluate at the training resolution
    # Default to the dataset the model was trained on: scoring on another split's test set can
    # include images the model trained on.
    data = data or model.overrides.get("data")
    if not data or not project_path(data).exists():
        data = "configs/dataset_clean.yaml"
    data_yaml = resolve_data_yaml(data)
    out_dir = Path(out_dir) if out_dir else weights.parent.parent / f"eval_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[{weights}] split={split} imgsz={imgsz}")

    m = model.val(data=str(data_yaml), split=split, imgsz=imgsz, project=str(out_dir), name="val", exist_ok=True, verbose=False, plots=True)
    stats = {"mAP50": float(m.box.map50), "mAP50-95": float(m.box.map)}

    img_dir, lbl_dir = split_dirs(data, split)
    per_image = collect(model, img_dir, lbl_dir, imgsz, iou, conf_floor)
    s, curves = confidence_stats(per_image)
    stats |= s
    stats |= {"weights": str(weights), "split": split, "imgsz": imgsz, "iou_match": iou}

    (out_dir / "confidence_report.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    plot_report(curves, stats, out_dir / "confidence_report.png", f"{weights.parent.parent.name} - {split}")
    print_table([stats])
    print(f"  report -> {out_dir}")
    if errors:
        export_errors(per_image, out_dir / "errors", errors)
    return stats


ROWS = [
    ("mAP50", "mAP@0.5"),
    ("mAP50-95", "mAP@0.5:0.95"),
    ("lesion_conf", "lesion confidence (headline)"),
    ("tp_conf_mean", "mean conf, correct dets"),
    ("tp_conf_median", "median conf, correct dets"),
    ("recall@0.5", "recall @ conf>=0.5"),
    ("recall@0.7", "recall @ conf>=0.7"),
    ("precision@0.5", "precision @ conf>=0.5"),
    ("precision@0.7", "precision @ conf>=0.7"),
    ("fp_per_img@0.5", "false alarms/img @0.5 (low=good)"),
    ("best_f1", "best F1"),
    ("best_f1_conf", "conf threshold at best F1"),
]


def print_table(results: list[dict]):
    names = [Path(r["weights"]).parent.parent.name for r in results]
    width = max(14, *(len(n) + 2 for n in names))
    print("\n" + " " * 34 + "".join(f"{n:>{width}}" for n in names))
    for key, label in ROWS:
        print(f"  {label:<32}" + "".join(f"{r[key]:>{width}.3f}" for r in results))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", nargs="+", help="one or more .pt files (several -> comparison table)")
    ap.add_argument("--data", help="default: the dataset the weights were trained on")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, help="default: the resolution the model was trained at")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU needed to count a detection as correct")
    ap.add_argument("--export-errors", type=int, nargs="?", const=40, default=0, metavar="N", help="save the N worst images for label review")
    args = ap.parse_args()

    results = [evaluate(w, args.data, args.split, args.imgsz, args.iou, errors=args.export_errors) for w in args.weights]
    if len(results) > 1:
        print("\n=== comparison ===")
        print_table(results)


if __name__ == "__main__":
    main()
