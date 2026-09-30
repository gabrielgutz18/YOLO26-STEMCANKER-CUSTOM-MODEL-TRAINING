"""Run the trained model on new images, folders or videos.

    python scripts/predict.py --source path/to/photos
    python scripts/predict.py --source field.jpg --conf 0.4
    python scripts/predict.py --weights runs/train/exp640/weights/best.pt --source clip.mp4

`--conf auto` (default) uses the best-F1 threshold measured by evaluate.py on the test split.
Always predict at the resolution the model was trained at (the default): a mismatched --imgsz
shrinks or blows up lesions relative to what the model learned and lowers its scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from common import project_path


def auto_conf(weights: Path) -> float:
    report = weights.parent.parent / "eval_test" / "confidence_report.json"
    if report.exists():
        conf = json.loads(report.read_text(encoding="utf-8"))["best_f1_conf"]
        print(f"conf=auto -> {conf:.2f} (best-F1 threshold from {report})")
        return conf
    print("conf=auto -> 0.25 (no evaluation report found; run scripts/evaluate.py first)")
    return 0.25


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="runs/train/yolo26s_stemcanker/weights/best.pt")
    ap.add_argument("--source", required=True, help="image, folder, video, glob or URL")
    ap.add_argument("--conf", default="auto", help="score threshold, or 'auto'")
    ap.add_argument("--imgsz", type=int, help="default: training resolution")
    ap.add_argument("--save-txt", action="store_true", help="also write YOLO-format .txt results with scores")
    ap.add_argument("--name", default="predict")
    ap.add_argument("--device", default=None, help="e.g. 0 or cpu")
    args = ap.parse_args()

    weights = project_path(args.weights)
    model = YOLO(weights)
    conf = auto_conf(weights) if args.conf == "auto" else float(args.conf)
    imgsz = args.imgsz or model.overrides.get("imgsz", 640)

    results = model.predict(
        source=args.source,
        conf=conf,
        imgsz=imgsz,
        device=args.device,
        save=True,
        save_txt=args.save_txt,
        save_conf=True,
        project=str(project_path("runs/predict")),
        name=args.name,
        stream=True,
        verbose=False,
    )
    n_img = n_det = 0
    save_dir = None
    for r in results:
        n_img += 1
        scores = r.boxes.conf.cpu().numpy()
        n_det += len(scores)
        save_dir = r.save_dir
        detail = f"max {scores.max():.2f}, mean {scores.mean():.2f}" if len(scores) else "-"
        print(f"{Path(r.path).name}: {len(scores)} lesion(s)  {detail}")
    print(f"\n{n_img} frame(s), {n_det} detection(s) at conf>={conf:.2f}, imgsz={imgsz} -> {save_dir}")


if __name__ == "__main__":
    main()
