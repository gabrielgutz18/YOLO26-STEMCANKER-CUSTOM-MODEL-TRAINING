"""Train YOLO26-small on the dragon-fruit stem-canker dataset.

    python scripts/train.py                                  # configs/train.yaml as-is
    python scripts/train.py epochs=100 imgsz=640 name=exp640 # override any Ultralytics train arg
    python scripts/train.py --config configs/train_baseline.yaml
    python scripts/train.py --resume runs/train/yolo26s_stemcanker/weights/last.pt
    python scripts/train.py --resume runs/train/yolo26s_stemcanker/weights/last.pt workers=1  # after an OOM crash

After training, the best checkpoint is evaluated on the *test* split and a
confidence report is written next to the weights (see evaluate.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from common import load_yaml, parse_overrides, project_path, resolve_data_yaml
from evaluate import evaluate


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/train.yaml", help="training hyper-parameter yaml")
    ap.add_argument("--resume", metavar="LAST_PT", help="resume an interrupted run from its last.pt")
    ap.add_argument("--no-eval", action="store_true", help="skip the test-set confidence report")
    ap.add_argument("overrides", nargs="*", metavar="key=value", help="Ultralytics train-arg overrides")
    args = ap.parse_args()

    if args.resume:
        # Ultralytics accepts a few overrides on resume (workers, batch, cache, patience, ...), e.g.
        # workers=1 after a Windows out-of-memory crash; anything else keeps the checkpoint's value.
        model = YOLO(project_path(args.resume))
        model.train(resume=True, **parse_overrides(args.overrides))
    else:
        cfg = load_yaml(project_path(args.config)) | parse_overrides(args.overrides)
        if not project_path(cfg["data"]).exists():
            raise SystemExit(f"{cfg['data']} not found - run `python scripts/make_clean_split.py` first")
        cfg["data"] = str(resolve_data_yaml(cfg["data"]))
        cfg["project"] = str(project_path(cfg.get("project", "runs/train")))
        weights = project_path(cfg.pop("model"))  # auto-downloaded to weights/ on first use
        weights.parent.mkdir(parents=True, exist_ok=True)

        model = YOLO(weights)
        model.train(**cfg)

    best = Path(model.trainer.best)
    print(f"\nBest weights: {best}")
    if not args.no_eval and best.exists():
        evaluate(best, data=model.trainer.args.data, split="test", imgsz=model.trainer.args.imgsz)


if __name__ == "__main__":  # required on Windows: dataloader workers re-import this module
    main()
