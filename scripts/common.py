"""Shared helpers for the training / evaluation scripts."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def project_path(p: str | Path) -> Path:
    """Resolve a path given relative to the project root (absolute and ~ paths pass through)."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else (ROOT / p).resolve()


def resolve_data_yaml(cfg: str | Path) -> Path:
    """Write a copy of a dataset yaml whose `path` is absolute.

    Ultralytics resolves a relative `path` against the current working directory
    (or its global datasets dir), not against the yaml file, so we pin it here.
    """
    cfg = project_path(cfg)
    data = load_yaml(cfg)
    data["path"] = str((cfg.parent / data.get("path", ".")).resolve())
    out = RUNS / "_resolved" / cfg.name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out


def split_dirs(data_yaml: str | Path, split: str) -> tuple[Path, Path]:
    """Return (images_dir, labels_dir) for a split name: train | val | test."""
    data = load_yaml(resolve_data_yaml(data_yaml))
    img_dir = Path(data["path"]) / data[split]
    lbl_dir = img_dir.parent / "labels" if img_dir.name == "images" else img_dir
    return img_dir, lbl_dir


def parse_overrides(pairs: list[str]) -> dict:
    """Parse CLI overrides like ["epochs=50", "imgsz=800", "cos_lr=false"]."""
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"Override must look like key=value, got: {pair!r}")
        k, v = pair.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def read_yolo_labels(label_file: Path) -> list[list[float]]:
    """Read a YOLO label file -> [[cls, xc, yc, w, h], ...] (normalized). Missing file -> []."""
    if not label_file.exists():
        return []
    rows = []
    for line in label_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts:
            rows.append([float(x) for x in parts])
    return rows


def row_to_xywhn(row: list[float]) -> list[float] | None:
    """[cls, xc, yc, w, h] or polygon [cls, x1, y1, x2, y2, ...] -> [cls, xc, yc, w, h].

    Polygons are reduced to their axis-aligned box, exactly as Ultralytics does. Returns None for
    malformed rows.
    """
    if len(row) == 5:
        return row
    if len(row) > 6 and len(row) % 2 == 1:
        xs, ys = row[1::2], row[2::2]
        return [row[0], (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)]
    return None


def xywhn_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> list[float]:
    return [(xc - w / 2) * img_w, (yc - h / 2) * img_h, (xc + w / 2) * img_w, (yc + h / 2) * img_h]
