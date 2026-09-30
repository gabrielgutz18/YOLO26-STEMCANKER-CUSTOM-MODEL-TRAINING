"""Build a leak-free train / val / test split.

The Roboflow export contains the same photo several times (resized, re-encoded, flipped,
pre-augmented copies) scattered across splits at random: ~40% of the test images have a
near-copy in train. The model has effectively already seen those, so val/test numbers -
including confidence - look better than what you will get on new field photos, and `best.pt`
is picked partly for memorisation.

This groups near-duplicates (border-trimmed 256-bit dHash, flips included, plus the source photo
name encoded in the Roboflow filename) and re-splits by *group*, so every copy of a photo lands
in the same split. The original export is untouched.

The copy goes to the internal SSD by default (this project sits on a slow USB drive, which
starved the GPU), and photos larger than --max-side are downscaled once. Training at imgsz<=1600
never uses more pixels than that anyway, and YOLO labels are normalised so they stay valid;
decoding 3000-4000 px phone photos every epoch was the other half of the slowdown.

    python scripts/make_clean_split.py
    python scripts/train.py            # configs/train.yaml already points at the clean split
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageChops, ImageOps

from common import IMG_EXTS, ROOT, load_yaml, project_path, resolve_data_yaml

SPLITS = {"train": "train", "val": "valid", "test": "test"}  # yaml key -> folder name


def trim_border(im: Image.Image) -> Image.Image:
    """Crop uniform letterbox borders (black/grey padding from pre-augmented copies)."""
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    box = ImageChops.add(diff, diff, 2.0, -20).getbbox()
    if box and (box[2] - box[0]) * (box[3] - box[1]) > 0.2 * im.size[0] * im.size[1]:
        return im.crop(box)
    return im


def dhash_bits(im: Image.Image, size: int = 16) -> np.ndarray:
    g = np.asarray(im.resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


def image_hashes(path: Path) -> list[np.ndarray]:
    with Image.open(path) as im:
        im.draft("L", (256, 256))  # fast reduced-size JPEG decode
        # Apply EXIF rotation (as Ultralytics does): the same photo exists both as an EXIF-rotated
        # original and as a physically rotated re-export.
        im = trim_border(ImageOps.exif_transpose(im).convert("L"))
        return [dhash_bits(x) for x in (im, ImageOps.mirror(im), ImageOps.flip(im))]


def source_name(filename: str) -> str:
    """'268__fliph__noise0-01_jpg.rf.XYZ.jpg' -> '268';  '70_jpeg_jpg.rf.XYZ.jpg' -> '70'."""
    stem = filename.split(".rf.")[0]
    return re.split(r"_(?:jpe?g|png)", stem)[0].split("__")[0]


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)  # same drive: no extra space
    except OSError:
        shutil.copy2(src, dst)


def write_image(src: Path, dst: Path, max_side: int) -> bool:
    """Copy an image, downscaling it if its long side exceeds max_side. Returns True if resized."""
    with Image.open(src) as im:
        if not max_side or max(im.size) <= max_side:
            link_or_copy(src, dst)
            return False
        exif = im.info.get("exif")  # keep the orientation tag so the image loads exactly as before
        fmt = im.format
        im.draft(im.mode, (max_side, max_side))  # fast JPEG decode at >= target size
        im = im.convert("RGB") if im.mode not in ("RGB", "L") else im
        im.thumbnail((max_side, max_side), Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        kwargs = {"quality": 95, **({"exif": exif} if exif else {})} if fmt == "JPEG" else {}
        im.save(dst, format=fmt, **kwargs)
        return True


def running_training() -> list[str]:
    """Command lines of live `train.py` processes (they read the dataset this script rewrites)."""
    import psutil  # installed with ultralytics

    hits = []
    for p in psutil.process_iter(["name", "cmdline"]):
        args = p.info["cmdline"] or []
        if "python" in (p.info["name"] or "").lower() and any(Path(a).name == "train.py" for a in args[1:]):
            hits.append(f"pid {p.pid}: {' '.join(args)[:120]}")
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="configs/dataset.yaml", help="source dataset yaml")
    ap.add_argument("--out", default="~/datasets/avx-dragonfruit/stemcanker_clean", help="put this on a fast local disk")
    ap.add_argument("--out-yaml", default="configs/dataset_clean.yaml")
    ap.add_argument("--max-side", type=int, default=1600, help="downscale larger photos to this long side (0 = keep)")
    ap.add_argument("--ratios", type=float, nargs=3, metavar=("TRAIN", "VAL", "TEST"), help="default: keep the original proportions")
    ap.add_argument("--max-bits", type=int, default=16, help="dHash distance (of 256) to call two images copies")
    ap.add_argument("--no-name-groups", action="store_true", help="group by image content only, not by filename")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="run even if a training process is alive")
    args = ap.parse_args()

    # This deletes and rewrites the dataset folder: doing that under a live training run crashes it
    # ("Image Not Found"), and a changed split also invalidates --resume of that run.
    if not args.force and (live := running_training()):
        raise SystemExit("Training is running - stop it first (or pass --force):\n  " + "\n  ".join(live))

    data = load_yaml(resolve_data_yaml(args.data))
    src_root = Path(data["path"])
    items = []  # (original split key, image path)
    for key in SPLITS:
        img_dir = src_root / data[key]
        items += [(key, p) for p in sorted(img_dir.iterdir()) if p.suffix.lower() in IMG_EXTS]
    print(f"hashing {len(items)} images ...")
    hashes = [image_hashes(p) for _, p in items]

    # ---- union-find over near-duplicates -------------------------------------------------------
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    originals = np.stack([h[0] for h in hashes])
    for i, hs in enumerate(hashes):
        for h in hs:  # as-is, mirrored, flipped
            for j in np.nonzero((originals != h).sum(1) <= args.max_bits)[0]:
                parent[find(i)] = find(int(j))
    if not args.no_name_groups:
        first = {}
        for i, (_, p) in enumerate(items):
            parent[find(i)] = find(first.setdefault(source_name(p.name), i))

    groups = defaultdict(list)
    for i in range(len(items)):
        groups[find(i)].append(i)
    groups = list(groups.values())
    leaky = sum(len({items[i][0] for i in g}) > 1 for g in groups)
    print(f"{len(groups)} groups (largest {max(map(len, groups))} images); {leaky} groups currently span more than one split")

    # ---- assign whole groups to splits, largest first, to whichever split is furthest below target
    counts = {k: sum(s == k for s, _ in items) for k in SPLITS}
    ratios = dict(zip(SPLITS, args.ratios)) if args.ratios else {k: counts[k] / len(items) for k in SPLITS}
    target = {k: ratios[k] / sum(ratios.values()) * len(items) for k in SPLITS}
    random.Random(args.seed).shuffle(groups)
    groups.sort(key=len, reverse=True)
    filled = dict.fromkeys(SPLITS, 0)
    assignment = {}
    for g in groups:
        k = max(SPLITS, key=lambda s: (target[s] - filled[s]) / target[s])
        filled[k] += len(g)
        for i in g:
            assignment[i] = k

    # ---- materialise -----------------------------------------------------------------------------
    out = project_path(Path(args.out).expanduser())
    if out.exists():
        shutil.rmtree(out)
    print(f"writing {out} ...")
    rows, resized = [], 0
    for gid, g in enumerate(groups):
        for i in g:
            old, img = items[i]
            new = assignment[i]
            resized += write_image(img, out / SPLITS[new] / "images" / img.name, args.max_side)
            lbl = img.parent.parent / "labels" / f"{img.stem}.txt"
            if lbl.exists():
                link_or_copy(lbl, out / SPLITS[new] / "labels" / lbl.name)
            rows.append([gid, new, old, img.name])
    with open(out / "split_manifest.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["group", "split", "original_split", "image"], *sorted(rows)])

    out_yaml = project_path(args.out_yaml)
    try:
        data_path = os.path.relpath(out, out_yaml.parent)
    except ValueError:  # different drive: must be absolute
        data_path = str(out)
    cfg = {
        "path": data_path.replace("\\", "/"),
        **{k: f"{SPLITS[k]}/images" for k in SPLITS},
        "names": data["names"],
    }
    out_yaml.write_text(
        "# Leak-free split generated by scripts/make_clean_split.py - do not edit by hand.\n"
        + yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8",
    )
    moved = sum(r[1] != r[2] for r in rows)
    print(f"images per split: {filled}  ({moved} moved from their original split, {resized} downscaled to <= {args.max_side}px)")
    print(f"wrote {out}  and  {out_yaml.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
