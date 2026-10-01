"""Live stem-canker detection on the laptop webcam with OpenCV.

    python scripts/webcam.py
    python scripts/webcam.py --conf 0.35 --camera 1
    python scripts/webcam.py --weights runs/train/yolo26s_stemcanker_v2/weights/best.pt
    python scripts/webcam.py --source DJI_0042.JPG --min-size 6     # drone photo, video file or rtsp:// stream
    python scripts/webcam.py --camera 1 --width 3840 --height 2160 --min-size 6   # drone video via HDMI capture card

Keys (click the video window first):
    q / Esc   quit
    + / -     raise / lower the confidence threshold by 0.05
    m         two-size detection on / off (800 + 1280 px; finds ~10% more lesions, ~2x GPU time)
    t         tile pass on / off: also detect on overlapping crops at full camera resolution (big / drone frames)
    [ / ]     smaller / bigger tiles (more / less zoom); the debug view [d] draws the tiles in blue
    g         plant check on / off
    d         debug: also show every raw model box (grey; red X = removed by the plant check or size limit)
    space     freeze / unfreeze the frame
    s         save the annotated frame to runs/webcam/
    n         save the RAW frame as a "no canker here" example (only when no canker is in view!)
              -> ~/datasets/avx-dragonfruit/negatives/, for retraining with background images

How it is tuned (measured 2026-10-01, RTX 5050 + built-in webcam):
  camera       Media Foundation runs at 30 fps here (DirectShow only 15). A reader thread always hands over
               the NEWEST frame: both backends queue old frames, which made boxes lag behind the stem.
  model        800 px (the training size) is the best single size on webcam-like frames. Adding a 1280 px
               pass and merging raises recall 0.38 -> 0.42 at the same precision; each pass takes ~14 ms.
  plant check  a box is kept only if >= 30% of a ring around it is plant green: keeps 94% of real test
               lesions; detections on a room/shirt had a median of 1% green, max ~27%. dragon.pt has never seen
               a photo without canker, so without this it boxes shirts and walls.
  tracking     a box shows after 2 detections, snaps to the newest detection, moves with the stem between
               detections (velocity), and stays ~0.25 s after the model last saw it, so it does not blink.
  size limit   boxes under 10 px are ignored: at webcam distance they are specks or frame-edge artifacts.
               Lower it (--min-size 6) on drone frames, where real lesions are 10-20 px.
  tiles        the model sees at most 1280 px of the frame, so on a 4K frame every lesion shrinks 3-5x. The tile pass
               runs the 800 px model on overlapping 800 px crops (1:1, no shrinking) plus the whole frame. It starts
               on by itself when the frame is bigger than the largest --sizes. Measured on test photos pasted into
               3840x2160 frames (lesions keep their size, median 30 px; at half size = twice as far, 15 px):
                                          recall @30 px   @15 px   time/frame
                 800 (whole frame)              0.09      0.00      16 ms
                 800+1280                       0.21      0.05      32 ms
                 800 + tiles 800 (default)      0.42      0.32     260 ms
                 800 + tiles 640 (1.25x zoom)   0.41      0.33     380 ms
                 800 + tiles 400 (2x zoom)      0.37      0.29     750 ms
               (the same photos one at a time at 800: 0.37). Zooming in past 1:1 adds no detail and does not help.

Distance matters most: with the stem filling 60% instead of 100% of the frame the model finds a third fewer
lesions. Hold it 15-30 cm from the camera in good light.
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from common import IMG_EXTS, project_path

WINDOW = "dragon.pt - stem canker (q to quit)"
NEGATIVES = Path.home() / "datasets/avx-dragonfruit/negatives"
EDGE = 6  # px: a tile-pass box this close to a tile border inside the frame is a lesion cut by that border
TILE_BATCH = 16  # crops per predict call: a 4K frame cut into 640 px tiles is 40 crops, too many for 8 GB at once


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    # Media Foundation gives 30 fps on this laptop, DirectShow 15 (exposure-limited); fall back if MSMF fails
    for api in (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY):
        cap = cv2.VideoCapture(index, api)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if cap.read()[0]:
                return cap
        cap.release()
    if index == 0:
        raise SystemExit("Could not open camera 0. Close other apps using it (Teams, Zoom, Camera) or try --camera 1")
    raise SystemExit(f"Could not open camera {index}: nothing is connected there (a drone feed needs its HDMI capture card "
                     "plugged in). Use --camera 0 for the laptop webcam, or --source <photo, video or rtsp:// URL>.")


class LatestFrame(threading.Thread):
    """Reads the camera non-stop and keeps only the newest frame, so the loop never processes stale video."""

    def __init__(self, cap: cv2.VideoCapture):
        super().__init__(daemon=True)
        self.cap, self.lock, self.fresh = cap, threading.Lock(), threading.Event()
        self.ok, self.frame, self.running = True, None, True
        self.start()

    def run(self):
        while self.running:
            ok, frame = self.cap.read()
            with self.lock:
                self.ok, self.frame = ok, frame
            self.fresh.set()
            if not ok:
                break

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.fresh.wait(2.0)
        self.fresh.clear()
        with self.lock:
            return self.ok, self.frame

    def stop(self):
        self.running = False
        self.join(timeout=2.0)
        self.cap.release()


class FileFrames:
    """A photo (served again every frame) or a video file (every frame in order, so slower than real time on big
    frames, and starting over at the end)."""

    def __init__(self, path: Path):
        self.still = cv2.imread(str(path)) if path.suffix.lower() in IMG_EXTS else None
        self.cap = None if self.still is not None else cv2.VideoCapture(str(path))
        if self.still is None and not self.cap.isOpened():
            raise SystemExit(f"Could not read {path} as a photo or a video")

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.still is not None:
            return True, self.still.copy()
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return ok, frame

    def stop(self):
        if self.cap is not None:
            self.cap.release()


def open_source(source: int | str, width: int, height: int) -> LatestFrame | FileFrames:
    """Camera index -> live camera; rtsp:// or http:// -> live stream (newest frame only); else a photo or video file."""
    if isinstance(source, int):
        return LatestFrame(open_camera(source, width, height))
    if "://" in source:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if not cap.isOpened() or not cap.read()[0]:
            raise SystemExit(f"Could not open the stream {source}")
        return LatestFrame(cap)
    path = Path(source).expanduser()
    if not path.exists():
        raise SystemExit(f"No such file: {path.resolve()}")
    return FileFrames(path)


def green_mask(img: np.ndarray) -> np.ndarray:
    """Plant-green pixels: OpenCV hue 25-95 (yellow-green .. blue-green), not grey, not dark."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return (hsv[..., 0] >= 25) & (hsv[..., 0] <= 95) & (hsv[..., 1] >= 50) & (hsv[..., 2] >= 40)


def ring_green(mask: np.ndarray, box: np.ndarray) -> float:
    """Share of green pixels in a ring around the box: the box grown by its own size on every side, minus the box."""
    H, W = mask.shape
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    outer = mask[int(max(y1 - bh, 0)) : int(min(y2 + bh, H)), int(max(x1 - bw, 0)) : int(min(x2 + bw, W))]
    inner = mask[int(max(y1, 0)) : int(min(y2, H)), int(max(x1, 0)) : int(min(x2, W))]
    return float(outer.sum() - inner.sum()) / max(outer.size - inner.size, 1)


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in a (N,4) with every box in b (M,4)."""
    lt, rb = np.maximum(a[:, None, :2], b[None, :, :2]), np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def merge(boxes: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Union of the detections from several input sizes: overlapping boxes (IoU > 0.5) keep the highest score."""
    keep = []
    for i in np.argsort(-scores):
        if not keep or iou(boxes[i : i + 1], boxes[keep]).max() <= 0.5:
            keep.append(i)
    return boxes[keep], scores[keep]


def tile_grid(W: int, H: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """(x1, y1, x2, y2) of tile x tile crops covering a W x H frame, neighbours overlapping by >= `overlap` of a tile.
    The crops are spread evenly and all have the same shape, so one batched predict covers the frame."""
    def starts(n: int) -> tuple[list[int], int]:
        t = min(tile, n)
        k = 1 + math.ceil((n - t) / (t * (1 - overlap)))
        return np.linspace(0, n - t, k).round().astype(int).tolist(), t
    (xs, tw), (ys, th) = starts(W), starts(H)
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


def detect_tiles(model: YOLO, frame: np.ndarray, imgsz: int, tile: int, overlap: float, conf: float, device,
                 quantize) -> tuple[np.ndarray, np.ndarray]:
    """Model boxes from overlapping crops of the frame, each scaled from `tile` px to `imgsz` px.

    A whole-frame pass shrinks a big (drone / 4K) frame to imgsz, and a far-away 15 px lesion becomes a 3 px speck. A
    crop keeps every camera pixel. A box touching a crop border inside the frame is a lesion cut in two: it is dropped,
    because the overlap shows that lesion whole in the neighbouring crop (lesions bigger than the overlap come from the
    whole-frame pass)."""
    H, W = frame.shape[:2]
    grid = tile_grid(W, H, tile, overlap)
    crops = [np.ascontiguousarray(frame[y1:y2, x1:x2]) for x1, y1, x2, y2 in grid]
    results = [r for i in range(0, len(crops), TILE_BATCH)
               for r in model.predict(crops[i : i + TILE_BATCH], imgsz=imgsz, conf=conf, device=device, quantize=quantize, verbose=False)]
    boxes, scores = [np.zeros((0, 4), np.float32)], [np.zeros(0, np.float32)]
    for (x1, y1, x2, y2), r in zip(grid, results):
        b = r.boxes.xyxy.cpu().numpy() + np.array([x1, y1, x1, y1], np.float32)
        cut = (((b[:, 0] < x1 + EDGE) & (x1 > 0)) | ((b[:, 1] < y1 + EDGE) & (y1 > 0))
               | ((b[:, 2] > x2 - EDGE) & (x2 < W)) | ((b[:, 3] > y2 - EDGE) & (y2 < H)))
        boxes.append(b[~cut])
        scores.append(r.boxes.conf.cpu().numpy()[~cut])
    return np.vstack(boxes), np.concatenate(scores)


def detect(model: YOLO, frame: np.ndarray, sizes: list[int], conf: float, device, quantize,
           green: float, min_size: float, tile: int = 0, overlap: float = 0.25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All model boxes >= conf for one frame (several input sizes and, with tile > 0, the tile pass merged), plus a
    mask of the ones that pass the size limit and the plant check (green = 0 turns the plant check off)."""
    parts = [model.predict(frame, imgsz=sz, conf=conf, device=device, quantize=quantize, verbose=False)[0].boxes for sz in sizes]
    boxes = [p.xyxy.cpu().numpy() for p in parts]
    scores = [p.conf.cpu().numpy() for p in parts]
    if tile:
        b, s = detect_tiles(model, frame, sizes[0], tile, overlap, conf, device, quantize)
        boxes.append(b)
        scores.append(s)
    boxes, scores = np.vstack(boxes), np.concatenate(scores)
    if len(sizes) + bool(tile) > 1 and len(boxes):
        boxes, scores = merge(boxes, scores)
    if not len(boxes):
        return boxes, scores, np.ones(0, bool)
    # boxes under min_size px are specks/edge artifacts at webcam distance (a 10 px lesion is ~2 mm at 25 cm)
    kept = (boxes[:, 2:] - boxes[:, :2]).min(1) >= min_size
    if green > 0:
        mask = green_mask(frame)
        kept &= np.array([ring_green(mask, b) >= green for b in boxes])
    return boxes, scores, kept


class Tracker:
    """Keeps lesion boxes steady between frames and on the moving stem.

    A detection >= conf starts a track; detections down to keep_conf keep an existing track alive, so a lesion whose
    score wobbles around the threshold does not blink. Tracks are matched to detections by centre distance (relative
    to box size), which still works when the stem moves faster than a box width per frame. Each track keeps a
    velocity: it moves the box forward between detections and on frames where the model misses the lesion, so the
    box does not trail behind. A track is drawn after min_hits matches and dropped `hold` frames after its last match.
    """

    def __init__(self, min_hits: int = 2, hold: int = 8, max_dist: float = 1.5, score_a: float = 0.5):
        self.min_hits, self.hold, self.max_dist, self.score_a = min_hits, hold, max_dist, score_a
        self.tracks: list[dict] = []

    def update(self, boxes: np.ndarray, scores: np.ndarray, conf: float) -> list[dict]:
        for t in self.tracks:  # predict: shift every track by its velocity (position only, the size stays)
            t["box"] = t["box"] + np.tile(t["vel"], 2)
        unmatched, matched = set(range(len(boxes))), set()
        if self.tracks and len(boxes):
            tb = np.array([t["box"] for t in self.tracks])
            centre = lambda b: (b[:, :2] + b[:, 2:]) / 2
            size = np.maximum(tb[:, 2:] - tb[:, :2], 1).max(1)
            dist = np.linalg.norm(centre(tb)[:, None] - centre(boxes)[None], axis=2) / size[:, None]
            for ti, di in sorted(np.argwhere(dist <= self.max_dist).tolist(), key=lambda p: dist[p[0], p[1]]):
                if di not in unmatched or ti in matched:
                    continue
                t = self.tracks[ti]
                err = (boxes[di, :2] + boxes[di, 2:] - t["box"][:2] - t["box"][2:]) / 2  # centre error of the prediction
                t["vel"] = t["vel"] + 0.5 * err
                t["box"] = boxes[di].copy()  # snap to the newest detection: no smoothing lag
                t["score"] = self.score_a * scores[di] + (1 - self.score_a) * t["score"]
                t["hits"], t["misses"] = t["hits"] + 1, 0
                unmatched.discard(di)
                matched.add(ti)
        for ti, t in enumerate(self.tracks):
            if ti not in matched:
                t["misses"] += 1
                t["vel"] = 0.7 * t["vel"]  # coast to a stop when the lesion is no longer seen
        self.tracks = [t for t in self.tracks if t["misses"] <= self.hold]
        self.tracks += [dict(box=boxes[d].copy(), vel=np.zeros(2), score=float(scores[d]), hits=1, misses=0)
                        for d in unmatched if scores[d] >= conf]
        return [t for t in self.tracks if t["hits"] >= self.min_hits]


def draw(img: np.ndarray, tracks: list[dict], ui: float = 1.0):
    """ui scales lines and text: 1 on a 1280 px frame, 3 on 4K, so labels stay readable when the window is shrunk."""
    px = lambda v: max(int(round(v * ui)), 1)
    for t in tracks:
        x1, y1, x2, y2 = t["box"].astype(int)
        color = (0, 0, 255) if t["score"] >= 0.5 else (0, 140, 255)  # red = confident, orange = unsure
        cv2.rectangle(img, (x1, y1), (x2, y2), color, px(3) if t["misses"] == 0 else px(1))
        label = f"canker {t['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * ui, px(2))
        ty = y1 if y1 > th + px(8) else y2 + th + px(8)  # label below the box when it touches the top edge
        cv2.rectangle(img, (x1, ty - th - px(8)), (x1 + tw + px(6), ty), color, -1)
        cv2.putText(img, label, (x1 + px(3), ty - px(5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * ui, (255, 255, 255), px(2))


def draw_raw(img: np.ndarray, boxes: np.ndarray, scores: np.ndarray, kept: np.ndarray):
    for (x1, y1, x2, y2), s, k in zip(boxes.astype(int), scores, kept):
        cv2.rectangle(img, (x1, y1), (x2, y2), (180, 180, 180), 1)
        cv2.putText(img, f"{s:.2f}", (x1, max(y1 - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        if not k:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.line(img, (x1, y2), (x2, y1), (0, 0, 255), 1)


def draw_hud(img: np.ndarray, lines: list[str], y: int | None = None, ui: float = 1.0):
    y = int(28 * ui) if y is None else y
    for text in lines:
        cv2.putText(img, text, (int(10 * ui), y), cv2.FONT_HERSHEY_SIMPLEX, 0.65 * ui, (0, 0, 0), int(4 * ui))
        cv2.putText(img, text, (int(10 * ui), y), cv2.FONT_HERSHEY_SIMPLEX, 0.65 * ui, (0, 255, 255), int(2 * ui))
        y += int(26 * ui)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="weights/dragon.pt")
    ap.add_argument("--camera", "--source", dest="camera", default=0, type=lambda s: int(s) if s.isdigit() else s,
                    help="0 = built-in webcam; or a photo, a video file or an rtsp:// stream URL")
    ap.add_argument("--conf", type=float, default=0.25, help="score needed to start a box (adjust live with +/-)")
    ap.add_argument("--green", type=float, default=0.30, help="plant check: min green share around a box")
    ap.add_argument("--min-size", type=float, default=10, help="ignore boxes whose shorter side is below this many pixels")
    ap.add_argument("--sizes", type=int, nargs="+", default=[800, 1280], help="input sizes; first one alone when 'm' is off")
    ap.add_argument("--tiles", action="store_true",
                    help="start with the tile pass on (it also starts on by itself when the frame is bigger than the largest --sizes)")
    ap.add_argument("--tile", type=int, default=800, help="tile size in frame px, scaled to the first --sizes (adjust live with [ / ])")
    ap.add_argument("--overlap", type=float, default=0.25, help="min overlap between neighbouring tiles, as a share of a tile")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    model = YOLO(project_path(args.weights))
    device, quantize = (0, 16) if torch.cuda.is_available() else ("cpu", None)  # FP16 on the GPU
    conf, green_on, multi, debug, tile = args.conf, True, device == 0, False, args.tile

    cam = open_source(args.camera, args.width, args.height)
    ok, frame = cam.read()
    if not ok or frame is None:
        raise SystemExit(f"{args.camera} opened but returned no frame")
    H, W = frame.shape[:2]
    tiled = args.tiles or max(W, H) > max(args.sizes)  # a whole-frame pass would shrink every lesion in this frame
    for sz in args.sizes:  # warm-up: CUDA init + one graph per input size
        model.predict(frame, imgsz=sz, device=device, quantize=quantize, verbose=False)
    detect_tiles(model, frame, args.sizes[0], tile, args.overlap, 0.25, device, quantize)
    print(f"source {args.camera}: {W}x{H}  sizes={args.sizes}  tiles={'ON' if tiled else 'OFF'} "
          f"({len(tile_grid(W, H, tile, args.overlap))} x {tile} px)  device={device}")

    tracker = Tracker()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    fit = min(1.0, 1600 / W, 900 / H)  # a 4K feed or a 20 MP drone photo would not fit on the screen
    cv2.resizeWindow(WINDOW, int(W * fit), int(H * fit))
    ui = 1 / fit  # text as big as on the 1280 px webcam once the window is shrunk
    frozen, annotated, fps, n_neg, last_seen = False, frame.copy(), 0.0, 0, time.time()
    while True:
        if not frozen:
            ok, frame = cam.read()
            if not ok or frame is None:
                print("camera stopped returning frames")
                break
            t0 = time.perf_counter()
            sizes = args.sizes if multi else args.sizes[:1]
            boxes, scores, kept = detect(model, frame, sizes, max(conf - 0.10, 0.05), device, quantize,
                                         args.green if green_on else 0.0, args.min_size, tile if tiled else 0, args.overlap)
            shown = tracker.update(boxes[kept], scores[kept], conf)
            dt = max(time.perf_counter() - t0, 1e-6)
            fps = 0.9 * fps + 0.1 / dt if fps else 1 / dt
            if shown:
                last_seen = time.time()

            annotated = frame.copy()
            grid = tile_grid(frame.shape[1], frame.shape[0], tile, args.overlap) if tiled else []
            if debug:
                for x1, y1, x2, y2 in grid:
                    cv2.rectangle(annotated, (x1, y1), (x2 - 1, y2 - 1), (255, 120, 0), int(ui))
                draw_raw(annotated, boxes, scores, kept)
            draw(annotated, shown, ui)
            draw_hud(annotated, [
                f"lesions: {len(shown)}" + (f"  (best {max(t['score'] for t in shown):.2f})" if shown else ""),
                f"conf >= {conf:.2f} [+/-]  plant check {'ON' if green_on else 'OFF'} [g]  sizes {'+'.join(map(str, sizes))} [m]",
                f"tiles {f'{len(grid)} x {tile} px = {args.sizes[0] / tile:.2f}x zoom' if tiled else 'OFF'} [t]  tile size [ / ]",
                f"{fps:.0f} fps  debug {'ON' if debug else 'OFF'} [d]  negatives saved {n_neg} [n]",
            ], ui=ui)
            if time.time() - last_seen > 2:
                draw_hud(annotated, ["No canker found: hold the stem 15-30 cm away, fill the frame, good light"],
                         y=annotated.shape[0] - int(16 * ui), ui=ui)
        cv2.imshow(WINDOW, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in (ord("+"), ord("=")):
            conf = min(conf + 0.05, 0.95)
        elif key in (ord("-"), ord("_")):
            conf = max(conf - 0.05, 0.05)
        elif key == ord("g"):
            green_on = not green_on
        elif key == ord("m"):
            multi = not multi
        elif key == ord("t"):
            tiled = not tiled
        elif key in (ord("["), ord("]")):  # smaller tile = more zoom; steps of 1.25x, multiples of 32 px
            tile = int(np.clip(round(tile * (1.25 if key == ord("]") else 0.8) / 32) * 32, 160, 1600))
        elif key == ord("d"):
            debug = not debug
        elif key == ord(" "):
            frozen = not frozen
            if frozen:
                draw_hud(annotated, ["FROZEN - space to resume"], y=annotated.shape[0] - int(16 * ui), ui=ui)
        elif key == ord("s"):
            out = project_path("runs/webcam")
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"webcam_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            cv2.imwrite(str(path), annotated)
            print("saved", path)
        elif key == ord("n"):
            NEGATIVES.mkdir(parents=True, exist_ok=True)
            path = NEGATIVES / f"webcam_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
            cv2.imwrite(str(path), frame)  # raw frame, no boxes drawn
            n_neg += 1
            print("negative saved", path)

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
