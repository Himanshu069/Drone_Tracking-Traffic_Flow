"""
Traffic Flow Counter with Interactive Polygon Selection
Vehicle classes tracked: car, motorcycle, bus, truck

Usage:
    on a video file
    python traffic_counter.py --source traffic.mp4

    with a larger/more accurate model
    python traffic_counter.py --source traffic.mp4 --model yolov8m.pt

    save output video too
    python traffic_counter.py --source traffic.mp4 --output result.mp4

    webcam
    python traffic_counter.py --source 0

Controls (Polygon Selection window):
    Left click  — add polygon point
    Right click — remove last point
    Enter       — confirm polygon and start counting
    R           — reset polygon
    Q/ESC       — quit

Controls (Counting window):
    Q/ESC       — quit
    R           — redefine polygon (pauses video)
    S           — save screenshot
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics not installed. Run:  pip install ultralytics")
    sys.exit(1)

VEHICLE_CLASSES = {
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
}

# Distinct colours per class
CLASS_COLORS = {
    "car":        (0,   200, 255),   # orange
    "motorcycle": (255, 100,   0),   # blue
    "bus":        (0,   255, 100),   # green
    "truck":      (0,    80, 255),   # red
}

POLYGON_COLOR   = (0, 255, 255)   # yellow
POLYGON_FILL    = (0, 255, 255)
FONT            = cv2.FONT_HERSHEY_SIMPLEX



class PolygonSelector:
    """
    Show the first frame and let the user draw a counting polygon.
    Returns a numpy array of shape (N, 2) with the confirmed points,
    or None if the user quits without confirming.
    """

    def __init__(self, frame: np.ndarray):
        self.frame   = frame.copy()
        self.points  = []
        self.done    = False
        self.aborted = False
        self.win     = "Define Counting Zone  |  LClick=add  RClick=undo  Enter=confirm  R=reset  Q=quit"

    def _mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                self.points.pop()

    def _draw(self):
        img = self.frame.copy()
        pts = self.points

        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                cv2.line(img, pts[i], pts[i + 1], POLYGON_COLOR, 2)
        if len(pts) >= 3:
            cv2.line(img, pts[-1], pts[0], POLYGON_COLOR, 2)   # close
            # semi-transparent fill
            overlay = img.copy()
            cv2.fillPoly(overlay, [np.array(pts, np.int32)], POLYGON_FILL)
            cv2.addWeighted(overlay, 0.20, img, 0.80, 0, img)

        for p in pts:
            cv2.circle(img, p, 5, POLYGON_COLOR, -1)

        # guide text
        lines = [
            f"Points: {len(pts)}",
            "LClick: add point",
            "RClick: undo last",
            "Enter : confirm (need >= 3 pts)",
            "R     : reset",
            "Q/ESC : quit",
        ]
        for i, ln in enumerate(lines):
            cv2.putText(img, ln, (10, 25 + i * 22), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        return img

    def run(self):
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, 1280, 720)
        cv2.setMouseCallback(self.win, self._mouse)

        while True:
            cv2.imshow(self.win, self._draw())
            key = cv2.waitKey(20) & 0xFF

            if key in (13, 10):          # Enter — confirm
                if len(self.points) >= 3:
                    self.done = True
                    break
                else:
                    print("[INFO] Need at least 3 points to form a polygon.")

            elif key in (ord('r'), ord('R')):
                self.points = []

            elif key in (ord('q'), ord('Q'), 27):  # Q / ESC
                self.aborted = True
                break

        cv2.destroyWindow(self.win)

        if self.done:
            return np.array(self.points, np.int32)
        return None


def draw_polygon(frame: np.ndarray, pts: np.ndarray, alpha: float = 0.15):
    overlay = frame.copy()
    cv2.polylines(frame, [pts], True, POLYGON_COLOR, 2, cv2.LINE_AA)
    cv2.fillPoly(overlay, [pts], POLYGON_FILL)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_counts(frame: np.ndarray, counts: dict, fps: float):
    h, w = frame.shape[:2]
    panel_w, panel_h = 220, 40 + len(counts) * 32 + 36
    x0, y0 = w - panel_w - 10, 10

    # background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (80, 80, 80), 1)

    cv2.putText(frame, "Traffic Count", (x0 + 10, y0 + 24),
                FONT, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    total = sum(counts.values())
    for i, (cls, cnt) in enumerate(counts.items()):
        ty = y0 + 50 + i * 32
        color = CLASS_COLORS[cls]
        cv2.rectangle(frame, (x0 + 10, ty - 14), (x0 + 20, ty + 2), color, -1)
        cv2.putText(frame, f"{cls:<12s} {cnt:>4d}", (x0 + 28, ty),
                    FONT, 0.56, (220, 220, 220), 1, cv2.LINE_AA)

    ty = y0 + panel_h - 12
    cv2.putText(frame, f"Total: {total}   FPS: {fps:.1f}", (x0 + 10, ty),
                FONT, 0.50, (180, 180, 180), 1, cv2.LINE_AA)


def draw_box(frame, x1, y1, x2, y2, cls_name, track_id, conf, in_zone):
    color = CLASS_COLORS.get(cls_name, (200, 200, 200))
    thick = 2 if in_zone else 1
    alpha_box = color if in_zone else tuple(c // 2 for c in color)
    cv2.rectangle(frame, (x1, y1), (x2, y2), alpha_box, thick)
    label = f"{cls_name} #{track_id} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.45, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), alpha_box, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4), FONT, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

def reset_state():
    counts        = {cls: 0 for cls in VEHICLE_CLASSES.values()}
    seen_ids      = set()          # track IDs already counted (never count twice)
    tid_class     = {}             # tid -> class locked on first detection
    return counts, seen_ids, tid_class
 

def run_counter(source, model_path: str, output_path: str | None, conf_thresh: float):

    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(model_path)
    print(f"[INFO] Model loaded. Vehicle classes in use: {list(VEHICLE_CLASSES.values())}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)

    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[INFO] Source: {W}x{H} @ {native_fps:.1f} fps")

    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] Could not read first frame.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # rewind

    polygon = None
    while polygon is None:
        sel = PolygonSelector(first_frame)
        polygon = sel.run()
        if sel.aborted:
            print("[INFO] Aborted by user.")
            cap.release()
            sys.exit(0)
        if polygon is None:
            print("[WARN] No valid polygon. Try again.")

    print(f"[INFO] Polygon confirmed with {len(polygon)} points.")

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, native_fps, (W, H))
        print(f"[INFO] Writing output to: {output_path}")

    counts, seen_ids, tid_class = reset_state()
    fps        = 0.0
    t_prev     = time.time()

    WIN = "Traffic Counter  |  Q=quit  R=redefine polygon  S=screenshot"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, min(W, 1280), min(H, 720))

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video.")
            break

        frame_idx += 1

        results = model.track(
            frame,
            tracker="bytetrack.yaml",
            persist=True,
            conf=conf_thresh,
            classes=list(VEHICLE_CLASSES.keys()),
            verbose=False,
        )

        draw_polygon(frame, polygon)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes   = results[0].boxes.xyxy.cpu().numpy().astype(int)
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            confs   = results[0].boxes.conf.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), cls_id, conf, tid in zip(boxes, cls_ids, confs, track_ids):
                if cls_id not in VEHICLE_CLASSES:
                    continue

                tid = int(tid)
                detected_cls = VEHICLE_CLASSES[cls_id]

                if tid not in tid_class:
                    tid_class[tid] = {}
                votes = tid_class[tid]
                votes[detected_cls] = votes.get(detected_cls, 0) + 1

                total_votes = sum(votes.values())
                if total_votes < 15:
                    cls_name = detected_cls   # show raw label while voting
                    locked   = False
                else:
                    cls_name = max(votes, key=votes.get)   # majority winner
                    locked   = True
                cx, cy   = (x1 + x2) // 2, (y1 + y2) // 2

                # point-in-polygon test
                inside = cv2.pointPolygonTest(polygon, (float(cx), float(cy)), False) >= 0

                # count once per unique track_id per class
                if locked and inside and tid not in seen_ids:
                    counts[cls_name] += 1
                    seen_ids.add(tid)

                draw_box(frame, x1, y1, x2, y2, cls_name, tid, conf, inside)

                # centroid dot
                dot_color = CLASS_COLORS.get(cls_name, (255, 255, 255))
                cv2.circle(frame, (cx, cy), 4, dot_color, -1)

        t_now = time.time()
        fps   = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now

        draw_counts(frame, counts, fps)
        cv2.putText(frame, f"Frame {frame_idx}", (10, H - 10),
                    FONT, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

        if writer:
            writer.write(frame)

        cv2.imshow(WIN, frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q'), 27):
            print("[INFO] Quit by user.")
            break

        elif key in (ord('r'), ord('R')):
            print("[INFO] Redefining polygon…")
            snapshot = frame.copy()
            sel2 = PolygonSelector(snapshot)
            new_poly = sel2.run()
            if new_poly is not None and len(new_poly) >= 3:
                polygon = new_poly
                counts, seen_ids, tid_class = reset_state()
                print("[INFO] Polygon updated. Counts reset.")
            else:
                print("[INFO] Polygon unchanged.")

        elif key in (ord('s'), ord('S')):
            fname = f"screenshot_{frame_idx}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[INFO] Screenshot saved: {fname}")

    print("\n" + "=" * 40)
    print("  FINAL VEHICLE COUNT")
    print("=" * 40)
    for cls, cnt in counts.items():
        print(f"  {cls:<14s}: {cnt}")
    print(f"  {'TOTAL':<14s}: {sum(counts.values())}")
    print("=" * 40)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()




def parse_args():
    p = argparse.ArgumentParser(description="Traffic flow counter with interactive polygon zone")
    p.add_argument("--source",  default="0",
                   help="Video file path or webcam index (default: 0)")
    p.add_argument("--model",   default="yolov8n.pt",
                   help="YOLOv8 model weights (default: yolov8n.pt). "
                        "Larger = more accurate: yolov8s.pt / yolov8m.pt / yolov8l.pt")
    p.add_argument("--output",  default=None,
                   help="Optional output video path, e.g. out.mp4")
    p.add_argument("--conf",    type=float, default=0.35,
                   help="Detection confidence threshold (default: 0.35)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    run_counter(
        source      = source,
        model_path  = args.model,
        output_path = args.output,
        conf_thresh = args.conf,
    )