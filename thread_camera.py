import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
import os
from serial_command import SerialCommander, compute_tilt_angle
import torch
# from pyngrok import ngrok
# from stream import set_frame, start as start_stream

torch.set_num_threads(4)
torch.set_num_interop_threads(2)
model = YOLO("best(1).pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[Main Thread] CRITICAL ERROR: Could not open camera backend!", flush=True)
    exit(1)
else:
    print("[Main Thread] Camera hardware opened successfully.", flush=True)

# KLT parameters
lk_params = dict(
    winSize=(41, 41),
    maxLevel=4,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)
feature_params = dict(
    maxCorners=150, 
    qualityLevel=0.15, 
    minDistance=4, 
    blockSize=7
    )

FRAME_H         = 480
VFOV_DEG        = 60.0
CAMERA_TILT_DEG = 0.0

commander = SerialCommander(port='/dev/ttyACM0', baud=9600)
# start_stream()

MIN_POINTS   = 6          # Re-detect if tracked points fall below this
BASE_CONF   = 0.4
MISS_CONF   = 0.25

prev_gray    = None
prev_pts     = None       # (K,1,2) float32  — KLT feature points
tracked_box  = None   
miss_streak = 0              # (x1,y1,x2,y2)   — last known bounding box
prev    = time.time()

latest_frame    = None
latest_yolo_box = None
frame_lock      = threading.Lock()
yolo_lock       = threading.Lock()
stop_event      = threading.Event()

def yolo_worker():
    if hasattr(os, 'sched_setaffinity'):
        try:
            os.sched_setaffinity(0, {2, 3})
            print("[YOLO Thread] Successfully pinned to Core 3 (Linux).")
        except Exception as e:
            print(f"[YOLO Thread] Core pinning failed: {e}")
    else:
        print("[YOLO Thread] Core pinning not supported on this OS. Skipping.")        # dedicate core 3 to YOLO
    global latest_yolo_box, miss_streak
    try:
        while not stop_event.is_set():
            with frame_lock:
                if latest_frame is None:
                    time.sleep(0.01)
                    continue
                frame = latest_frame.copy()
            
            with yolo_lock:
                miss = miss_streak
                current = latest_yolo_box
            
            conf    = MISS_CONF if miss > 2 else BASE_CONF
            results = model.predict(frame, imgsz=320, conf=conf, verbose=False, device="cpu")
            boxes   = results[0].boxes
            box     = None
            
            if boxes is not None and len(boxes):
                best      = int(boxes.conf.argmax())
                best_conf = float(boxes.conf[best])
                print(f"[YOLO Thread] Found class {int(boxes.cls[best])} with conf {best_conf:.2f}")
                
                box       = tuple(boxes.xyxy[best].cpu().numpy().astype(int))
                with yolo_lock:
                    current = latest_yolo_box
                if best_conf < BASE_CONF and current is not None:
                    cx_new   = (box[0]+box[2])/2
                    cy_new   = (box[1]+box[3])/2
                    cx_old   = (current[0]+current[2])/2
                    cy_old   = (current[1]+current[3])/2
                    dist     = np.sqrt((cx_new-cx_old)**2 + (cy_new-cy_old)**2)
                    box_diag = np.sqrt((current[2]-current[0])**2 +
                                    (current[3]-current[1])**2)
                    if dist > 5 * box_diag:
                        box = None
            with yolo_lock:
                latest_yolo_box = box
    except Exception as e:
        # This will prevent the thread from dying silently!
        print(f"\n[YOLO Thread] CRITICAL CRASH: {e}\n")                

yolo_thread = threading.Thread(target=yolo_worker, daemon=True)
yolo_thread.start()

os.sched_setaffinity(0, {0, 1})

def sample_points_in_box(gray, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(gray.shape[1]-1, x2), min(gray.shape[0]-1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    mask = np.zeros_like(gray)
    mask[y1:y2, x1:x2] = 255
    return cv2.goodFeaturesToTrack(gray, mask=mask, **feature_params)

def shift_box(box, dx, dy):
    x1, y1, x2, y2 = box
    return int(x1+dx), int(y1+dy), int(x2+dx), int(y2+dy)
ret, frame = cap.read()
results = model.predict(frame, imgsz=640, conf=0.1, verbose=True, device="cpu")
print("Classes in model:", model.names)
print("Detections:", results[0].boxes)
while True:
    ret, frame = cap.read()
    if not ret:
        print("[Main Thread] CRITICAL ERROR: Failed to grab frame from camera! Exiting loop...", flush=True)
        break

    with frame_lock:
        latest_frame = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    with yolo_lock:
        yolo_box = latest_yolo_box

    klt_box = None
    if yolo_box is None and prev_gray is not None and prev_pts is not None and len(prev_pts) >= MIN_POINTS:
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, prev_pts, None, **lk_params
        )
        if next_pts is not None:
            back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, prev_gray, next_pts, None, **lk_params
            )
            # Keep only points where forward AND backward tracking succeeded
            fb_error = np.linalg.norm(
            prev_pts.reshape(-1, 2) - back_pts.reshape(-1, 2), axis=1
            )
            good_mask = (status.flatten() == 1) & \
                    (back_status.flatten() == 1) & \
                    (fb_error < 2.0)          # pixel threshold

            prev_good = prev_pts.reshape(-1, 2)[good_mask]
            curr_good = next_pts.reshape(-1, 2)[good_mask]

            if tracked_box is not None and len(curr_good) > 0:
                x1, y1, x2, y2 = tracked_box
                cx, cy = (x1+x2)/2, (y1+y2)/2
                hw = (x2-x1)/2 * 1.8
                hh = (y2-y1)/2 * 1.8
                in_box    = np.array([
                    abs(pt[0]-cx) <= hw and abs(pt[1]-cy) <= hh
                    for pt in curr_good
                ])
                prev_good = prev_good[in_box]
                curr_good = curr_good[in_box]

            if len(curr_good) >= MIN_POINTS and tracked_box is not None:
                dx, dy  = np.median(curr_good - prev_good, axis=0)
                klt_box = shift_box(tracked_box, dx, dy)
                prev_pts = curr_good.reshape(-1, 1, 2)
    with yolo_lock:
        if yolo_box is not None:
            tracked_box = tuple(yolo_box)           # hard anchor to YOLO
            prev_pts    = sample_points_in_box(gray, tracked_box)
            miss_streak = 0
        elif klt_box is not None:
            tracked_box = klt_box      
            miss_streak += 1
            if prev_pts is not None and len(prev_pts) < MIN_POINTS * 2:
                new_pts = sample_points_in_box(gray, tracked_box)
                if new_pts is not None:
                    prev_pts = new_pts
        else:
            miss_streak += 1
            prev_pts = None
    display = frame.copy()
    if tracked_box is not None:
        tilt = compute_tilt_angle(tracked_box, FRAME_H, VFOV_DEG, CAMERA_TILT_DEG)
        commander.send_angle(tilt)
        # commander.read_response() # can cause delay 
        x1, y1, x2, y2 = tracked_box
        color = (0, 165, 255) if yolo_box is not None else (0, 255, 0)
        label = "Drone_y" if yolo_box is not None else "Drone_klt"
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display, label, (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    if prev_pts is not None:
        for pt in prev_pts.reshape(-1, 2):
            cv2.circle(display, (int(pt[0]), int(pt[1])), 3, (255, 0, 255), -1)


    # FPS counter
    now = time.time()
    fps = 1.0 / max(now - prev, 1e-6)
    prev = now
    cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # set_frame(display)
    cv2.imshow("Drone Tracker", display)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    prev_gray = gray

stop_event.set()
commander.close()
cap.release()
cv2.destroyAllWindows()