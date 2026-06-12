from ultralytics import YOLO
import glob

model = YOLO("yolov8n.pt")

results = model.train(
    data="Drone Detection.v3i.yolov8/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device="cpu",
    scale=0.5,
    mosaic=1.0,
    hsv_v=0.4,
    fliplr=0.5,
)

model = YOLO("runs/detect/train/weights/best.pt")

metrics = model.val(data="Drone Detection.v3i.yolov8/data.yaml")

print("\n── Val Results ──")
print(f"mAP50:     {metrics.box.map50:.3f}")
print(f"mAP50-95:  {metrics.box.map:.3f}")
print(f"Precision: {metrics.box.p.mean():.3f}")
print(f"Recall:    {metrics.box.r.mean():.3f}")

metrics_test = model.val(data="Drone Detection.v3i.yolov8/data.yaml", split="test")

print("\n── Test Results ──")
print(f"mAP50:     {metrics_test.box.map50:.3f}")
print(f"mAP50-95:  {metrics_test.box.map:.3f}")
print(f"Precision: {metrics_test.box.p.mean():.3f}")
print(f"Recall:    {metrics_test.box.r.mean():.3f}")


test_images = glob.glob("Drone Detection.v3i.yolov8/data.yaml")[:10]

model.predict(
    source=test_images,
    imgsz=640,
    conf=0.4,
    save=True,
    save_conf=True,
)

print("\nDone. Check runs/detect/predict/ for annotated images.")