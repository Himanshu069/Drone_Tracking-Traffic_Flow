from ultralytics import YOLO
test_images = [
        "Drone Detection.v3i.yolov8/test/images/39_JPEG_jpg.rf.76ce40ed1f2b066f070c31e429c519b2.jpg",
        "Drone Detection.v3i.yolov8/test/images/yoto03046_jpg.rf.cdca91de95e3e4d0e7d2608944ea26ab.jpg",
        "Drone Detection.v3i.yolov8/test/images/yoto10934_jpg.rf.754198fc9c3b19b7eb97895a5dcb85dd.jpg",
]
model = YOLO("best.pt")

model.predict(
    source=test_images,
    imgsz=640,
    conf=0.4,
    save=True,
    save_conf=True,
)

print("\nDone. Check runs/detect/predict/ for annotated images.")
# Download the best weights
