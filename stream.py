from flask import Flask, Response
import cv2
import threading
import ngrok

app          = Flask(__name__)
lock         = threading.Lock()
output_frame = None

def set_frame(frame):
    global output_frame
    with lock:
        output_frame = frame.copy()

def generate():
    while True:
        with lock:
            if output_frame is None:
                continue
            ret, buf = cv2.imencode('.jpg', output_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                continue
            frame_bytes = buf.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video')
def video():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '<html><body><img src="/video" width="100%"></body></html>'

def start(host='0.0.0.0', port=5000):
    listener = ngrok.forward(5000, authtoken="3EzFLmqVEe4pnuL7O5fS2dlRAqi_4RqCPNZ9FHTUaQsEu4MZP")
    print(f"Public stream URL: {listener.url()}/video", flush=True)
    threading.Thread(target=lambda: app.run(host=host, port=port,
                     threaded=True, use_reloader=False),
                     daemon=True).start()