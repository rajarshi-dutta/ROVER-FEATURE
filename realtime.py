import cv2
import threading
from ultralytics import YOLO

model = YOLO('blacknwhite'
'.pt')
stream_url = "http://192.168.29.25:81/stream"

class StreamReader:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        self.ret, self.frame = False, None
        self.stopped = False
        self.lock = threading.Lock()
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()


class Detector:
    def __init__(self, reader):
        self.reader = reader
        self.annotated = None
        self.stopped = False
        self.lock = threading.Lock()
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.reader.read()
            if not ret or frame is None:
                continue
            results = model.predict(source=frame, imgsz=640, device='cpu', verbose=False)
            annotated = results[0].plot()
            with self.lock:
                self.annotated = annotated

    def read(self):
        with self.lock:
            return self.annotated

    def stop(self):
        self.stopped = True


reader = StreamReader(stream_url)
detector = Detector(reader)

while True:
    frame = detector.read()
    if frame is None:
        continue

    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

reader.stop()
detector.stop()
cv2.destroyAllWindows()