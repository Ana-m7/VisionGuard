import cv2
import requests
import threading

CAM2_VIDEO = "../videos/crowd.mp4"
CAM4_VIDEO = "../videos/normal_walking.mp4"
CAM2_URL   = "http://localhost:8000/push_frame/2"
CAM4_URL   = "http://localhost:8000/push_frame/4"

# --- NON-BLOCKING PUSH for each cam ---
_frames = {2: [None], 4: [None]}
_locks  = {2: threading.Lock(), 4: threading.Lock()}

def _worker(cam_id, url):
    while True:
        with _locks[cam_id]:
            frame = _frames[cam_id][0]
        if frame is not None:
            try:
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                requests.post(url, files={'frame': ('f.jpg', buf.tobytes(), 'image/jpeg')}, timeout=1)
            except:
                pass
        threading.Event().wait(0.04)

threading.Thread(target=_worker, args=(2, CAM2_URL), daemon=True).start()
threading.Thread(target=_worker, args=(4, CAM4_URL), daemon=True).start()

def push(cam_id, frame):
    with _locks[cam_id]:
        _frames[cam_id][0] = frame.copy()

cap2 = cv2.VideoCapture(CAM2_VIDEO)
cap4 = cv2.VideoCapture(CAM4_VIDEO)

print("CAM-02 and CAM-04 streaming...")

while True:
    ret2, f2 = cap2.read()
    if not ret2:
        cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret2, f2 = cap2.read()

    ret4, f4 = cap4.read()
    if not ret4:
        cap4.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret4, f4 = cap4.read()

    if ret2 and f2 is not None:
        push(2, cv2.resize(f2, (1280, 720)))
    if ret4 and f4 is not None:
        push(4, cv2.resize(f4, (1280, 720)))

cap2.release()
cap4.release()