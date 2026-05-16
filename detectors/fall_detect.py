import cv2
from ultralytics import YOLO

# --- CONFIG ---
VIDEO_PATH = "../videos/fall.mp4"   # swap to a video with someone falling
ASPECT_RATIO_THRESHOLD = 1.3      # width/height > this = horizontal = fallen

model = YOLO("yolov8n-pose.pt")   # auto-downloads
cap = cv2.VideoCapture(VIDEO_PATH)

print("Starting fall detection... Press Q to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, classes=[0], verbose=False)[0]
    fall_detected = False

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        width = x2 - x1
        height = y2 - y1

        if height == 0:
            continue

        ratio = width / height

        if ratio > ASPECT_RATIO_THRESHOLD:
            fall_detected = True
            # draw red box around fallen person
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
            cv2.putText(frame, f"⚠ FALL DETECTED (ratio:{ratio:.2f})",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # normal standing person — green box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

    annotated = cv2.resize(frame, (1280, 720))
    status = "⚠ FALL DETECTED" if fall_detected else "OK — No Fall"
    color = (0, 0, 255) if fall_detected else (0, 255, 0)
    cv2.putText(annotated, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

    cv2.imshow("Fall Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")