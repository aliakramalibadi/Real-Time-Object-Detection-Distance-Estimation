import cv2
import numpy as np

# Real-world widths in cm
OBJECT_WIDTHS_CM = {
    "person": 48.7,
    "car": 194.0,
    "truck": 250.0,
    "bus": 250.0,
    "motorcycle": 80.0
}

FOCAL_LENGTH = 250
MIN_BOX_WIDTH = 15
MAX_DISTANCE_METERS = 20

def estimate_distance(width_px, label):
    real_width_cm = OBJECT_WIDTHS_CM.get(label.lower())
    if real_width_cm and width_px >= MIN_BOX_WIDTH:
        distance_m = (real_width_cm * FOCAL_LENGTH) / width_px / 100
        return distance_m if distance_m <= MAX_DISTANCE_METERS else None
    return None

# Load class labels
with open("data/coco.names", "r") as f:
    class_names = f.read().strip().split("\n")

# Load detection model
net = cv2.dnn_DetectionModel("model/frozen_inference_graph.pb", "model/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Load video
cap = cv2.VideoCapture("test_video.mp4")

cv2.namedWindow("Video Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video Detection", 800, 600)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]
    zone_left = int(frame_width * 0.3)
    zone_right = int(frame_width * 0.7)
    zone_top = int(frame_height * 0.5)
    zone_bottom = frame_height
    #cv2.rectangle(frame, (zone_left, zone_top), (zone_right, zone_bottom), (0, 255, 255), 1)

    show_stop = False  # Flag to trigger STOP overlay

    outputs = net.detect(frame, confThreshold=0.6)
    if outputs is not None and len(outputs) == 3:
        class_ids, confidences, boxes = outputs
    else:
        class_ids, confidences, boxes = [], [], []

    for class_id, confidence, box in zip(np.array(class_ids).flatten(), np.array(confidences).flatten(), boxes):
        label = class_names[class_id - 1] if class_id <= len(class_names) else str(class_id)
        if label.lower() not in OBJECT_WIDTHS_CM:
            continue

        x, y, w, h = box
        if w < MIN_BOX_WIDTH:
            continue

        cx, cy = x + w // 2, y + h // 2
        is_in_front = zone_left < cx < zone_right and zone_top < cy < zone_bottom

        distance_m = estimate_distance(w, label)
        if distance_m is None:
            continue

        is_close = distance_m < 2.0
        if distance_m <= 1.0 and is_in_front:
            show_stop = True  # Trigger mid-screen STOP

        distance_label = f"{distance_m:.2f} m"
        box_color = (0, 0, 255) if is_in_front and is_close else (255, 0, 0)

        cv2.rectangle(frame, box, box_color, 1)
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, distance_label, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)

        if is_in_front and is_close:
            cv2.putText(frame, "WARNING!", (x, y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    if show_stop:
        cv2.putText(
            frame, "STOP!", (frame_width // 2 - 100, frame_height // 2),
            cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 255), 4
        )

    cv2.imshow("Video Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
