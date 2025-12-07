import cv2
import numpy as np

from picamera2 import Picamera2
from libcamera import Transform

import dbus
import dbus.service
import dbus.mainloop.glib
from gi.repository import GLib

# Real-world object widths in cm
OBJECT_WIDTHS_CM = {
    "person": 48.7,
    "car": 194.0,
    "truck": 250.0,
    "bus": 250.0,
    "motorcycle": 80.0
}

# Camera and detection settings
FOCAL_LENGTH = 350 			 # in pixels
MIN_BOX_WIDTH = 30  		 # filter out tiny boxes
MAX_DISTANCE_METERS = 15.0   # max distance to report
SAFETY_DISTANCE = 2.0        # distance to emit WARNING
STOP_DISTANCE = 1.0          # distance to emit STOP

# DBus setup
dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
bus = dbus.SessionBus()

# DBus object
class ObstacleDetector(dbus.service.Object):
    def __init__(self):
        name = dbus.service.BusName('com.car.ObstacleDetector', bus)
        super().__init__(name, '/com/car/ObstacleDetector')
        self.gui_enabled = True  # flag to control the window on GUI

    @dbus.service.signal(dbus_interface='com.car.ObstacleDetector.Interface', signature='s')
    def WarningSignal(self, message): pass

    @dbus.service.signal(dbus_interface='com.car.ObstacleDetector.Interface', signature='s')
    def StopSignal(self, message): pass

    @dbus.service.method(dbus_interface='com.car.ObstacleDetector.Interface', in_signature='b')
    def SetGUIEnabled(self, enabled: bool):
        print(f"[DBus] GUI display set to: {enabled}")
        self.gui_enabled = enabled

# Instantiate DBus object
obstacle_detector = ObstacleDetector()

def emit_warning(msg: str):
    obstacle_detector.WarningSignal(msg)

def emit_stop(msg: str):
    obstacle_detector.StopSignal(msg)

# Distance estimation
def estimate_distance(box_width_px: int, label: str) -> float:
    real_width_cm = OBJECT_WIDTHS_CM.get(label.lower())
    if real_width_cm and box_width_px >= MIN_BOX_WIDTH:
        dist_m = (real_width_cm * FOCAL_LENGTH) / box_width_px / 100.0
        return dist_m if dist_m <= MAX_DISTANCE_METERS else None
    return None

# Load class names
with open("data/coco.names", "r") as f:
    class_names = [line.strip() for line in f]

# Load model
net = cv2.dnn_DetectionModel(
    "model/frozen_inference_graph.pb",
    "model/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Start camera
picam = Picamera2()
config = picam.create_preview_configuration(
    main={"size": (640, 480)},
    transform=Transform(hflip=1)
)
picam.configure(config)
picam.start()

cv2.namedWindow("Live Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live Detection", 800, 600)

try:
    while True:
        frame = picam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        h, w = frame.shape[:2]

        # Define front zone
        zone_left = int(w * 0.4)
        zone_right = int(w * 0.6)
        zone_top = int(h * 0.5)
        zone_bottom = h
        show_stop = False

        # Detect objects
        outputs = net.detect(frame, confThreshold=0.5)
        class_ids, confidences, boxes = outputs if outputs and len(outputs) == 3 else ([], [], [])

        for cid, conf, box in zip(np.array(class_ids).flatten(), np.array(confidences).flatten(), boxes):
            label = class_names[cid - 1] if 1 <= cid <= len(class_names) else str(cid)
            if label.lower() not in OBJECT_WIDTHS_CM:
                continue

            x, y, bw, bh = box
            if bw < MIN_BOX_WIDTH:
                continue

            cx, cy = x + bw // 2, y + bh // 2
            in_front = (zone_left < cx < zone_right) and (zone_top < cy < zone_bottom)
            dist_m = estimate_distance(bw, label)
            if dist_m is None:
                continue

            close_enough = dist_m < SAFETY_DISTANCE

            if dist_m <= STOP_DISTANCE and in_front:
                show_stop = True
                stop_msg = f"STOP: {label} in front at {dist_m:.2f}m"
                emit_stop(stop_msg)

            # Draw annotations
            color = (0, 0, 255) if in_front and close_enough else (255, 0, 0)
            cv2.rectangle(frame, box, color, 1)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"{dist_m:.2f} m", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if in_front and close_enough:
                warn_msg = f"WARNING: {label} in front at {dist_m:.2f}m"
                cv2.putText(frame, "WARNING!", (x, y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                emit_warning(warn_msg)

        if show_stop:
            cv2.putText(frame, "STOP!", (w // 2 - 100, h // 2), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 255), 4)

        # ?? Show/hide GUI window based on DBus toggle
        if obstacle_detector.gui_enabled:
            cv2.imshow("Live Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            cv2.destroyWindow("Live Detection")

        # Process DBus messages
        while GLib.MainContext.default().pending():
            GLib.MainContext.default().iteration(False)

finally:
    cv2.destroyAllWindows()
    picam.stop()
