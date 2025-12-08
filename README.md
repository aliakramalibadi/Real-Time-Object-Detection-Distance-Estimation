🚗 Real-Time Object Detection & Distance Estimation
Smart Vehicle Computer Vision System (Python + OpenCV + MobileNet + D-Bus + Raspberry Pi)

This project implements a complete vision-based safety system capable of:

Real-time object detection using MobileNet SSD

Distance estimation based on camera geometry

Automatic WARNING and STOP events for close obstacles

Raspberry Pi live camera streaming using Picamera2

D-Bus messaging to communicate hazard alerts to other system modules

Designed for smart vehicles, robotics, and ADAS-style applications.

🔍 Project Features:
✔ Real-Time Object Detection with 30 FPS

Detects: person, car, bus, truck, motorcycle

✔ Distance Estimation

Uses the formula:

Distance = (Real Width * Focal Length) / Bounding Box Width

Supports adjustable safety distances:

SAFETY_DISTANCE (default: 2.0m → WARNING)

STOP_DISTANCE (default: 1.0m → STOP)

✔ Raspberry Pi Camera Support

Uses Picamera2 + libcamera for low-latency video capture.

✔ D-Bus Safety Alerts

Sends IPC signals across a Linux-based vehicle system:

WarningSignal("WARNING: object at 2.0m")

StopSignal("STOP: object at 1.0m")

This allows your vision system to integrate with:

Vehicle control unit

Dashboard UI

Safety module

Braking logic (future integration)

✔ GUI Toggle Over D-Bus

Another module can call:

SetGUIEnabled(true/false)

to show or hide the OpenCV live display.

🧠 Technology Stack:

Python 3

OpenCV (cv2.dnn)

NumPy

MobileNet SSD (TensorFlow model)

Picamera2 + libcamera

D-Bus (Python bindings)

GLib event loop

Designed for Raspberry Pi

▶️ Running the Project:

1. Install dependencies
pip install -r requirements.txt


For Raspberry Pi:

sudo apt install python3-picamera2 -y
sudo apt install python3-dbus python3-gi

2. Run offline video detection
python car_vision.py

3. Run Raspberry Pi live detection with D-Bus
python car_vision_Dbus.py

Press Q to exit the GUI.

🎯 Why This Project Is Valuable:

This project demonstrates real-world skills required for work in:

Autonomous vehicles

Robotics

ADAS systems

Embedded vision

Realtime AI inference

IoT and Linux-based automotive systems

🎓 What I Learned:

Deep learning vision inference on embedded hardware

Designing safety zones and alerts for vehicle perception

D-Bus IPC for automotive modular architecture

Performance tuning on Raspberry Pi

Real-world geometry for distance estimation

Multi-process communication in robotics systems

Building a complete perception → logic → output pipeline
