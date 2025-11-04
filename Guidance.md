Cozmo Freeroam — Developer Implementation Plan (Pi 5 + PyCozmo)
System Overview

Core stack:
Cozmo ⇔ Raspberry Pi 5 ⇔ (Optional Webserver/UI)
Languages: Python 3.11+
Libraries: pycozmo, opencv-python, numpy, flask (optional), pygame, speechrecognition, pyttsx3

Phase 1 — Core Loop and Safety (Beta)
Objectives

Establish communication loop between Cozmo and Pi 5.

Implement smooth, randomized motion.

Add camera feed + OpenCV pipeline.

Implementation

Communication

Use pycozmo.connect() and heartbeat loop.

Maintain a non-blocking control loop (asyncio or threaded).

Motion control

import pycozmo, random, time

def explore_loop(robot):
    while True:
        speed = random.randint(50, 100)
        angle = random.choice([-30, 0, 30])
        robot.drive_wheels(speed, speed)
        time.sleep(1)
        robot.turn_in_place(angle)


Collision & safety

Use Cozmo’s cliff sensors + IMU (if available via PyCozmo).

Implement emergency stop on threshold.

Vision

Capture camera frames from Cozmo (robot.camera.latest_image)

Convert to numpy and feed to cv2 for face detection:

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.3, 5)

Phase 2 — Exploration + Interaction (Pre-V1)
Objectives

Add interactive exploration (react to faces, objects).

Basic path recording and replay.

Implementation

Path system

Maintain list of (x, y, angle) waypoints via odometry.

Serialize paths to JSON.

Expressive screen

Use pycozmo.display_text() for eye animations.

Preload emotion states (happy, curious, idle).

Face recognition

Store known faces using face_recognition library.

Match new detections; trigger expression change or sound.

Object & cube recognition

Use color/shape segmentation in OpenCV.

Simple centroid tracking for approach behavior.

Phase 3 — Autonomy + Charging (V1)
Objectives

Implement home-base navigation.

Add simple games.

Add environmental audio responses.

Implementation

Return to charger

Define AprilTag or AR marker on charger.

When battery low → detect marker → navigate via vision and odometry.

Minigames

Example: cube reaction (Cozmo taps cube when user waves hand).

Use random state machine controlling actions.

Audio

Use pygame.mixer or pyttsx3 for sounds tied to states (idle, excited).

Phase 4 — AI Framework Integration (V1.5)
Objectives

Map environment.

Implement rudimentary spatial awareness.

Link webserver dashboard for control + map view.

Implementation

Mapping

Use dead-reckoning + visual landmarks → 2D occupancy grid (numpy array).

Visualize map via Flask webserver endpoint using matplotlib → PNG.

AI Framework

Basic reinforcement learning loop (Q-table of movement → reward).

Save policy on Pi5 for reuse.

Sound generation

Procedural tones using pyo or simpleaudio.

Phase 5 — Natural Interaction & Voice (V2)
Objectives

Add voice I/O.

Smooth movement.

Train lightweight behavioral model.

Implementation

Speech

STT: speech_recognition + local model (e.g. Whisper.cpp).

TTS: pyttsx3 for offline speech.

Behavior model

Use small neural net (PyTorch Lite) for state → motion/sound mapping.

Train on logs of human-Cozmo interactions.

Smooth motion

Implement bezier or PID smoothing on wheel velocity commands.

Phase 6 — Dedicated AI & Mapping Visualization (Beyond)
Objectives

Separate motion AI (RL-trained in Isaac Lab).

Interaction AI on Pi 5.

User-teachable object recognition.

Block-coding interface.

Implementation

AI separation

ai_core.py: controls navigation and movement (trained model).

interaction_core.py: handles vision, audio, expression, and dialogue.

User-teachable objects

Collect images, store embeddings via opencv + sklearn.

Train incremental classifier (e.g. SGDClassifier.partial_fit).

Web interface

Flask or FastAPI with WebSocket stream from Cozmo’s camera.

Block-coding frontend using Blockly or ScratchBlocks for user programming.

Optimization

Run inference with torch.compile or ONNX Runtime on Pi 5.

Offload non-realtime tasks (training) to external PC if needed.

Development Notes

Use modular directory:

/cozmo_freeroam/
  core/
    control.py
    vision.py
    mapping.py
    ai.py
  web/
    app.py
  assets/
    sounds/
    expressions/


Commit early; use GitHub Actions for lint + unit tests.

Log every sensor reading and decision for later model training.