# Cozmo Freeroam  
**Powered by PyCozmo + Raspberry Pi 5 (Python 3.10)**  

---

## 🎯 General Objective
To create an independent, AI-powered Cozmo robot that can explore, interact, and learn autonomously using a Raspberry Pi 5 and Python 3.10 — without relying on the original Cozmo SDK or cloud services.

---

## 🚀 Overview
Cozmo Freeroam replaces the original SDK with **PyCozmo**, allowing direct communication between Cozmo and a Raspberry Pi.  
The goal is to make Cozmo fully autonomous, expressive, and interactive — blending robotics, AI, and vision systems into one modular framework.

---

## ⚙️ Core Features
- **Free Exploration:** Smooth, random, and safe movement.  WIP
- **Vision:** Face, object, and cube recognition using OpenCV.  WIP
- **Expression:** Dynamic screen emotions and responsive behaviors.  WIP
- **Pathing:** Path creation, saving, and replay.  WIP
- **AI Layer:** Basic reinforcement learning and mapping.  WIP
- **Optional Web App:** Remote monitoring and map visualization.  WIP

---

## 🧠 System Stack
Cozmo ⇔ Raspberry Pi 5 ⇔ (Optional Webserver)


**Stack Summary:**  
- **Cozmo:** Hardware, sensors, and camera feed  
- **Pi 5:** Main control and AI logic  
- **Webserver:** (Optional) Map, data, and remote control interface  

---

## 🧩 Dependencies
- `pycozmo`  
- `opencv-python`  
- `numpy`  
- `pygame`  
- `flask` *(optional)*  
- `speechrecognition` *(optional)*  
- `pyttsx3` *(optional)*  

Install dependencies:
```bash
pip install -r requirements.txt

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
  config/
    settings.json
  logs/

