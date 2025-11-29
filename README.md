# AI Playground Arena — Run locally on your laptop

This repository contains a lightweight, student-friendly implementation of the "AI Playground Arena" web app. It is designed to run on a laptop and to gracefully degrade when heavy ML libraries are not installed.

Quick steps (Windows CMD):

1) Install lightweight dependencies (recommended for quick demo):

```
cd c:\Users\HP\Desktop\Robotics
install_requirements.bat
```

2) (Optional) Install full ML stack (diffusers, torch, ultralytics):

```
cd c:\Users\HP\Desktop\Robotics
install_requirements.bat full
```

3) Run the Flask app:

```
python app.py
```

4) Open the app in your browser:

```
http://127.0.0.1:5000/
```



