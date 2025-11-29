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

Notes:
- The code contains fallbacks so the server still works even without `torch`, `diffusers` or `ultralytics`:
  - Object detection returns no detections if YOLO isn't installed.
  - Sketch-to-image uses a lightweight PIL stylizer if diffusers/torch aren't installed.
  - GAN generator falls back to procedural noise if torch isn't installed.
- To get full functionality (real detection and diffusion), run `install_requirements.bat full` and ensure you have internet access for model downloads.

Files of interest:
- `app.py` — Flask app with API endpoints for object arena, sketch-to-image, and GAN.
- `models/` — model wrappers with fallbacks.
- `templates/` and `static/` — frontend UI files.

If you'd like, I can now:
- Add a small sample `uploads/sample1.png` and a demo script that uploads it to the detection endpoint.
- Create a Jupyter notebook showing how to train a tiny DCGAN on CIFAR-10 (64x64) and save weights.
- Wire a simple HTML UI for the three modes if the current templates are minimal.
