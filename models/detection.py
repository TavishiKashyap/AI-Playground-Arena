# models/detection.py
from PIL import Image, ImageDraw
import os

try:
    from ultralytics import YOLO
    _HAS_ULTRALYTICS = True
except Exception:
    YOLO = None
    _HAS_ULTRALYTICS = False


_model = None
if _HAS_ULTRALYTICS:
    try:
        # attempt to load the nano model (this will download weights on first run)
        _model = YOLO("yolov8n.pt")
    except Exception:
        _model = None


def run_object_detection(image_path):
    """
    Returns:
      detections: list of dicts {bbox:[x1,y1,x2,y2], label:str, score:float}
      annotated_img: PIL.Image with drawn boxes

    If Ultralytics/YOLO is not available, returns an empty detection list
    and the original image (so the app remains functional on laptops).
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    if _model is None:
        # fallback: no detections
        return [], img

    try:
        results = _model(image_path)[0]
    except Exception:
        return [], img

    detections = []
    names = getattr(results, 'names', {})
    for box in getattr(results, 'boxes', []):
        try:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            cls_id = int(box.cls[0].item())
            label = names.get(cls_id, str(cls_id))
            score = float(box.conf[0].item())

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "label": label,
                "score": round(score, 3)
            })

            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            text = f"{label} {score:.2f}"
            draw.text((x1 + 3, y1 + 3), text, fill="yellow")
        except Exception:
            # ignore single-box problems
            continue

    return detections, img
