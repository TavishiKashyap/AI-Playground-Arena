from flask import Flask, render_template, request, jsonify
import os
import base64
from io import BytesIO
from PIL import Image
from datetime import datetime

from models.detection import run_object_detection
from models.segmentation import simple_segmentation_mask
from models.sketch_diffusion import sketch_to_image
from models.gan_playground import generate_gan_image

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


def save_uploaded_image(file_storage, prefix="img"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{prefix}_{ts}.png"
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file_storage.save(path)
    return path


def pil_to_base64(img: Image.Image) -> str:
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.route("/")
def index():
    return render_template("index.html")


# ========== 1) OBJECT REMOVAL ARENA ==========

@app.route("/api/detect_objects", methods=["POST"])
def api_detect_objects():
    """
    Input: image file
    Output: detected bboxes with labels & scores
    """
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400

    path = save_uploaded_image(request.files["image"], prefix="det")
    detections, annotated_img = run_object_detection(path)

    # return annotated image + bbox metadata
    annotated_b64 = pil_to_base64(annotated_img)

    return jsonify({
        "bboxes": detections,
        "annotated_image": annotated_b64
    })


@app.route("/api/object_edit", methods=["POST"])
def api_object_edit():
    """
    Input: image + bbox actions
    {
      "image": base64,
      "actions": [
         {"bbox": [x1,y1,x2,y2], "action": "remove"},
         {"bbox": [...], "action": "keep"}
      ]
    }
    Output: edited image (inpainted / blurred regions)
    """
    data = request.get_json()
    if not data or "image" not in data or "actions" not in data:
        return jsonify({"error": "Invalid payload"}), 400

    img_b64 = data["image"]
    actions = data["actions"]

    img_bytes = base64.b64decode(img_b64)
    img = Image.open(BytesIO(img_bytes)).convert("RGB")

    # We will do a simple effect: blur regions where action == "remove"
    import cv2
    import numpy as np

    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    for act in actions:
        x1, y1, x2, y2 = act["bbox"]
        if act["action"] == "remove":
            roi = cv_img[y1:y2, x1:x2]
            if roi.size > 0:
                roi_blur = cv2.GaussianBlur(roi, (15, 15), 0)
                cv_img[y1:y2, x1:x2] = roi_blur

    edited_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    edited_b64 = pil_to_base64(edited_img)

    return jsonify({"edited_image": edited_b64})


# ========== 2) SKETCH TO IMAGE ==========

@app.route("/api/sketch_to_image", methods=["POST"])
def api_sketch_to_image():
    """
    Input: sketch image + optional style parameters
    """
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400

    guidance_scale = float(request.form.get("guidance_scale", 3.0))
    num_steps = int(request.form.get("num_steps", 15))

    path = save_uploaded_image(request.files["image"], prefix="sketch")
    out_img = sketch_to_image(path,
                              guidance_scale=guidance_scale,
                              num_inference_steps=num_steps)
    out_b64 = pil_to_base64(out_img)

    return jsonify({"generated_image": out_b64})


# ========== 3) GAN PLAYGROUND ==========

@app.route("/api/gan_generate", methods=["POST"])
def api_gan_generate():
    """
    Input: latent_dim, noise_scale
    """
    data = request.get_json()
    latent_dim = int(data.get("latent_dim", 16))
    noise_scale = float(data.get("noise_scale", 1.0))

    img = generate_gan_image(latent_dim=latent_dim,
                             noise_scale=noise_scale)
    img_b64 = pil_to_base64(img)
    return jsonify({"generated_image": img_b64})


if __name__ == "__main__":
    app.run(debug=True)
