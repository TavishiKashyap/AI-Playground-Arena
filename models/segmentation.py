# models/segmentation.py
from PIL import Image
import numpy as np


def simple_segmentation_mask(image_path):
    """
    Placeholder: returns a crude mask.
    Later: replace with real segmentation model (U-Net / Fast-SCNN).
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    h, w, _ = arr.shape

    # simple fake mask: top half = 1, bottom half = 0
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:h // 2, :] = 1

    return mask  # (h, w) with values 0/1
