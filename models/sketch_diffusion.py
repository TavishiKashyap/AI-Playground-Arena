# models/sketch_diffusion.py
from PIL import Image, ImageFilter, ImageOps
import numpy as np

_HAS_DIFFUSERS = True
try:
    from diffusers import StableDiffusionImg2ImgPipeline
    import torch
except Exception:
    StableDiffusionImg2ImgPipeline = None
    torch = None
    _HAS_DIFFUSERS = False

# You can change this to a lighter model if needed
MODEL_ID = "stabilityai/sd-turbo"

_pipe = None
_device = None
if _HAS_DIFFUSERS and torch is not None:
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        _pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if _device == "cuda" else torch.float32
        )
        _pipe = _pipe.to(_device)
    except Exception:
        _pipe = None


def _fallback_stylize(image_path, style="cartoon"):
    """Lightweight fallback: apply simple PIL filters to simulate a style."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((512, 512))
    if style == "cartoon":
        img = img.filter(ImageFilter.EDGE_ENHANCE)
        img = img.filter(ImageFilter.SMOOTH_MORE)
    elif style == "pencil":
        img = ImageOps.grayscale(img).filter(ImageFilter.FIND_EDGES)
        img = ImageOps.colorize(img, black="black", white="#f0f0f0")
    else:
        img = img.filter(ImageFilter.DETAIL)
    return img


def sketch_to_image(image_path,
                    guidance_scale=3.0,
                    num_inference_steps=15,
                    prompt="a cute digital art, clean, high quality",
                    style="cartoon"):
    """
    Convert rough sketch to nicer image using img2img. If diffusers/torch
    are not available, uses a lightweight PIL-based stylization fallback.
    """
    if _pipe is None:
        return _fallback_stylize(image_path, style=style)

    init_image = Image.open(image_path).convert("RGB")
    init_image = init_image.resize((512, 512))

    # Light preprocessing: increase contrast / threshold to emphasize sketch lines
    arr = np.array(init_image.convert("L"))
    arr = (arr < 200) * 255
    init_image = Image.fromarray(arr).convert("RGB")

    try:
        if _device == "cuda":
            with torch.autocast(_device):
                out = _pipe(
                    prompt=prompt,
                    image=init_image,
                    strength=0.8,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                )
        else:
            out = _pipe(
                prompt=prompt,
                image=init_image,
                strength=0.8,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )
        gen_img = out.images[0]
        return gen_img
    except Exception:
        return _fallback_stylize(image_path, style=style)
