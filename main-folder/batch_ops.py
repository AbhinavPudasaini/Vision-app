from PIL import Image, ImageDraw, ImageFont
from rembg import remove
import cv2
import numpy as np

def resize_image(img, width, height):
    return img.resize((width, height))

def add_watermark(img, text, position="bottom_right"):
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    text_size = draw.textsize(text, font)

    positions = {
        "bottom_right": (img.width - text_size[0] - 10, img.height - text_size[1] - 10),
        "top_left": (10, 10),
        "center": ((img.width - text_size[0]) // 2, (img.height - text_size[1]) // 2)
    }

    pos = positions.get(position, positions["bottom_right"])
    draw.text(pos, text, fill="white", font=font)
    return img



def blur_background(img):
    # Ensure RGBA
    img = img.convert("RGBA")

    # Remove background to get subject mask
    no_bg = remove(img)

    # Convert to NumPy for OpenCV
    orig = np.array(img)
    subject = np.array(no_bg)

    # Create a blurred version of the original image
    blurred = cv2.GaussianBlur(orig, (21, 21), 0)

    # Use alpha channel of subject to blend
    mask = subject[:, :, 3] > 0  # Alpha > 0 means subject
    blended = blurred.copy()
    blended[mask] = subject[mask][:, :3]  # Replace background with subject

    return Image.fromarray(blended).convert("RGB")


def convert_format(img, fmt):
    return img.convert("RGB") if fmt != "PNG" else img
