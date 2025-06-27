from PIL import Image, ImageDraw, ImageFont
from rembg import remove
import cv2
import numpy as np

def resize_image(img, width, height):
    return img.resize((width, height))

def add_watermark(img, text, position="bottom_right"):
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    text_size = draw.textsize(text, font=font)


    positions = {
        "bottom_right": (img.width - text_size[0] - 10, img.height - text_size[1] - 10),
        "top_left": (10, 10),
        "center": ((img.width - text_size[0]) // 2, (img.height - text_size[1]) // 2)
    }

    pos = positions.get(position, positions["bottom_right"])
    draw.text(pos, text, fill="white", font=font)
    return img

def blur_background(image, blur_strength=50):
    # Convert PIL Image to NumPy array
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    # Blur the entire image
    blurred = cv2.GaussianBlur(image, (81, 31), 0)

    # Keep original foreground where mask is white
    mask_inv = cv2.bitwise_not(mask)
    foreground = cv2.bitwise_and(image, image, mask=mask)
    background = cv2.bitwise_and(blurred, blurred, mask=mask_inv)

    # Combine both
    final = cv2.add(foreground, background)

    cv2.imwrite('blurred_background3.jpg', final)



def convert_format(img, fmt):
    return img.convert("RGB") if fmt != "PNG" else img
def remove_background(img):
    # Convert PIL Image to NumPy array
    img_np = np.array(img)

    # Remove background using rembg
    img_no_bg = remove(img_np)

    # Convert back to PIL Image
    return Image.fromarray(img_no_bg)

import cv2
import numpy as np

# Load image
# image = cv2.imread('input.jpg')

### 1. Add Watermark ###
def add_watermark(img, text="Watermark", pos=(30, 30), opacity=0.5):
    overlay = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    color = (255, 255, 255)
    thickness = 5
    cv2.putText(overlay, text, pos, font, scale, color, thickness, cv2.LINE_AA)
    return cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0)

### 2. Enhance Image (Brightness and Contrast) ###
def enhance_image(img, alpha=1.2, beta=20):
    # alpha: contrast (1.0-3.0), beta: brightness (0-100)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

### 3. Add Border Frame ###
def add_border(img, border_thickness=20, border_color=(0, 0, 0)):
    return cv2.copyMakeBorder(img, border_thickness, border_thickness, border_thickness, border_thickness,
                              cv2.BORDER_CONSTANT, value=border_color)

#


if __name__ == "__main__":
    # Load the image
    image = cv2.imread("C:\\Users\\Abhinav\\Downloads\\photo2.jpg")
    image = enhance_image(image)
    image = add_watermark(image, text="© Abhinav", pos=(50, 50), opacity=0.6)
    image = add_border(image, border_thickness=30, border_color=(100, 0, 200))  # purple border

    # Save the final image
    cv2.imwrite('output2.jpg', image)
    print("Image saved as output.jpg")