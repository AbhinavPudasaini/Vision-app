import os
from PIL import Image
from batch_ops import resize_image, add_watermark, blur_background, convert_format

# Configurable folders
INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_IMAGES = 30

def prompt_choice(prompt, options):
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        print(f"[{i}] {option}")
    choice = input("Enter your choice (number or leave blank to skip): ").strip()
    return int(choice) if choice.isdigit() and 1 <= int(choice) <= len(options) else None

def process_batch():
    print("📦 Batch Processing Started")

    # -- Resize Option --
    resize_choice = prompt_choice("Choose resize option", ["Resize to custom width/height", "Skip resizing"])
    if resize_choice == 1:
        width = int(input("Enter width: "))
        height = int(input("Enter height: "))

    # -- Watermark Option --
    watermark_choice = prompt_choice("Choose watermark option", ["Add watermark", "Skip watermark"])
    if watermark_choice == 1:
        watermark_text = input("Enter watermark text: ")
        watermark_position = input("Enter watermark position (top_left / bottom_right / center): ").strip()

    # -- Background Blur Option --
    blur_choice = prompt_choice("Blur background?", ["Yes", "No"])

    # -- Format Conversion --
    convert_choice = prompt_choice("Choose format to convert", ["JPEG", "PNG", "WEBP", "Skip conversion"])
    if convert_choice in [1, 2, 3]:
        formats = ["JPEG", "PNG", "WEBP"]
        output_format = formats[convert_choice - 1]
    else:
        output_format = None

    # Load images (limit 30)
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:MAX_IMAGES]

    for img_name in image_files:
        print(f"\n🔄 Processing: {img_name}")
        img_path = os.path.join(INPUT_DIR, img_name)
        img = Image.open(img_path).convert("RGBA")

        # Apply operations
        if resize_choice == 1:
            img = resize_image(img, width, height)

        if blur_choice == 1:
            img = blur_background(img)

        if watermark_choice == 1:
            img = add_watermark(img, watermark_text, position=watermark_position)

    

        if output_format:
            img = convert_format(img, output_format)
            out_name = os.path.splitext(img_name)[0] + f"_processed.{output_format.lower()}"
        else:
            out_name = os.path.splitext(img_name)[0] + "_processed.png"

        # Save output
        out_path = os.path.join(OUTPUT_DIR, out_name)
        img.save(out_path)
        print(f"✅ Saved: {out_path}")

    print("\n🎉 Batch processing complete.")

if __name__ == "__main__":
    process_batch()


# background remove
# enhance image
# add border
# add text overlay
# add frame
# add filter
# add color correction
# add tags/ categories
