import os
import random
from PIL import Image, ImageFilter
import numpy as np

# =========================
# CONFIG
# =========================
SYMBOLS_DIR = "symbols/extracted_images"
OUTPUT_IMG_DIR = "dataset/expression_images"
OUTPUT_LABEL_DIR = "dataset/expression_labels"

NUM_IMAGES = 10000
IMG_WIDTH = 640
IMG_HEIGHT = 320
MAX_SYMBOLS_PER_IMAGE = 12

# =========================
# LOAD SYMBOL PATHS
# =========================
def load_symbols(symbols_dir):
    symbol_paths = {}
    for label in os.listdir(symbols_dir):
        class_dir = os.path.join(symbols_dir, label)
        if not os.path.isdir(class_dir):
            continue
        symbol_paths[label] = [
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if f.endswith(".jpg")
        ]
    return symbol_paths

# =========================
# CREATE CLASS MAP
# =========================
def create_label_map(symbol_paths):
    return {label: idx for idx, label in enumerate(sorted(symbol_paths.keys()))}

# =========================
# CREATE CANVAS
# =========================
def create_canvas():
    return Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), (255, 255, 255))

# =========================
# PLACE SYMBOLS
# =========================
def place_symbols(canvas, symbol_paths):
    annotations = []

    num_lines = random.randint(1, 3)  # up to 3 lines
    line_spacing = random.randint(60, 100)

    start_y = random.randint(40, 80)

    for line in range(num_lines):
        x_cursor = random.randint(5, 20)
        y_baseline = start_y + line * line_spacing

        for _ in range(random.randint(3, MAX_SYMBOLS_PER_IMAGE // num_lines)):
            label = random.choice(list(symbol_paths.keys()))
            img_path = random.choice(symbol_paths[label])

            symbol = Image.open(img_path).convert("RGBA")

            # Scale
            scale = random.uniform(0.5, 1.5)
            w, h = symbol.size
            symbol = symbol.resize((int(w * scale), int(h * scale)))

            # Rotation
            angle = random.uniform(-15, 15)
            symbol = symbol.rotate(angle, expand=True)

            # Vertical jitter
            y_offset = y_baseline + random.randint(-20, 20)

            # Stop if out of bounds
            if x_cursor + symbol.width >= IMG_WIDTH:
                break

            canvas.paste(symbol, (x_cursor, y_offset), symbol)

            # Bounding box
            x1 = x_cursor
            y1 = y_offset
            x2 = x_cursor + symbol.width
            y2 = y_offset + symbol.height

            annotations.append((label, (x1, y1, x2, y2)))

            # Spacing
            x_cursor += symbol.width + random.randint(5, 20)

    return canvas, annotations
# =========================
# ADD NOISE / BLUR
# =========================
def apply_augmentations(image):
    # Convert to numpy
    arr = np.array(image).astype(np.float32)

    # Add Gaussian noise
    noise = np.random.normal(0, 8, arr.shape)
    arr += noise

    # Clip
    arr = np.clip(arr, 0, 255)

    image = Image.fromarray(arr.astype("uint8"))

    # Blur
    if random.random() < 0.7:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    return image

# =========================
# SAVE YOLO LABELS
# =========================
def save_yolo_labels(filepath, annotations, label_map):
    with open(filepath, "w") as f:
        for label, (x1, y1, x2, y2) in annotations:
            x_center = ((x1 + x2) / 2) / IMG_WIDTH
            y_center = ((y1 + y2) / 2) / IMG_HEIGHT
            w = (x2 - x1) / IMG_WIDTH
            h = (y2 - y1) / IMG_HEIGHT

            class_id = label_map[label]
            f.write(f"{class_id} {x_center} {y_center} {w} {h}\n")

# =========================
# MAIN GENERATION LOOP
# =========================
def main():
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

    symbol_paths = load_symbols(SYMBOLS_DIR)
    label_map = create_label_map(symbol_paths)

    #print("Classes:", label_map)

    for i in range(NUM_IMAGES):
        canvas = create_canvas()

        canvas, annotations = place_symbols(canvas, symbol_paths)

        canvas = apply_augmentations(canvas)

        img_filename = f"img_{i:05d}.jpg"
        label_filename = f"img_{i:05d}.txt"

        canvas.save(os.path.join(OUTPUT_IMG_DIR, img_filename))
        save_yolo_labels(
            os.path.join(OUTPUT_LABEL_DIR, label_filename),
            annotations,
            label_map
        )

        if i % 500 == 0:
            print(f"Generated {i} images")

    print("Done!")

if __name__ == "__main__":
    main()