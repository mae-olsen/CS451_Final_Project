import cv2
import torch
import numpy as np
from ultralytics import YOLO
import cv2
import torch
import pickle
import torch.nn as nn
from IPython.display import display, Image as IPImage
import subprocess
import os
import argparse

idx_to_label = {
    0: "(",
    1: ")",
    2: "+",
    3: "-",
    4: "0",
    5: "1",
    6: "2",
    7: "3",
    8: "4",
    9: "5",
    10: "6",
    11: "7",
    12: "8",
    13: "9",
    14: "=",
    15: "X",
    16: "div",
    17: "f",
    18: "forward_slash",
    19: "neq",
    20: "times",
    21: "y"
}

def preprocess_cnn(crop):
    crop = cv2.resize(crop, (64, 64))
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    tensor = torch.tensor(crop).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)

def extract_lr_features(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # --- engineered features ---
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges) / 255.0

    avg_intensity = np.mean(gray) / 255.0

    y_coords, x_coords = np.indices(gray.shape)
    total = np.sum(gray) + 1e-8

    cx = np.sum(x_coords * gray) / total / gray.shape[1]
    cy = np.sum(y_coords * gray) / total / gray.shape[0]

    extra_features = np.array([edge_density, avg_intensity, cx, cy])

    # --- image features ---
    small = cv2.resize(gray, (16, 16))
    pixels = small.flatten() / 255.0

    return np.concatenate([pixels, extra_features]).reshape(1, -1)


yolo = YOLO("best.pt")

def get_crops(image_path, iou=0.05, conf=0.4):
    results = yolo(image_path, iou=iou, conf=conf)  # pass PATH to yolo
    img = cv2.imread(image_path)                     # read array for cropping
    boxes_and_crops = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                boxes_and_crops.append((x1, y1, x2, y2, crop))

    boxes_and_crops = sorted(boxes_and_crops, key=lambda b: (b[1] // 80, b[0]))
    return [crop for *_, crop in boxes_and_crops]


def infer(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    crops = get_crops(image_path)  # pass path, not array

    results = []
    for crop in crops:
        cnn_input = preprocess_cnn(crop).to(device)
        with torch.no_grad():
            cnn_pred = cnn_model(cnn_input).argmax(dim=1).item()

        lr_input = extract_lr_features(crop)
        lr_pred = lr_model.predict(lr_input)[0]

        results.append({
            "cnn": idx_to_label[cnn_pred],
            "lr": idx_to_label[lr_pred]
        })

    return results


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # RGB input
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),  # correct for 64x64 input
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


cnn_model = CNN(num_classes=22)

state_dict = torch.load('cnn_model_weights.pth', weights_only=True, map_location=torch.device("cpu"))
cnn_model.load_state_dict(state_dict)
cnn_model.eval()
lr_model = pickle.load(open("lr_model.pkl", "rb"))


def visualize_boxes(image_path, results):
    """Draws bounding boxes and labels on the image — useful for debugging."""
    img = cv2.imread(image_path)
    for (x, y, w, h), pred, conf in results:
        label = idx_to_label.get(pred, "?")
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite("debug_boxes.jpg", img)
    print("Saved debug_boxes.jpg")

#result = infer('expression3.jpg')
#visualize_boxes("expression.jpg", result)
#print(result)

#results = yolo("expression3.jpg", iou=0.3, conf=0.5)

# Save annotated image with boxes drawn
#results[0].save("debug_yolo3.jpg")
#display(IPImage("debug_yolo3.jpg"))

# to latex
# -----------------------------
# LATEX TOKEN MAP
# -----------------------------
latex_map = {
    "(":            "(",
    ")":            ")",
    "+":            "+",
    "-":            "-",
    "=":            "=",
    "0":            "0",
    "1":            "1",
    "2":            "2",
    "3":            "3",
    "4":            "4",
    "5":            "5",
    "6":            "6",
    "7":            "7",
    "8":            "8",
    "9":            "9",
    "X":            "x",
    "y":            "y",
    "f":            "f",
    "div":          r"\div",
    "times":        r"\times",
    "neq":          r"\neq",
    "forward_slash": "/",
}

# -----------------------------
# DECODE CNN RESULTS TO LATEX
# -----------------------------
def decode_to_latex(results):
    tokens = []
    for r in results:
        label = r["cnn"]
        token = latex_map.get(label, "?")
        tokens.append(token)
    return " ".join(tokens)

# -----------------------------
# WRITE .tex FILE
# -----------------------------
def write_tex(latex_exprs, tex_path="output.tex", 
              title="Math OCR Output", 
              author="Your Name", 
              date=None):
    """
    latex_exprs: either a single string or a list of strings (one per line)
    """
    if isinstance(latex_exprs, str):
        latex_exprs = [latex_exprs]

    if date is None:
        date = r"\today"

    # Build one \[ ... \] block per expression
    equations = "\n\n".join(f"\\[\n{expr}\n\\]" for expr in latex_exprs)

    content = r"""\documentclass{article}
\usepackage{amsmath}
\usepackage{amssymb}

\title{""" + title + r"""}
\author{""" + author + r"""}
\date{""" + date + r"""}

\begin{document}

\maketitle

""" + equations + r"""

\end{document}
"""
    with open(tex_path, "w") as f:
        f.write(content)
    print(f"Saved: {tex_path}")

# -----------------------------
# COMPILE .tex TO PDF
# -----------------------------
def compile_tex(tex_path="output.tex"):
    # pdflatex must be installed — on Ubuntu: sudo apt install texlive
    # on Windows: install MiKTeX or TeX Live
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", tex_path],
        capture_output=True,
        text=True
    )
    
    pdf_path = tex_path.replace(".tex", ".pdf")
    
    if os.path.exists(pdf_path):
        print(f"Saved: {pdf_path}")
    else:
        print("pdflatex failed. Log:")
        print(result.stdout[-1000:])  # last 1000 chars of log
    
    # Clean up auxiliary files pdflatex generates
    for ext in [".aux", ".log"]:
        aux = tex_path.replace(".tex", ext)
        if os.path.exists(aux):
            os.remove(aux)

# -----------------------------
# RUN
# -----------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--img_dir", help = "image dataset directory", type=str) #specify string
args = parser.parse_args()
image_dir = args.img_dir
image_paths = []
for root, dirs, files in os.walk(image_dir):
    for f in files:
        if f.endswith(('.jpg', '.png', 'jpeg')):
            image_paths.append(os.path.join(root, f))  # full path, not just filename
print(image_paths)
latex_exprs = []
for path in image_paths:
    result = infer(path)
    print(result)
    latex_expr = decode_to_latex(result)
    print(f"{path}: {latex_expr}")
    latex_exprs.append(latex_expr)

write_tex(
    latex_exprs,
    tex_path="output.tex",
    title="Handwritten Math OCR",
    author="Mae Olsen",
    date="\\today"  # or omit for \today
)
compile_tex("output.tex")