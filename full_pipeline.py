import pickle
from ultralytics import YOLO
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from sklearn.linear_model import LogisticRegression
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import os

img_dir = "dataset/images/val"
label_dir = "dataset/labels"

image_files = sorted(os.listdir(img_dir))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo_model = YOLO("best.pt")
with open("lr_model.pkl", 'rb') as file:
    lr_model = pickle.load(file)

def load_yolo_labels(label_path, img_w, img_h):
    boxes = []

    with open(label_path, "r") as f:
        for line in f.readlines():
            cls, x, y, w, h = map(float, line.split())

            x1 = int((x - w/2) * img_w)
            y1 = int((y - h/2) * img_h)
            x2 = int((x + w/2) * img_w)
            y2 = int((y + h/2) * img_h)

            boxes.append((int(cls), x1, y1, x2, y2))

    return boxes

# LR Feature extraction
def build_feature_vector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
    pixels = small.flatten() / 255.0   # normalize

    # --- combine ---
    return np.concatenate([pixels, extra_features])

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

# extract boxes
def crop_from_gt(img, gt_boxes):
    crops = []

    for cls, x1, y1, x2, y2 in gt_boxes:
        crop = img[y1:y2, x1:x2]

        if crop is None or crop.size == 0:
            continue

        if crop.size > 0:
            crops.append((crop, cls))

    return crops

# LR validation
def validate_lr(image_path, true_label):
    crops = crop_from_gt(image_path)

    preds = []

    for crop in crops:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (16, 16)).flatten() / 255.0

        pred = lr_model.predict([gray])[0]
        preds.append(pred)

    # majority vote (important for multiple boxes)
    final_pred = max(set(preds), key=preds.count)

    return final_pred == true_label

cnn_model.eval()

cnn_preds = []
lr_preds = []
true_labels = []

for img_name in image_files:

    img_path = os.path.join(img_dir, img_name)
    print("Processing image:", img_path)
    label_path = os.path.join(label_dir, img_name.replace(".jpg", "_unmasked.txt"))
    if not os.path.exists(label_path):
        print(f"Missing label: {label_path}")
        continue
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    gt_boxes = load_yolo_labels(label_path, w, h)
    crops = crop_from_gt(img, gt_boxes)

    for crop, true_label in crops:

        # LR
        feat = build_feature_vector(crop)
        lr_pred = lr_model.predict([feat])[0]

        lr_preds.append(lr_pred)
        true_labels.append(true_label)

        # CNN
        crop_resized = cv2.resize(crop, (64, 64))
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)

        tensor = torch.tensor(crop_rgb).permute(2,0,1).float()/255.0
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            cnn_pred = cnn_model(tensor).argmax(dim=1).item()

        cnn_preds.append(cnn_pred)
        
print(cnn_preds)
print(lr_preds)
print(true_labels)
# Results
cnn_acc = accuracy_score(true_labels, cnn_preds)
lr_acc = accuracy_score(true_labels, lr_preds)

print("CNN Accuracy:", cnn_acc)
print("LR Accuracy:", lr_acc)

cm = confusion_matrix(true_labels, cnn_preds)
print(cm)
cm2 = confusion_matrix(true_labels, lr_preds)
print(cm2)
# generate tex and pdf files
