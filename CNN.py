
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ---------------- CONFIG ---------------- #

BASE_DIR = Path("kitchen_mockdata")
IMG_DIR = BASE_DIR / "images"
CSV_PATH = BASE_DIR / "mock_kitchen_data.csv"
MODEL_PATH = BASE_DIR / "cnn_model.pth"

IMG_SIZE = (128, 128)          # width, height
N_SAMPLES = 10000               # total samples, balanced across classes
CLASSES = ["fresh", "warning", "spoiled"]


# ---------------- DIR UTILS ---------------- #

def setup_dirs():
    """Create base and class-specific image directories."""
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    for cls in CLASSES:
        (IMG_DIR / cls).mkdir(parents=True, exist_ok=True)


# ---------------- IMAGE GENERATION ---------------- #

def generate_food_image(label: str, idx: int) -> Path:
    """
    Create a simple synthetic image representing a food item.

    - fresh: greenish, few/no dark spots
    - warning: yellowish, some spots
    - spoiled: brownish, many dark spots

    Returns the path to the saved image.
    """
    w, h = IMG_SIZE
    img = Image.new("RGB", IMG_SIZE, (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Base color depending on label
    if label == "fresh":
        base_color = (50, 160, 50)      # greenish
        n_spots = random.randint(0, 2)
    elif label == "warning":
        base_color = (200, 180, 60)     # yellowish
        n_spots = random.randint(3, 6)
    else:  # spoiled
        base_color = (130, 80, 40)      # brownish
        n_spots = random.randint(7, 15)

    # Fill background
    draw.rectangle([0, 0, w, h], fill=base_color)

    # Add darker "mold" spots
    for _ in range(n_spots):
        x = random.randint(0, w - 10)
        y = random.randint(0, h - 10)
        r = random.randint(5, 20)
        spot_color = (
            max(0, base_color[0] - random.randint(30, 80)),
            max(0, base_color[1] - random.randint(30, 80)),
            max(0, base_color[2] - random.randint(30, 80)),
        )
        draw.ellipse([x, y, x + r, y + r], fill=spot_color)

    # Optional noise
    noise_level = 10
    np_img = np.array(img).astype(np.int16)
    noise = np.random.randint(-noise_level, noise_level + 1, np_img.shape)
    np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(np_img)

    # Save
    filename = f"{label}_{idx:04d}.png"
    img_path = IMG_DIR / label / filename
    img.save(img_path)

    return img_path


# ---------------- TABULAR MOCK DATA ---------------- #

STORAGE_LOCATIONS = ["fridge_1", "fridge_2", "freezer_1", "shelf_1"]

def generate_tabular_row(item_id: int, label: str, img_path: Path) -> dict:
    """
    Create one row of sensor-style mock data that is consistent with the label.
    """
    item_type = random.choice(["chicken", "beef", "milk", "lettuce", "tomato"])
    storage_location = random.choice(STORAGE_LOCATIONS)

    # Base distributions depending on label
    if label == "fresh":
        hours_since_delivery = random.uniform(1, 48)
        expiry_hours_remaining = random.uniform(24, 120)
        visual_mold_score = random.randint(0, 1)
    elif label == "warning":
        hours_since_delivery = random.uniform(24, 120)
        expiry_hours_remaining = random.uniform(0, 24)
        visual_mold_score = random.randint(2, 3)
    else:  # spoiled
        hours_since_delivery = random.uniform(48, 200)
        expiry_hours_remaining = random.uniform(-48, 0)
        visual_mold_score = random.randint(4, 5)

    # Temperature & humidity depending on storage + label
    if "fridge" in storage_location:
        if label == "fresh":
            temperature_c = random.uniform(1, 5)
        elif label == "warning":
            temperature_c = random.uniform(4, 10)
        else:
            temperature_c = random.uniform(8, 18)
        humidity_percent = random.uniform(60, 90)
    elif "freezer" in storage_location:
        if label == "fresh":
            temperature_c = random.uniform(-20, -10)
        elif label == "warning":
            temperature_c = random.uniform(-15, -5)
        else:
            temperature_c = random.uniform(-10, 5)  # half-thawed
        humidity_percent = random.uniform(20, 60)
    else:  # shelf
        if label == "fresh":
            temperature_c = random.uniform(15, 22)
        elif label == "warning":
            temperature_c = random.uniform(20, 28)
        else:
            temperature_c = random.uniform(25, 35)
        humidity_percent = random.uniform(30, 80)

    # CO2 – higher for spoiled
    if label == "fresh":
        co2_ppm = random.uniform(400, 800)
    elif label == "warning":
        co2_ppm = random.uniform(700, 1400)
    else:
        co2_ppm = random.uniform(1200, 2500)

    return {
        "item_id": item_id,
        "item_type": item_type,
        "storage_location": storage_location,
        "temperature_c": round(temperature_c, 2),
        "humidity_percent": round(humidity_percent, 2),
        "co2_ppm": round(co2_ppm, 2),
        "hours_since_delivery": round(hours_since_delivery, 2),
        "expiry_hours_remaining": round(expiry_hours_remaining, 2),
        "visual_mold_score": visual_mold_score,
        "label": label,
        "image_path": str(img_path),
    }


def create_mock_dataset(n_samples: int = N_SAMPLES) -> pd.DataFrame:
    """
    Generate images + tabular data for all samples, balanced across labels.
    """
    setup_dirs()
    rows = []
    item_id = 1
    samples_per_class = n_samples // len(CLASSES)

    for label in CLASSES:
        for i in range(samples_per_class):
            img_path = generate_food_image(label, i)
            row = generate_tabular_row(item_id, label, img_path)
            rows.append(row)
            item_id += 1

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(df)} samples to {CSV_PATH}")
    return df


# ---------------- PYTORCH DATASET ---------------- #

class KitchenImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(CLASSES)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        label_str = row["label"]
        label = self.class_to_idx[label_str]

        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label


# ---------------- CNN MODEL ---------------- #

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = len(CLASSES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),    # 64x64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),    # 32x32

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),    # 16x16

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),    # 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------- TRAINING LOOP ---------------- #

def train_model(model, train_loader, val_loader, device, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.to(device)

    for epoch in range(1, epochs + 1):
        # ---- TRAIN ----
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}"
        )
def predict_single_image(model, image_path: Path, transform, device):
    """Predict class for a single external image."""
    model.eval()
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = int(torch.argmax(probs))
        pred_label = CLASSES[pred_idx]

    print("\nPrediction on external image:")
    print(f"Image path: {image_path}")
    print(f"Predicted label: {pred_label}")
    print("Class probabilities:")
    for cls, p in zip(CLASSES, probs.cpu().numpy()):
        print(f"  {cls}: {p:.3f}")


# ---------------- MAIN ---------------- #

def main():
    # 1. Create mock images + tabular data
    df = create_mock_dataset(N_SAMPLES)

    # 2. Train/validation split
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    train_frac = 0.8
    n_train = int(len(df) * train_frac)

    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train:].reset_index(drop=True)

    # 3. Define transforms
    transform = T.Compose([
        T.Resize(IMG_SIZE),
        T.ToTensor(),
        # Optionally normalize (values chosen roughly, not critical for mock data)
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
    ])

    # 4. Create datasets & loaders
    train_dataset = KitchenImageDataset(train_df, transform=transform)
    val_dataset = KitchenImageDataset(val_df, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # 5. Build and train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleCNN(num_classes=len(CLASSES))
    print(model)

    train_model(model, train_loader, val_loader, device, epochs=5)

    # 6. Save model weights
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model weights saved to {MODEL_PATH}")

    # 7. Example prediction on one validation image
    model.eval()
    sample_row = val_df.sample(1).iloc[0]
    img_path = sample_row["image_path"]
    true_label = sample_row["label"]

    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)  # add batch dimension

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = int(torch.argmax(probs))
        pred_label = CLASSES[pred_idx]

    print("\nExample prediction:")
    print(f"Image path: {img_path}")
    print(f"True label: {true_label}")
    print(f"Predicted label: {pred_label}")
    print("Class probabilities:")
    for cls, p in zip(CLASSES, probs.cpu().numpy()):
        print(f"  {cls}: {p:.3f}")
        
     # 8. Predict on an external test image (simulating camera input)
    test_img_path = BASE_DIR / "banana.png"

    if test_img_path.exists():
        predict_single_image(model, test_img_path, transform, device)
    else:
        print("\nTest image not found. Place banana.png in kitchen_mockdata/")

        


if __name__ == "__main__":
    main()
