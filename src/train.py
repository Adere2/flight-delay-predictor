import glob  # <--- NEW: To find files
import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data import DataProcessor, FlightDataset
from src.model import FlightPredictor


def validate(model, loader, crit_reg, crit_cls, device):
    model.eval()
    total_loss = 0
    correct_cls = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            x_cat = batch["x_cat"].to(device)
            x_num = batch["x_num"].to(device)
            y_r = batch["y_reg"].to(device).unsqueeze(1)
            y_c = batch["y_cls"].to(device).unsqueeze(1)

            preds_r, preds_c = model(x_cat, x_num)
            loss = crit_reg(preds_r, y_r) + crit_cls(preds_c, y_c)
            total_loss += loss.item()

            probs = torch.sigmoid(preds_c)
            predicted_labels = (probs > 0.5).float()
            correct_cls += (predicted_labels == y_c).sum().item()
            total_samples += y_c.size(0)

    return total_loss / len(loader), correct_cls / total_samples


def train_pipeline(data_dir, models_dir, epochs=10, batch_size=64):
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"RUNNING ON {device}...")

    # 2. Find Files
    # If the user passed a directory, grab all .csv files inside
    if os.path.isdir(data_dir):
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        print(f"Found {len(csv_files)} CSV files in {data_dir}")
    else:
        # Fallback if they passed a specific file
        csv_files = [data_dir]

    if not csv_files:
        print("No CSV files found!")
        return

    # 3. Process Data
    processor = DataProcessor()
    X_cat, X_num, y_reg, y_cls = processor.process_raw_data(csv_files, is_training=True)

    # 4. SPLIT (Indices)
    indices = list(range(len(X_cat)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    train_dataset = FlightDataset(
        X_cat[train_idx], X_num[train_idx], y_reg[train_idx], y_cls[train_idx]
    )
    val_dataset = FlightDataset(
        X_cat[val_idx], X_num[val_idx], y_reg[val_idx], y_cls[val_idx]
    )

    print(
        f"Training on {len(train_dataset)} flights. Validating on {len(val_dataset)} flights."
    )

    # Use num_workers to speed up data loading if on Linux/Mac (set to 0 on Windows usually)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 5. Initialize Model
    artifacts = processor.get_artifacts()
    model = FlightPredictor(artifacts["emb_config"], num_numerical_cols=7).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    crit_reg = nn.SmoothL1Loss()
    crit_cls = nn.BCEWithLogitsLoss()

    # 6. Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch in train_loader:
            x_cat = batch["x_cat"].to(device)
            x_num = batch["x_num"].to(device)
            y_r = batch["y_reg"].to(device).unsqueeze(1)
            y_c = batch["y_cls"].to(device).unsqueeze(1)

            preds_r, preds_c = model(x_cat, x_num)
            loss = crit_reg(preds_r, y_r) + crit_cls(preds_c, y_c)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        val_loss, val_acc = validate(model, val_loader, crit_reg, crit_cls, device)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    # 7. Save
    os.makedirs(models_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(models_dir, "model.pth"))

    with open(os.path.join(models_dir, "artifacts.pkl"), "wb") as f:
        pickle.dump(artifacts, f)

    print("Training Complete.")
