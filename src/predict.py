import os
import pickle

import numpy as np
import pandas as pd
import torch

from src.model import FlightPredictor


class FlightOracle:
    def __init__(self, models_dir):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA (NVIDIA GPU)")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS (Apple Silicon GPU)")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")
        # Load Artifacts
        with open(os.path.join(models_dir, "artifacts.pkl"), "rb") as f:
            self.artifacts = pickle.load(f)

        # Load Model
        self.model = FlightPredictor(self.artifacts["emb_config"], num_numerical_cols=7)
        self.model.load_state_dict(
            torch.load(os.path.join(models_dir, "model.pth"), map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, airline, origin, dest, date, time, distance):
        # 1. Preprocess Single Input
        # (We replicate logic from DataProcessor manually for single inference)
        dt = pd.to_datetime(f"{date} {time}")

        # Time Cyclic
        dep_mins = dt.hour * 60 + dt.minute
        arr_mins = (dep_mins + 120) % 1440  # Mock arrival time
        dep_sin, dep_cos = (
            np.sin(dep_mins / 1440 * 2 * np.pi),
            np.cos(dep_mins / 1440 * 2 * np.pi),
        )
        arr_sin, arr_cos = (
            np.sin(arr_mins / 1440 * 2 * np.pi),
            np.cos(arr_mins / 1440 * 2 * np.pi),
        )
        day_sin, day_cos = (
            np.sin(dt.day / 31 * 2 * np.pi),
            np.cos(dt.day / 31 * 2 * np.pi),
        )

        # Scale Distance
        dist_scaled = self.artifacts["scaler"].transform([[distance]])[0][0]

        # Categorical
        try:
            air_idx = self.artifacts["airline_encoder"].transform([airline])[0]
            orig_idx = self.artifacts["airport_encoder"].transform([origin])[0]
            dest_idx = self.artifacts["airport_encoder"].transform([dest])[0]
        except:
            print("Warning: Unknown category. Using index 0.")
            air_idx, orig_idx, dest_idx = 0, 0, 0

        # Tensors
        x_cat = torch.tensor(
            [[air_idx, orig_idx, dest_idx, dt.month - 1, dt.dayofweek]],
            dtype=torch.long,
        ).to(self.device)
        x_num = torch.tensor(
            [[dep_sin, dep_cos, arr_sin, arr_cos, day_sin, day_cos, dist_scaled]],
            dtype=torch.float32,
        ).to(self.device)

        # Predict
        with torch.no_grad():
            r, c = self.model(x_cat, x_num)

        return r.item(), torch.sigmoid(c).item()
