import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset
from tqdm import tqdm  # For progress bar


class FlightDataset(Dataset):
    def __init__(self, X_cat, X_num, y_reg=None, y_cls=None):
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.y_reg = (
            torch.tensor(y_reg, dtype=torch.float32) if y_reg is not None else None
        )
        self.y_cls = (
            torch.tensor(y_cls, dtype=torch.float32) if y_cls is not None else None
        )

    def __len__(self):
        return len(self.X_cat)

    def __getitem__(self, idx):
        item = {"x_cat": self.X_cat[idx], "x_num": self.X_num[idx]}
        if self.y_reg is not None:
            item["y_reg"] = self.y_reg[idx]
            item["y_cls"] = self.y_cls[idx]
        return item


class DataProcessor:
    def __init__(self):
        self.airline_encoder = LabelEncoder()
        self.airport_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.emb_config = {}

    def _to_minutes(self, series):
        return (series // 100 * 60) + (series % 100)

    def _get_cyclic(self, series, period):
        rads = (series / period) * 2 * np.pi
        return np.sin(rads), np.cos(rads)

    def _load_and_clean_single_csv(self, filepath):
        """Helper to load one file and drop garbage immediately to save RAM"""
        try:
            df = pd.read_csv(filepath)
            # Drop Cancelled/Diverted immediately
            df = df[(df["CANCELLED"] == 0) & (df["DIVERTED"] == 0)]
            df = df.dropna(subset=["ARR_DELAY"])
            return df
        except Exception as e:
            print(f"Warning: Failed to process {filepath}: {e}")
            return pd.DataFrame()

    def process_raw_data(self, file_inputs, is_training=True):
        """
        Accepts a single string path OR a list of paths.
        Returns: (X_cat, X_num, y_reg, y_cls)
        """
        # 1. Handle Input Types (String vs List)
        if isinstance(file_inputs, str):
            filepaths = [file_inputs]
        else:
            filepaths = file_inputs

        print(f"Processing {len(filepaths)} file(s)...")

        # 2. Iterative Load & Prune
        data_frames = []
        for fp in tqdm(filepaths, desc="Loading CSVs"):
            df_part = self._load_and_clean_single_csv(fp)
            if not df_part.empty:
                data_frames.append(df_part)

        if not data_frames:
            raise ValueError("No valid data found in provided files.")

        # 3. Merge into one giant table
        df = pd.concat(data_frames, ignore_index=True)
        print(f"Total Combined Rows: {len(df)}")

        # 4. Feature Engineering (Cyclic Time)
        dep_mins = self._to_minutes(df["CRS_DEP_TIME"])
        arr_mins = self._to_minutes(df["CRS_ARR_TIME"])

        df["dep_sin"], df["dep_cos"] = self._get_cyclic(dep_mins, 1440)
        df["arr_sin"], df["arr_cos"] = self._get_cyclic(arr_mins, 1440)
        df["day_sin"], df["day_cos"] = self._get_cyclic(df["DAY_OF_MONTH"], 31)

        # 5. Categorical Encoding
        df["month_idx"] = df["MONTH"] - 1
        df["day_idx"] = df["DAY_OF_WEEK"] - 1

        if is_training:
            # Fit Encoders
            print("Fitting encoders...")
            df["airline_idx"] = self.airline_encoder.fit_transform(
                df["OP_UNIQUE_CARRIER"]
            )

            all_airports = set(df["ORIGIN"]).union(set(df["DEST"]))
            self.airport_encoder.fit(list(all_airports))

            df["dist_scaled"] = self.scaler.fit_transform(df[["DISTANCE"]])

            # Save Config
            self.emb_config = {
                "airline": (len(self.airline_encoder.classes_) + 1, 8),
                "airport": (len(self.airport_encoder.classes_) + 1, 20),
                "month": (12, 4),
                "day": (7, 4),
            }
        else:
            # Transform Only
            df["airline_idx"] = self.airline_encoder.transform(df["OP_UNIQUE_CARRIER"])
            df["dist_scaled"] = self.scaler.transform(df[["DISTANCE"]])

        # Transform Airports
        df["origin_idx"] = self.airport_encoder.transform(df["ORIGIN"])
        df["dest_idx"] = self.airport_encoder.transform(df["DEST"])

        # 6. Prepare Tensors
        print("Converting to Tensors...")
        X_cat = df[
            ["airline_idx", "origin_idx", "dest_idx", "month_idx", "day_idx"]
        ].values.astype(np.int64)
        X_num = df[
            [
                "dep_sin",
                "dep_cos",
                "arr_sin",
                "arr_cos",
                "day_sin",
                "day_cos",
                "dist_scaled",
            ]
        ].values.astype(np.float32)

        y_reg = df["ARR_DELAY"].values.astype(np.float32)
        y_cls = df["ARR_DEL15"].values.astype(np.float32)

        return X_cat, X_num, y_reg, y_cls

    def get_artifacts(self):
        return {
            "airline_encoder": self.airline_encoder,
            "airport_encoder": self.airport_encoder,
            "scaler": self.scaler,
            "emb_config": self.emb_config,
        }
