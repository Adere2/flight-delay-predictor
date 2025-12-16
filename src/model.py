# src/model.py
import torch
import torch.nn as nn


class FlightPredictor(nn.Module):
    def __init__(self, emb_config, num_numerical_cols):
        super(FlightPredictor, self).__init__()

        # Embeddings
        self.emb_airline = nn.Embedding(
            emb_config["airline"][0], emb_config["airline"][1]
        )
        self.emb_origin = nn.Embedding(
            emb_config["airport"][0], emb_config["airport"][1]
        )
        self.emb_dest = nn.Embedding(emb_config["airport"][0], emb_config["airport"][1])
        self.emb_month = nn.Embedding(emb_config["month"][0], emb_config["month"][1])
        self.emb_day = nn.Embedding(emb_config["day"][0], emb_config["day"][1])

        # Calculate Input Size
        # airline_emb + origin_emb + dest_emb + month_emb + day_emb + numerical_cols
        total_input_size = (
            emb_config["airline"][1]
            + emb_config["airport"][1]
            + emb_config["airport"][1]
            + emb_config["month"][1]
            + emb_config["day"][1]
            + num_numerical_cols
        )

        # The Body
        self.body = nn.Sequential(
            nn.Linear(total_input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
        )

        # The Heads
        self.reg_head = nn.Linear(64, 1)
        self.cls_head = nn.Linear(64, 1)

    def forward(self, x_cat, x_num):
        e1 = self.emb_airline(x_cat[:, 0])
        e2 = self.emb_origin(x_cat[:, 1])
        e3 = self.emb_dest(x_cat[:, 2])
        e4 = self.emb_month(x_cat[:, 3])
        e5 = self.emb_day(x_cat[:, 4])

        x = torch.cat([e1, e2, e3, e4, e5, x_num], dim=1)
        x = self.body(x)

        return self.reg_head(x), self.cls_head(x)
