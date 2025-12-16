import sys
import warnings

from src.predict import FlightOracle
from src.train import train_pipeline

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py [train|predict]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "train":
        # Point this to your raw CSV
        train_pipeline("data/raw/", "models/")

    elif command == "predict":
        oracle = FlightOracle("models/")
        # Example prediction
        mins, prob = oracle.predict("AA", "ABQ", "DFW", "2025-12-30", "19:00", 569)
        print(f"Predicted Delay: {mins:.1f} mins")
        print(f"Probability of Late: {prob * 100:.1f}%")

    else:
        print("Unknown command.")
