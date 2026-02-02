import pickle
import os
from datetime import datetime

MODEL_DIR = "models/saved"

def save_model(model, model_name):
    os.makedirs(MODEL_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.pkl"
    path = os.path.join(MODEL_DIR, filename)

    with open(path, "wb") as f:
        pickle.dump(model, f)

    return {
        "model_name": model_name,
        "file_path": path,
        "filename": filename
    }

def load_model(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)
