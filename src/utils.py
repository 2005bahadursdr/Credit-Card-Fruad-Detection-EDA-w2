import os
import pickle
import pandas as pd

def load_data(filepath):
    """Utility to load a CSV dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)

def save_model(model, filepath):
    """Utility to save a trained model."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model successfully saved to {filepath}")

def load_model(filepath):
    """Utility to load a trained model."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model
