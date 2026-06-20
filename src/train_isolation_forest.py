import pandas as pd
from sklearn.ensemble import IsolationForest
import pickle
import os

def train_isolation_forest(data_path="data/processed/train.csv", model_path="models/isolation_forest.pkl"):
    print("Loading training data...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"File {data_path} not found. Please run data_preprocessing.py first.")
        return

    # Isolation forest is unsupervised, so we don't strictly need 'Class' for training,
    # but we drop it from features.
    X_train = df.drop('Class', axis=1)
    
    print("Training Isolation Forest...")
    # Assume ~0.0017 fraction of outliers based on general knowledge of the dataset
    model = IsolationForest(n_estimators=100, max_samples=len(X_train), 
                            contamination=0.0017, random_state=42, n_jobs=-1)
    
    model.fit(X_train)
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_isolation_forest()
