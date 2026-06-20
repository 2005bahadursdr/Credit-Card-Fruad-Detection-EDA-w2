import pandas as pd
from xgboost import XGBClassifier
import pickle
import os

def train_xgboost(data_path="data/processed/train.csv", model_path="models/xgboost_model.pkl"):
    print("Loading training data...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"File {data_path} not found. Please run data_preprocessing.py first.")
        return

    X_train = df.drop('Class', axis=1)
    y_train = df['Class']
    
    # Calculate scale_pos_weight to handle class imbalance
    # ratio = count(negative examples) / count(positive examples)
    neg_cases = y_train.value_counts()[0]
    pos_cases = y_train.value_counts()[1]
    scale_pos_weight = neg_cases / pos_cases

    print(f"Training XGBoost with scale_pos_weight: {scale_pos_weight:.2f}...")
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='aucpr' # Area under PR curve is better for highly imbalanced data
    )
    
    model.fit(X_train, y_train)
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_xgboost()
