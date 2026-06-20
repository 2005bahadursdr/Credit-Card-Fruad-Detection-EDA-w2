import pandas as pd
import pickle
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_models(test_data_path="data/processed/test.csv", 
                    xgb_path="models/xgboost_model.pkl",
                    if_path="models/isolation_forest.pkl"):
    print("Loading test data...")
    try:
        df = pd.read_csv(test_data_path)
    except FileNotFoundError:
        print(f"File {test_data_path} not found.")
        return

    X_test = df.drop('Class', axis=1)
    y_test = df['Class']

    # Evaluate XGBoost
    print("\n--- Evaluating XGBoost ---")
    try:
        with open(xgb_path, 'rb') as f:
            xgb_model = pickle.load(f)
            
        xgb_preds = xgb_model.predict(X_test)
        xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
        
        print(classification_report(y_test, xgb_preds))
        
        precision, recall, _ = precision_recall_curve(y_test, xgb_probs)
        pr_auc = auc(recall, precision)
        print(f"XGBoost PR AUC: {pr_auc:.4f}")
        
    except FileNotFoundError:
        print(f"Model file {xgb_path} not found.")

    # Evaluate Isolation Forest
    print("\n--- Evaluating Isolation Forest ---")
    try:
        with open(if_path, 'rb') as f:
            if_model = pickle.load(f)
            
        # Isolation forest returns 1 for normal, -1 for anomaly
        # We need to map it to 0 for normal, 1 for anomaly to match our 'Class'
        if_preds = if_model.predict(X_test)
        if_preds_mapped = [1 if x == -1 else 0 for x in if_preds]
        
        print(classification_report(y_test, if_preds_mapped))
    except FileNotFoundError:
        print(f"Model file {if_path} not found.")

if __name__ == "__main__":
    evaluate_models()
