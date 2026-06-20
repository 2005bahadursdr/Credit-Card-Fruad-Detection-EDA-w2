import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import os

def load_and_preprocess_data(filepath="data/raw/creditcard.csv"):
    """
    Loads raw data, scales Time and Amount, and splits into train/test sets.
    """
    print("Loading dataset...")
    df = pd.read_csv(filepath)

    print("Scaling Time and Amount using RobustScaler...")
    # We use RobustScaler since it is less prone to outliers
    scaler = RobustScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

    # Drop old Time and Amount
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    
    # Reorder columns to put scaled features first
    scaled_amount = df['scaled_amount']
    scaled_time = df['scaled_time']
    df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
    df.insert(0, 'scaled_amount', scaled_amount)
    df.insert(1, 'scaled_time', scaled_time)

    # Save processed base dataset
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/scaled_data.csv', index=False)

    print("Splitting dataset...")
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Stratified split is crucial for highly imbalanced data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Save train and test sets
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv('data/processed/train.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    
    print("Preprocessing complete. Data saved to data/processed/")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    load_and_preprocess_data()
