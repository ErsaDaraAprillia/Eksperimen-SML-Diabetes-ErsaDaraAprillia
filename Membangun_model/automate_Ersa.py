import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def automate_preprocessing(output_dir="diabetes_preprocessing"):
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv("diabetes_raw.csv")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Kolom: {df.columns.tolist()}")
    
    # Encoding kolom kategorikal
    categorical_cols = ['gender', 'smoking_history']
    encoders = {}
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
    
    # Target
    X = df.drop('diabetes', axis=1)
    y = df['diabetes']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Simpan
    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(f"{output_dir}/train_processed.csv", index=False)
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(f"{output_dir}/test_processed.csv", index=False)
    pd.DataFrame(y_train).to_csv(f"{output_dir}/y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv(f"{output_dir}/y_test.csv", index=False)
    
    print(f"✅ Preprocessing Diabetes selesai! File disimpan di {output_dir}")

if __name__ == "__main__":
    automate_preprocessing()