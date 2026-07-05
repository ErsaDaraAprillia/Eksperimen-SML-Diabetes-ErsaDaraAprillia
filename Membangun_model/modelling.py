import pandas as pd
import mlflow
import mlflow.sklearn
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json

# ====================== MLflow Config ======================
mlflow.set_experiment("Diabetes_Prediction")

# Nonaktifkan autolog agar kita manual logging (lebih baik untuk Skilled/Advance)
# mlflow.sklearn.autolog()  

with mlflow.start_run(run_name="RandomForest_Tuned") as run:
    
    # Load data
    df = pd.read_csv("diabetes_raw.csv")   # pastikan path benar
    
    X = df.drop("diabetes", axis=1)
    y = df["diabetes"]
    
    X = pd.get_dummies(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # ====================== Manual Logging ======================
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    # Log model
    mlflow.sklearn.log_model(pipeline, "model")
    
    # Simpan confusion matrix sebagai artifact
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    
    print(f"✅ Run ID: {run.info.run_id}")
    print(f"Accuracy: {accuracy:.4f}")