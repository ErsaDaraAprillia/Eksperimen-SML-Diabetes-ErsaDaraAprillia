import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =====================================================
# MLflow Configuration
# =====================================================

mlflow.set_experiment("Diabetes_Model")
mlflow.sklearn.autolog()

# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv("diabetes_raw.csv")

# Pisahkan fitur dan target
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# One Hot Encoding jika ada fitur kategorikal
X = pd.get_dummies(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# Modelling
# =====================================================

with mlflow.start_run(run_name="Basic_Model"):

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    # Training
    pipeline.fit(X_train, y_train)

    # Prediction
    y_pred = pipeline.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Manual metric (boleh walaupun autolog juga mencatat)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    print("=" * 50)
    print("HASIL EVALUASI MODEL")
    print("=" * 50)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("=" * 50)

print("Training selesai.")
print("MLflow autolog berhasil dijalankan.")