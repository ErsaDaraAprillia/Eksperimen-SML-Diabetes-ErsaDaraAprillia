import os
import joblib
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import mlflow

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# =======================================
# Folder
# =======================================

preprocessing_dir = "diabetes_preprocessing"
output_dir = "models_tuning"

os.makedirs(output_dir, exist_ok=True)

# =======================================
# Load Dataset
# =======================================

X_train = pd.read_csv(f"{preprocessing_dir}/train_processed.csv")
X_test = pd.read_csv(f"{preprocessing_dir}/test_processed.csv")

y_train = pd.read_csv(f"{preprocessing_dir}/y_train.csv").squeeze()
y_test = pd.read_csv(f"{preprocessing_dir}/y_test.csv").squeeze()

# =======================================
# MLflow
# =======================================

mlflow.set_experiment("Diabetes_Model_Tuning")

with mlflow.start_run(run_name="GridSearchCV"):

    print("Hyperparameter Tuning...")

    param_grid = {
        "n_estimators":[50,100],
        "max_depth":[None,10],
        "min_samples_split":[2,5]
    }

    rf = RandomForestClassifier(random_state=42)

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="accuracy",
        cv=3,
        n_jobs=-1
    )

    grid.fit(X_train,y_train)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)

    mlflow.log_param("best_params",grid.best_params_)

    mlflow.log_metric("accuracy",accuracy)
    mlflow.log_metric("precision",precision)
    mlflow.log_metric("recall",recall)
    mlflow.log_metric("f1_score",f1)

    joblib.dump(best_model,f"{output_dir}/best_tuned_model.joblib")
    mlflow.log_artifact(f"{output_dir}/best_tuned_model.joblib")

    cm = confusion_matrix(y_test,y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")
    plt.title("Confusion Matrix Tuned")

    plt.savefig(f"{output_dir}/confusion_matrix_tuned.png")
    plt.close()

    mlflow.log_artifact(f"{output_dir}/confusion_matrix_tuned.png")

print("Tuning selesai.")