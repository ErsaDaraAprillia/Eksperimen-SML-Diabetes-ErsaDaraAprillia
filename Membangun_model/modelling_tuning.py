import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

preprocessing_dir = "diabetes_preprocessing"
output_dir = "models_tuning"
os.makedirs(output_dir, exist_ok=True)

X_train = pd.read_csv(f"{preprocessing_dir}/train_processed.csv")
X_test = pd.read_csv(f"{preprocessing_dir}/test_processed.csv")
y_train = pd.read_csv(f"{preprocessing_dir}/y_train.csv").squeeze()
y_test = pd.read_csv(f"{preprocessing_dir}/y_test.csv").squeeze()

print("Hyperparameter Tuning (versi ringan)...")

# Parameter lebih sedikit biar cepat
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)  # cv=3 lebih cepat
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Best Accuracy: {acc:.4f}")
print(f"Best Params: {grid_search.best_params_}")

joblib.dump(best_model, f"{output_dir}/best_tuned_model.joblib")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - Tuned Model')
plt.savefig('confusion_matrix_tuned.png')
plt.close()

print("Tuning selesai!")