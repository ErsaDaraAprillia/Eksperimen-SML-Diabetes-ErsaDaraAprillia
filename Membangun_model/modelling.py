import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

preprocessing_dir = "diabetes_preprocessing"
output_dir = "models"
os.makedirs(output_dir, exist_ok=True)

# Load data
X_train = pd.read_csv(f"{preprocessing_dir}/train_processed.csv")
X_test = pd.read_csv(f"{preprocessing_dir}/test_processed.csv")
y_train = pd.read_csv(f"{preprocessing_dir}/y_train.csv").squeeze()
y_test = pd.read_csv(f"{preprocessing_dir}/y_test.csv").squeeze()

print("Training models for Diabetes dataset...")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

best_model = None
best_acc = 0
best_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name:20}: {acc:.4f}")
    
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

print(f"\n✅ Best Model: {best_name} with accuracy {best_acc:.4f}")

joblib.dump(best_model, f"{output_dir}/best_model.joblib")

# Confusion Matrix
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - {best_name}')
plt.savefig('confusion_matrix.png')
plt.close()

print("Confusion Matrix saved!")