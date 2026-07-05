import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('diabetes_raw.csv')

# Preprocessing sederhana
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Encoding (jika ada categorical)
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MLflow
mlflow.set_experiment("Diabetes_Model")

with mlflow.start_run(run_name="Basic_Model"):
    mlflow.sklearn.autolog()   # <--- Ini yang wajib

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    
    joblib.dump(model, "best_model.joblib")
    mlflow.log_artifact("best_model.joblib")

print("Modelling selesai dengan autolog!")