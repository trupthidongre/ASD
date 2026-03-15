"""
retrain_model.py — Run this once to generate autism_model.joblib & autism_encoders.joblib
Place this file next to app.py and clean_data1.csv, then run:  python retrain_model.py
"""
import os, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

BASE = os.path.dirname(os.path.abspath(__file__))
CAT_COLS = ["ethnicity","contry_of_res","used_app_before","relation"]

print("📂 Loading clean_data1.csv ...")
df = pd.read_csv(os.path.join(BASE,"clean_data1.csv"))
print(f"   {len(df)} rows  |  columns: {list(df.columns)}\n")

X = df.drop("Class/ASD", axis=1)
y = df["Class/ASD"]

encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le
    print(f"   Encoded '{col}' → {list(le.classes_)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n📊 Train: {len(X_train)}  |  Test: {len(X_test)}")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("🎯 Training complete!")

y_pred = model.predict(X_test)
print(f"\n📈 Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(model,    os.path.join(BASE,"autism_model.joblib"))
joblib.dump(encoders, os.path.join(BASE,"autism_encoders.joblib"))
print("\n✅ autism_model.joblib saved")
print("✅ autism_encoders.joblib saved")
print("\nRestart Flask — the model will load automatically.")
