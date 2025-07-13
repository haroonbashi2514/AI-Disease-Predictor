import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -------------------- Load and Clean Dataset --------------------
df = pd.read_csv("C:\\Users\\haroo\\OneDrive\\Dokumen\\Desktop\\Dataset f1\\datasets\\final_merged_health_dataset.csv")

# Strip column names
df.columns = df.columns.str.strip()

# Drop if target column is missing
df = df.dropna(subset=['target'])

# Drop non-numeric string-based columns like disease_type
df = df.drop(columns=['disease_type'], errors='ignore')

# Separate X and y
X = df.drop(columns='target', errors='ignore')
y = df['target']

# Keep only numeric columns
X = X.select_dtypes(include='number')

# Fill missing values with column-wise mean
combined = pd.concat([X, y], axis=1)
combined.fillna(combined.mean(numeric_only=True), inplace=True)

# Final assignment
X = combined.drop(columns='target')
y = combined['target']

# ✅ Convert target to integer class labels
y = y.round().astype(int)


# Final check
if X.shape[0] == 0:
    raise ValueError("❌ Dataset is still empty after filling. Check input CSV.")

print(f"✅ After fillna → X: {X.shape}, y: {y.shape}")

# -------------------- Train/Test Split --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------- Scaling --------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------- Model Training --------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# -------------------- Evaluation --------------------
y_pred = model.predict(X_test_scaled)

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# -------------------- Confusion Matrix --------------------
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix 🧠")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
# Save model and scaler after training
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model and scaler saved successfully.")
# -------------------- Save Predictions --------------------
results = X_test.copy()
results['Actual'] = y_test.values
results['Predicted'] = y_pred
results.to_csv("prediction_output.csv", index=False)
print("📁 Predictions saved to prediction_output.csv")
