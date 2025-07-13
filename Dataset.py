import pandas as pd

# --- Load datasets ---
diabetes = pd.read_csv("C:\\Users\\haroo\\OneDrive\\Dokumen\\Desktop\\Dataset f1\\datasets\\diabetes.csv")
heart = pd.read_csv("C:\\Users\\haroo\\OneDrive\\Dokumen\\Desktop\\Dataset f1\\datasets\\heart_cleveland_upload.csv")
kidney = pd.read_csv("C:\\Users\\haroo\\OneDrive\\Dokumen\\Desktop\\Dataset f1\\datasets\\kidney_disease.csv")

# --- DIABETES ---
diabetes['target'] = diabetes['Outcome']
diabetes['disease_type'] = 'diabetes'
diabetes = diabetes.drop(columns=['Outcome'])
diabetes = diabetes.select_dtypes(include='number')
diabetes['disease_type'] = 'diabetes'

# --- HEART ---
heart['target'] = heart['condition']
heart['disease_type'] = 'heart'
heart = heart.drop(columns=['condition'])
heart = heart.select_dtypes(include='number')
heart['disease_type'] = 'heart'

# --- KIDNEY ---
kidney.columns = kidney.columns.str.strip()
kidney['target'] = kidney['classification'].map({'ckd': 1, 'notckd': 0})
kidney = kidney.drop(columns=['id', 'classification'], errors='ignore')
kidney = kidney.apply(pd.to_numeric, errors='coerce')
kidney = kidney.select_dtypes(include='number')
kidney['disease_type'] = 'kidney'

# --- Find common numeric features across all ---
common_cols = list(set(diabetes.columns) & set(heart.columns) & set(kidney.columns))
common_cols = [col for col in common_cols if col not in ['target']]
final_cols = common_cols + ['target', 'disease_type']

# --- Filter ---
diabetes = diabetes[final_cols]
heart = heart[final_cols]
kidney = kidney[final_cols]

# --- Merge ---
merged = pd.concat([diabetes, heart, kidney], ignore_index=True)
merged = merged.dropna()

# --- Save ---
merged.to_csv("final_merged_health_dataset.csv", index=False)
print("✅ Final merged dataset saved: final_merged_health_dataset.csv")
print("🧾 Shape:", merged.shape)
print("📊 Columns:", merged.columns.tolist())
