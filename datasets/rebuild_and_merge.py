import pandas as pd

# -------- Load datasets --------
diabetes = pd.read_csv("C:\\Users\\haroo\\OneDrive\\Dokumen\\Desktop\\Dataset f1\\datasets\\diabetes.csv")
heart = pd.read_csv("C:\\Users\\haroo\\OneDrive\\Dokumen\\Desktop\\Dataset f1\\datasets\\heart_cleveland_upload.csv")
kidney = pd.read_csv("C:\\Users\\haroo\\OneDrive\\Dokumen\\Desktop\\Dataset f1\\datasets\kidney_disease.csv")

# -------- Clean diabetes --------
diabetes['target'] = diabetes['Outcome']
diabetes['disease_type'] = 'diabetes'
diabetes = diabetes.drop(columns=['Outcome'])

# -------- Clean heart --------
heart['target'] = heart['condition']
heart['disease_type'] = 'heart'
heart = heart.drop(columns=['condition'])

# -------- Clean kidney --------
kidney.columns = kidney.columns.str.strip()
kidney['target'] = kidney['classification'].map({'ckd': 1, 'notckd': 0})
kidney['disease_type'] = 'kidney'
kidney = kidney.drop(columns=['id', 'classification'], errors='ignore')
kidney = kidney.apply(pd.to_numeric, errors='coerce')  # convert all to numbers

# -------- Fill missing values --------
diabetes = diabetes.apply(pd.to_numeric, errors='coerce').fillna(diabetes.mean(numeric_only=True))
heart = heart.apply(pd.to_numeric, errors='coerce').fillna(heart.mean(numeric_only=True))
kidney = kidney.fillna(kidney.mean(numeric_only=True))

# -------- Add missing columns manually to align --------
all_columns = set(diabetes.columns) | set(heart.columns) | set(kidney.columns)

for df in [diabetes, heart, kidney]:
    for col in all_columns:
        if col not in df.columns:
            df[col] = pd.NA

# -------- Reorder columns the same for merge --------
diabetes = diabetes[sorted(all_columns)]
heart = heart[sorted(all_columns)]
kidney = kidney[sorted(all_columns)]

# -------- Merge all datasets --------
merged = pd.concat([diabetes, heart, kidney], ignore_index=True)

# -------- Final clean --------
merged = merged.dropna(subset=['target'])  # ensure no NaN target
merged = merged.fillna(merged.mean(numeric_only=True))  # fill remaining

# -------- Save final dataset --------
merged.to_csv("final_merged_health_dataset.csv", index=False)
print("✅ Merged dataset saved as 'final_merged_health_dataset.csv'")
print("📊 Shape:", merged.shape)
