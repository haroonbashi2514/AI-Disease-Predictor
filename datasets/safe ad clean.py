import pandas as pd

kidney = pd.read_csv("C:\\Users\\haroo\\OneDrive\\Dokumen\\Desktop\\Dataset f1\\datasets\\kidney_disease.csv")

# 🔧 Remove any whitespace from column names
kidney.columns = kidney.columns.str.strip()

# ✅ Map target safely
kidney['target'] = kidney['classification'].map({'ckd': 1, 'notckd': 0})

# 💣 Check if mapping worked
if kidney['target'].isnull().all():
    raise ValueError("❌ Mapping failed — check if 'classification' contains unexpected values.")

# Drop ID and classification after use
kidney = kidney.drop(columns=['id', 'classification'], errors='ignore')

# Try to convert all columns to numeric (ignore errors)
kidney = kidney.apply(pd.to_numeric, errors='coerce')

print("✅ Kidney dataset cleaned. Shape:", kidney.shape)
