import pandas as pd

# Load dataset
df = pd.read_csv("C:\\Users\\haroo\\OneDrive\\Dokumen\\Desktop\\Dataset f1\\datasets\\final_merged_health_dataset.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Print full list of columns
print("🧾 All columns:", df.columns.tolist())

# Show dtypes
print("\n📊 Data types:")
print(df.dtypes)

# Check if 'target' column exists
if 'target' not in df.columns:
    raise ValueError("❌ 'target' column is missing!")

# Get numeric columns (excluding target)
numeric_df = df.select_dtypes(include='number')
print("\n🔢 Numeric columns (excluding 'target'):", numeric_df.drop(columns='target', errors='ignore').columns.tolist())

# Show how many rows have missing values
print("\n❗ Rows with missing values:", df.isnull().sum().sum())

# Preview first few rows
print("\n🧪 Sample rows:")
print(df.head())
