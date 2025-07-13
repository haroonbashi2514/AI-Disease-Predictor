import streamlit as st
import pandas as pd
import joblib

# Set Streamlit page config
st.set_page_config(page_title="Disease Prediction App", layout="centered")

# Title
st.title("🩺 Disease Prediction App")
st.write("Upload a patient's medical data to predict the likelihood of a disease.")

# Upload CSV
uploaded_file = st.file_uploader("C:\\Users\\haroo\\OneDrive\\Dokumen\\Desktop\\Dataset f1\\patient_data.csv", type=["csv"])

if uploaded_file:
    # Read the uploaded CSV
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")
    st.write(df.head())

    try:
        # Load trained model
        model = joblib.load("C:\\Users\\haroo\\OneDrive\\Dokumen\\Desktop\\Dataset f1\\model.pkl")

        # Manually specify the features used during training
        feature_columns = [
            'Age', 'BMI', 'BloodPressure', 'DiabetesPedigreeFunction', 'Glucose',
            'Insulin', 'Pregnancies', 'SkinThickness', 'age', 'al', 'ane', 'appet',
            'ba', 'bgr', 'bp', 'bu', 'ca', 'cad', 'chol', 'cp', 'dm', 'exang', 'fbs',
            'hemo', 'htn', 'oldpeak', 'pc', 'pcc', 'pcv', 'pe', 'pot', 'rbc', 'rc',
            'restecg', 'sc', 'sex', 'sg', 'slope', 'sod', 'su', 'thal', 'thalach',
            'trestbps', 'wc'
        ]

        # Align data with features
        input_data = df[feature_columns]

        # Make predictions
        predictions = model.predict(input_data)
        prediction_probs = model.predict_proba(input_data)

        # Map prediction to disease names
        disease_mapping = {
            0: "No Disease",
            1: "Diabetes"  # Update this if you have other disease labels
        }

        df['Predicted Disease'] = [disease_mapping.get(pred, "Unknown") for pred in predictions]
        df['Confidence (%)'] = prediction_probs.max(axis=1) * 100

        # Display results
        st.success("✅ Predictions completed!")
        st.subheader("🔍 Prediction Results")
        st.dataframe(df)

        # Download results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results as CSV", csv, "prediction_results.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
else:
    st.warning("📌 Please upload a CSV file to continue.")

