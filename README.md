# AI Healthcare Multi-Disease Predictor

This project is a machine learning-powered web application that predicts possible diseases from patient data. It supports both CSV uploads and manual form inputs, with optional image upload capability (for future extension). Built using Scikit-learn and Streamlit, the model uses a Random Forest Classifier trained on a merged dataset of diabetes, heart, and kidney disease data.

# Features

- Predicts diseases using patient clinical data
- CSV upload and manual input form
- Model trained on merged dataset of 3 diseases
- Handles missing values with preprocessing
- Built-in feature scaling and model saving
- Ready for future SHAP explainability and image scan analysis

# How to Run the App

# 1. Clone the Repo

git clone https://github.com/haroonbashi2514/AI-Disease-Predictor
cd ai-healthcare-predictor

# 2. Create Virtual Environment

python -m venv venv
venv\Scripts\activate  # On Windows

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Launch Streamlit App
streamlit run streamlit_app.py

# Model Info
- Algorithm: `RandomForestClassifier`
- Evaluation: Accuracy, Precision, F1-Score (can be added in `main.py`)
- Scaler: `StandardScaler`
- Output: Model (`model.pkl`) and scaler (`scaler.pkl`) saved for inference

# Future Enhancements
- SHAP Explainability plots
- Image-based OCR upload (X-ray reports, medical scans)
- Visual dashboards for predictions
- Add more diseases and specialized models
- Deploy to Streamlit Cloud / HuggingFace Spaces

#  Author
Haroon Bashi 
AI/ML Developer | Python Expert  
[LinkedIn](http://www.linkedin.com/in/haroon-bashi/) | haroonbashi2514@gmail.com

⚠️ Note: The predictions are for educational/research use only. Always consult a medical professional.
