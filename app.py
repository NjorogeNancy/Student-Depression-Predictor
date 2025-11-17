import streamlit as st
import pickle
import pandas as pd
import numpy as np



#               LOAD TRAINED MODELS & PREPROCESSOR


# Load your saved models
logistic_model = pickle.load(open("logistic_regression.pkl", "rb"))
knn_model = pickle.load(open("knn.pkl", "rb"))
rf_model = pickle.load(open("random_forest.pkl", "rb"))
svm_model = pickle.load(open("svm.pkl", "rb"))

# Store models in a dictionary for easy selection
models = {
    "Logistic Regression": logistic_model,
    "KNN": knn_model,
    "Random Forest": rf_model,
    "SVM": svm_model
}

# Load preprocessing pipeline (ColumnTransformer)
preprocessor = pickle.load(open("preprocessor.pkl", "rb"))



#                        STREAMLIT UI


st.title("Student Depression Prediction App")
st.write("This app predicts **whether a student may be depressed** based on lifestyle & academic factors.")



#                 MODEL SELECTION DROPDOWN

model_choice = st.selectbox("Choose a model:", list(models.keys()))
model = models[model_choice]



#                        USER INPUTS

st.subheader("Enter Student Information")

gender = st.selectbox("Gender", ["Male", "Female"])

age = st.number_input("Age", min_value=5, max_value=100)

city = st.selectbox("City", [
    "Srinagar","Thane","Kalyan","Kolkata","Lucknow",
    "Surat","Ludhiana","Kanpur","Nagpur","Indore","Agra"
])

profession = st.selectbox("Profession", [
    "Student","Working Professional","Unemployed","Others"
])

academic_pressure = st.number_input("Academic Pressure (1–5)", min_value=1, max_value=5)
work_pressure = st.number_input("Work Pressure (0–5)", min_value=0, max_value=5)

cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, format="%.2f")

study_satisfaction = st.number_input("Study Satisfaction (1–5)", min_value=1, max_value=5)
job_satisfaction = st.number_input("Job Satisfaction (1–5)", min_value=1, max_value=5)

sleep_duration = st.selectbox("Sleep Duration", [
    "Less Than 5 Hours", "5-6 Hours", "7-8 Hours", "More Than 8 Hours", "Others"
])

dietary_habits = st.selectbox("Dietary Habits", [
    "Healthy", "Unhealthy", "Moderate", "Others"
])

degree = st.selectbox("Degree", [
    "Bca","M.Tech","Class 12","B.Ed","Msc","Bhm","B.Com","B.Arch"
])

suicidal_thoughts = st.selectbox(
    "Have you ever had suicidal thoughts ?",
    ["Yes", "No"]
)

work_study_hours = st.number_input("Work/Study Hours (0–24)", min_value=0, max_value=24)

financial_stress = st.selectbox(
    "Financial Stress",
    ["1","2","3","4","5","?"]  # this column is categorical in training
)

family_history = st.selectbox(
    "Family History of Mental Illness (0 = No, 1 = Yes)",
    [0, 1]
)



#                 CREATE INPUT DATAFRAME

input_data = pd.DataFrame({
    "Gender": [gender],
    "Age": [age],
    "City": [city],
    "Profession": [profession],
    "Academic Pressure": [academic_pressure],
    "Work Pressure": [work_pressure],
    "CGPA": [cgpa],
    "Study Satisfaction": [study_satisfaction],
    "Job Satisfaction": [job_satisfaction],
    "Sleep Duration": [sleep_duration],
    "Dietary Habits": [dietary_habits],
    "Degree": [degree],
    "Have you ever had suicidal thoughts ?": [suicidal_thoughts],
    "Work/Study Hours": [work_study_hours],
    "Financial Stress": [financial_stress],
    "Family History of Mental Illness": [family_history]
})

# Debug visibility (optional)
# st.write("DEBUG Input Data:", input_data)
# st.write("DEBUG dtypes:", input_data.dtypes)



#               FIX COLUMN TYPES (only numeric)


numeric_cols = [
    "Age",
    "Academic Pressure",
    "Work Pressure",
    "CGPA",
    "Study Satisfaction",
    "Job Satisfaction",
    "Work/Study Hours",
    "Family History of Mental Illness"
]

# Convert numeric columns properly
for col in numeric_cols:
    input_data[col] = pd.to_numeric(input_data[col], errors="ignore")


# Convert suicidal thoughts to numeric (as in training)
input_data["Have you ever had suicidal thoughts ?"] = (
    input_data["Have you ever had suicidal thoughts ?"].map({"Yes": 1, "No": 0})
)



#                 PREPROCESS & PREDICT


if st.button("Predict"):
    try:
        # Transform input using saved ColumnTransformer
        processed_input = preprocessor.transform(input_data)

        # Make prediction
        prediction = model.predict(processed_input)[0]

        result = "Depressed" if prediction == 1 else "Not Depressed"

        st.success(f" Prediction: **{result}**")

    except Exception as e:
        st.error("ERROR during prediction. Details below:")
        st.error(str(e))
