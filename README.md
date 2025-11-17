
---

# **Student Depression Prediction App**

A **machine learning–powered Streamlit web application** that predicts whether a student may be experiencing signs of depression based on lifestyle, academic pressure, sleep patterns, diet, work/study habits, and other personal factors.

This project integrates multiple ML models — Logistic Regression, KNN, Random Forest, and SVM — allowing users to compare predictions from different algorithms.
All preprocessing is handled through a serialized `preprocessor.pkl` pipeline to ensure consistent input transformation.

---

## **Features**

* Predict depression probability using:

  * Logistic Regression
  * K-Nearest Neighbors (KNN)
  * Random Forest
  * Support Vector Machine (SVM)
  * User-friendly Streamlit interface
  * Automated preprocessing (OneHotEncoding + StandardScaling)
  *  Modular design with separate ML model files
  * Debug-friendly (shows input structure before prediction)
  *  Handles categorical & numerical features consistently
  * Ready for local use or deployment (Streamlit Cloud, Render, etc.)

---

## **Project Structure**

```
├── app.py                     # Main Streamlit application
├── preprocessor.pkl           # Preprocessing pipeline (encoder + scaler)
├── logistic_regression.pkl    # Trained Logistic Regression model
├── knn.pkl                    # Trained KNN model
├── random_forest.pkl          # Trained Random Forest model
├── svm.pkl                    # Trained SVM model
├── requirements.txt           # Dependencies for running the app
└── README.md                  # Project documentation
```

---

## **Technologies Used**

* **Python 3.x**
* **Streamlit**
* **Scikit-learn**
* **Pandas**
* **NumPy**
* **Pickle**
* **Streamlit**

---

## **Running the App Locally**

### **1. Clone the repository**

```bash
git clone https://github.com/<NjorogeNancy>/<student_depression_predictor>.git
cd <student_depression_predictor>
```

### **2. Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows
```

### **3. Install dependencies**

```bash
pip install -r requirements.txt
```

### **4. Run the Streamlit app**

```bash
streamlit run app.py
```

---

## **Deployment**

This project can easily be deployed on:

* **Streamlit Cloud**


If you want, I can generate the deployment steps for any of these.

---

## **Model Input Features**

The app uses the following features:

### **Categorical Features**

* Gender
* City
* Profession
* Sleep Duration
* Dietary Habits
* Degree
* Financial Stress

### **Numerical Features**

* Age
* Academic Pressure
* Work Pressure
* CGPA
* Study Satisfaction
* Job Satisfaction
* Suicidal Thoughts History
* Work/Study Hours
* Family History of Mental Illness

---

## **Model Training Overview**

* Dataset: Student Depression Dataset
* Preprocessing:

  * OneHotEncoding (drop='first')
  * StandardScaler for numerical inputs
* Train-test split: 80/20
* Evaluation metrics: Accuracy, Precision, Recall, Confusion Matrix

---

**Contributing**

Contributions are welcome!
Feel free to submit issues, request features, or open pull requests.


