# 🧠 Autism Spectrum Disorder (ASD) Screening Tool

An AI-powered Autism Spectrum Disorder (ASD) screening system built using **XGBoost Machine Learning**, **AQ-10 clinical screening**, and **facial behavioural analysis**.

This project combines predictive analytics, healthcare-focused machine learning, and explainable AI techniques to estimate ASD likelihood based on behavioural screening responses and optional facial cues.

---

## 🚀 Features

* ✅ AQ-10 based ASD screening questionnaire
* ✅ XGBoost predictive analytics model
* ✅ Separate train/test dataset evaluation
* ✅ Facial behavioural analysis using OpenCV
* ✅ Explainable AI metrics and feature importance
* ✅ ROC-AUC, F1-score, Recall, Specificity evaluation
* ✅ Streamlit interactive web interface
* ✅ Clinical probability calibration
* ✅ Confusion matrix and ROC curve visualization

---

## 🧠 Machine Learning Pipeline

### Data Preprocessing

* Missing value handling
* Dynamic AQ-10 column mapping
* One-hot encoding
* Feature alignment between train/test datasets
* SMOTE balancing for minority ASD class

### Model

* XGBoost Classifier
* Regularization to reduce overfitting
* Probability threshold calibration
* Healthcare-oriented sensitivity optimization

---

## 📊 Model Performance

| Metric               | Score  |
| -------------------- | ------ |
| Accuracy             | 96.91% |
| Balanced Accuracy    | 97.22% |
| Recall / Sensitivity | 98.04% |
| Specificity          | 96.40% |
| Precision            | 92.59% |
| F1 Score             | 0.952  |
| ROC-AUC              | 0.999  |
| Brier Score          | 0.021  |

---

## 🩺 Clinical Screening Logic

The system uses:

* AQ-10 behavioural screening
* supplementary behavioural indicators
* optional facial gaze analysis
* calibrated predictive probabilities

This tool is intended for **screening support only** and is **not a medical diagnosis system**.

---

## 🖼️ Facial Analysis

Optional facial analysis is performed using:

* Haar Cascade face detection
* eye visibility analysis
* behavioural gaze indicators

Facial analysis contributes only a small calibrated weight to the final prediction.

---

## 📂 Dataset Sources

This project was trained using clinically inspired AQ-10 ASD screening datasets from the UCI Machine Learning Repository:

- [Adult ASD Screening Dataset](https://archive.ics.uci.edu/dataset/426/autism+screening+adult)

- [Adolescent ASD Screening Dataset](https://archive.ics.uci.edu/dataset/420/autistic+spectrum+disorder+screening+data+for+adolescent)

---
## 🚀 Live Demo

🔗 https://asdprognosis.streamlit.app/

---

## 🎥 Streamlit Profile

🔗 https://share.streamlit.io/user/akshita-2006

---

## 🛠️ Tech Stack

* Python
* Streamlit
* XGBoost
* Scikit-learn
* OpenCV
* Pandas
* NumPy
* Matplotlib

---

## ▶️ Run Locally

```bash
# Clone the repository
git clone https://github.com/Akshita-2006/ASDPrognosis.git

# Go to project folder
cd ASDPrognosis

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run autism_predictor.py
```
---

## ⚠️ Disclaimer

This project is intended for educational and screening purposes only.

It does not replace professional medical evaluation, diagnosis, or treatment.
