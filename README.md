# Autism Predictor with Machine Learning


## 🚀 Overview
This project is a **Machine Learning-based Autism Predictor** designed to assess the likelihood of Autism Spectrum Disorder (ASD) based on user responses and optional facial analysis. It employs **XGBoost Classifier with GridSearchCV** for hyperparameter tuning and **SMOTE** for handling imbalanced datasets. The application is built with **Streamlit**, providing an interactive and user-friendly interface for ASD screening.


## 💻Demo Video

https://acesse.one/CLICK-TO-SEE-THE-DEMO-VIDEO


## 🌟 Features
✅ **ASD Screening Questions** – A structured questionnaire assessing behavioral traits associated with ASD.

✅ **Facial Analysis (Optional)** – Uses image processing techniques to detect facial features that may contribute to ASD prediction.

✅ **Machine Learning Model** – Incorporates an optimized **XGBoost Classifier** for accurate predictions.

✅ **Data Handling** – Uses **SMOTE** to balance class distribution and improve model performance.

✅ **Real-time Visualization** – Displays dataset insights, including ASD class distribution and prediction results.

✅ **Streamlit Web App** – A fully interactive and easy-to-use interface with live model predictions.

## 🛠️ Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/autism-predictor.git
   cd autism-predictor
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## 📊 Dataset
- The model is trained on an **Autism Spectrum Disorder (ASD) Screening dataset** (e.g., `train.csv`).
- The dataset includes key features such as **age, gender, ethnicity, screening test responses, and ASD diagnosis**.
- Data preprocessing includes handling missing values, dropping redundant columns, and encoding categorical features.

## 🧠 Model & Training
- **Preprocessing:** The dataset is cleaned by removing irrelevant features and encoding categorical variables.
- **Oversampling:** **SMOTE (Synthetic Minority Over-sampling Technique)** is used to address class imbalance.
- **Classifier:** The model leverages **XGBoost**, known for its high efficiency and performance in classification tasks.
- **Hyperparameter Tuning:** **GridSearchCV** optimizes parameters such as learning rate, max depth, and number of estimators.
- **Evaluation:** The model reports the best accuracy score after training and tuning.

## 🎯 Usage
1. **Answer the ASD screening questionnaire by selecting responses to behavioral questions.**
2. **(Optional) Upload a facial image for analysis.**
3. **Click the 'Predict Autism (ASD)' button.**
4. **The app calculates a final ASD likelihood score using questionnaire responses and facial analysis.**
5. **Results are displayed with recommendations for further screening if necessary.**

## 🖼️ Example

https://acesse.one/CLICK-TO-OPEN-THE-RESULTS

## 🤝 Contributing
We welcome contributions from the community! Follow these steps to contribute:
1. **Fork the repository.**
2. **Create a new branch:**
   ```bash
   git checkout -b feature-branch
   ```
3. **Make your changes and commit them:**
   ```bash
   git commit -m "Add new feature"
   ```
4. **Push to the branch and create a Pull Request.**

## 🚀 Future Enhancements
🔹 Improve facial analysis by integrating **deep learning models for feature extraction**.

🔹 Expand the dataset to include **more diverse populations** for better generalization.

🔹 Enhance the web app with **real-time feedback and report generation**.

🔹 Deploy the application using **cloud services** for wider accessibility.



## 📧 Contact
For any queries, feel free to open an issue on **[GitHub Issues](https://github.com/yourusername/autism-predictor/issues)** or reach out via email.

