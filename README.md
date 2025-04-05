# 🫁 Asthma Severity Prediction Model

This project uses a **Logistic Regression** model to predict the **severity of asthma** in patients based on various health-related features. The model classifies patients into different severity levels to support early intervention and better treatment decisions.

## 📊 Dataset  
The dataset includes anonymized medical records with features such as:

- Age  
- Gender  
- Smoking History  
- Air Pollution Index  
- Genetic Risk  
- Shortness of Breath  
- Coughing Frequency  
- Wheezing Intensity  
- Chest Tightness  
- Severity Level (Target Variable: Mild, Moderate, Severe)

> *Note: Please ensure that the dataset is cleaned and preprocessed before training.*

---

## 🛠️ Features

- Logistic Regression classification  
- Data cleaning and preprocessing  
- Feature encoding and scaling  
- Model evaluation (Accuracy, Confusion Matrix)  
- Exported trained model using `joblib`

---

## 🧪 Steps

1. **Data Preprocessing**  
   - Handle missing values  
   - Encode categorical variables  
   - Normalize/scale numerical features  

2. **Model Training**  
   - Train a Logistic Regression model  
   - Evaluate performance using test data  

3. **Model Evaluation**  
   - Accuracy Score  
   - Confusion Matrix  
   - Classification Report  

4. **Model Deployment (Optional)**  
   - Save trained model using `joblib`  
   - Ready for integration into a web app or API

---

## 📦 Requirements

Install the required libraries with:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Model

```bash
python asthma_severity_prediction.py
```

Make sure your dataset is located correctly and the script points to the right file path.

---

## 🚀 Future Improvements

- Add support for multiclass classification using other models like Random Forest or SVM  
- Deploy as a web app using Streamlit or Flask  
- Add real-time input prediction form

---

## 🤝 Contribution

Feel free to fork the repo and improve the model or UI. Contributions are welcome!

---

## 📫 Contact

**Name**: David Osei Kumi  
**Email**: [12dkumi@gmail.com](mailto:12dkumi@gmail.com)  
**GitHub**: [@dkumi12](https://github.com/dkumi12)

