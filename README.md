# 💳 Credit Score Prediction Model

This machine learning project uses a **Random Forest Classifier** to predict whether an individual is likely to have a **good or bad credit score**, based on their demographic and financial data.

It was developed as part of my learning experience with **CodeAlpha**.

---

## 📊 Dataset

**Features Include:**

- Age  
- Gender  
- Education  
- Marital Status  
- Number of Children  
- Income  
- Home Ownership  
- Employment Status  
- Credit Score (Target: Good/Bad)

---

## 🔍 Project Workflow

1. **Data Preprocessing**  
   - Handling missing values  
   - Encoding categorical variables  
   - Feature scaling

2. **Model Training**  
   - Random Forest Classifier  
   - GridSearchCV for hyperparameter tuning

3. **Model Evaluation**  
   - Accuracy, Confusion Matrix, Precision, Recall

4. **Model Export**  
   - Trained model saved using `joblib`

---

## 🛠️ Requirements

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Model

```bash
python src/credit_scoring_model.py
```

Ensure the dataset is placed in the `data/` directory and paths are correctly referenced in the script.

---

## 🚀 Future Improvements

- Integration into a loan application dashboard  
- Add visualization of feature importances  
- Deploy as an interactive web tool with Streamlit or Flask

---

## 📫 Contact

**Name**: David Osei Kumi  
**Email**: [12dkumi@gmail.com](mailto:12dkumi@gmail.com)  
**GitHub**: [@dkumi12](https://github.com/dkumi12)
