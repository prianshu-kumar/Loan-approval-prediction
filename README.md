# 🏦 Loan Approval Prediction using Machine Learning

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_App-red?logo=streamlit)](https://loan-approval-prediction-prianshu.streamlit.app/)

A supervised machine learning project that predicts whether a loan application will be **Approved or Rejected** using a complete **end-to-end ML pipeline** with preprocessing, model training, and hyperparameter tuning.

This project is built using industry-standard practices like:
- Pipeline architecture (no data leakage)
- ColumnTransformer for preprocessing
- Random Forest Classifier
- Hyperparameter tuning (GridSearchCV)
- Model persistence for deployment

---

# 🚀 Features

- Loan approval classification (binary classification)
- End-to-end ML Pipeline using Scikit-learn
- Automatic handling of missing values
- OneHotEncoding for categorical variables
- Random Forest Classifier
- Hyperparameter tuning using GridSearchCV
- Model evaluation using multiple metrics
- Production-ready model saving

---

# 📁 Project Structure

```bash
loan-approval-prediction/
│
├── notebook/
│   └── main.ipynb
│
├── model/
│   └── model.pkl
│
├── data/
│   └── loan.csv
│
├── app/
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

# 📊 Dataset Information

This project uses a loan prediction dataset containing applicant details such as:

- Gender
- Marital Status
- Education
- Income
- Loan Amount
- Credit History
- Property Area

Target variable:
- `Loan_Status` (Y/N)

---

# 🧠 Machine Learning Concepts Used

- Supervised Learning (Classification)
- Decision Trees & Random Forest
- Feature Engineering
- Missing Value Imputation
- One-Hot Encoding
- ColumnTransformer
- Pipeline (ML Workflow Automation)
- Hyperparameter Tuning (GridSearchCV)
- Model Evaluation Metrics

---

# ⚙️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

# 🔁 Machine Learning Pipeline

```text
Raw Data
↓
ColumnTransformer
↓
Missing Value Imputation
↓
One-Hot Encoding
↓
Random Forest Model
↓
Hyperparameter Tuning (GridSearchCV)
↓
Evaluation
↓
Saved Model
```

---

# 🌲 Model Used

## Random Forest Classifier

Why Random Forest?

- Handles categorical + numerical data well
- Reduces overfitting compared to Decision Tree
- Works well on tabular data
- Provides feature importance

---

# 🔥 Hyperparameter Tuning

Hyperparameter tuning was performed using **GridSearchCV** to improve model performance.

### Parameters tuned:
- n_estimators
- max_depth
- min_samples_split
- min_samples_leaf

### Benefit:
- Improved generalization
- Reduced overfitting
- Better validation accuracy

---

# 📈 Evaluation Metrics

The model is evaluated using:

- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

# 💾 Model Saving

The final trained pipeline is saved using pickle:

```python
import pickle

pickle.dump(model, open("model/model.pkl", "wb"))
```

This allows direct deployment without re-training.

---

# 🚀 How to Run Locally

## 1. Clone Repository
```bash
git clone https://github.com/prianshu-kumar/loan-approval-prediction.git
```

## 2. Move into directory
```bash
cd loan-approval-prediction
```

## 3. Install dependencies
```bash
pip install -r requirements.txt
```

## 4. Run notebook
```bash
jupyter notebook
```

---

# 🧪 Future Improvements

- XGBoost / LightGBM model comparison
- SHAP explainability (feature interpretation)
- Streamlit deployment
- Better hyperparameter optimization (RandomizedSearchCV)
- Handling class imbalance techniques

---

# 📚 Key Learnings

- How real ML pipelines are built
- Importance of preventing data leakage
- Encoding categorical variables properly
- Hyperparameter tuning in production models
- End-to-end ML system design

---

# 👨‍💻 Author

Prianshu Kumar  
GitHub: https://github.com/prianshu-kumar
