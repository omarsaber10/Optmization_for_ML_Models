# Smoking Status Prediction using Optimized Machine Learning Models

## 📌 Overview

This project focuses on predicting smoking status using medical and biological indicators with the help of Machine Learning and Metaheuristic Optimization Algorithms.

The project combines:
- Machine Learning Models
- Hyperparameter Optimization
- Medical Data Analysis
- Performance Evaluation

Optimization algorithms such as **Grey Wolf Optimizer (GWO)** and **Hybrid PSOGSA** were applied to improve model accuracy and achieve better classification performance.

---

# 🎯 Objectives

- Predict smoking status using health-related features.
- Improve machine learning performance using optimization algorithms.
- Compare model performance before and after optimization.
- Evaluate optimization effectiveness on medical datasets.

---

# 📂 Dataset

The dataset contains medical and biological information related to smoking status.

## Features
- Hemoglobin
- Cholesterol
- Blood Pressure
- Liver Enzymes
- Blood Sugar
- BMI
- Gender
- Age

## Label
- Smoking Status
  - 0 → Non-Smoker
  - 1 → Smoker

---

# ⚙️ Data Preprocessing

The following preprocessing steps were applied:

- Removing unnecessary columns
- Encoding categorical features using LabelEncoder
- Feature scaling using StandardScaler
- Splitting dataset into training and testing sets

Example:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

# 🤖 Machine Learning Models

## 1. Support Vector Machine (SVM)

```python
from sklearn.svm import SVC

svm = SVC()
```

Optimized using:
- Grey Wolf Optimizer (GWO)

### Optimized Parameters
- C
- gamma

---

## 2. XGBoost Classifier

```python
from xgboost import XGBClassifier

xgb = XGBClassifier()
```

Optimized using:
- Hybrid PSOGSA

### Optimized Parameters
- n_estimators
- max_depth
- learning_rate

---

# 🧠 Optimization Algorithms

## Grey Wolf Optimizer (GWO)

GWO simulates the hunting behavior and leadership hierarchy of grey wolves.

### Advantages
- Fast convergence
- Strong exploration and exploitation balance
- Avoids local optimum solutions

---

## Hybrid PSOGSA

PSOGSA combines:
- Particle Swarm Optimization (PSO)
- Gravitational Search Algorithm (GSA)

### Advantages
- Better global search capability
- Handles high-dimensional problems
- Improves model generalization

---

# 📐 Mathematical Formulation

## Objective Function

The goal is to maximize classification accuracy:

\[
f(X)=Accuracy(X)
\]

---

## Constraints

- \(0.1 \leq C \leq 100\)
- \(0.0001 \leq gamma \leq 1\)
- \(50 \leq n_{estimators} \leq 300\)
- \(2 \leq max\_depth \leq 10\)

---

# 📊 Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

---

# 📈 Results

| Model | Optimization Algorithm | Accuracy Before | Accuracy After |
|------|------------------------|----------------|----------------|
| SVM | GWO | 76% | 78% |
| XGBoost | PSOGSA | 78% | 81% |

The optimized models achieved better performance compared to default configurations.

---

# 🚀 How to Run the Project

## 1. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow pyswarm
```

---

## 2. Run the Project

```bash
python final_project.py
```

Or open the notebook using Jupyter Notebook or Google Colab.

---

# 📁 Project Structure

```text
├── smoking.csv
├── final_project.py
├── README.md
└── results/
```

---

# 🔬 Future Work

- Add Deep Learning models
- Increase dataset size
- Apply Explainable AI techniques
- Deploy using Streamlit or Flask

---

# 👨‍💻 Author

**Omar Saber Ali**

Faculty of Engineering – New Ismailia National University

---

# 📚 References

1. Grey Wolf Optimizer (2014)
2. PSOGSA Hybrid Algorithm
3. XGBoost: A Scalable Tree Boosting System
4. Scikit-learn Documentation
5. XGBoost Documentation

