# Insurance Charges Prediction using Machine Learning

## Overview

This project addresses a regression problem aimed at predicting individual medical insurance charges based on demographic and lifestyle features. The study applies classical machine learning techniques to tabular data and compares a linear baseline model with a non-linear ensemble model.

The project was developed as part of a Data Science / Machine Learning final project.

---

## Dataset

We use the Medical Cost Personal Dataset from Kaggle:

- Source: https://www.kaggle.com/mirichoi0218/insurance
- Records: 1,338
- Target variable: charges (medical insurance cost)

### Features

- age: age of primary beneficiary
- sex: gender of the insured
- bmi: body mass index
- children: number of dependents
- smoker: smoking status
- region: residential area

---

## Task

- Type: Regression
- Goal: Predict medical insurance charges (`charges`)

---

## Methodology

### Data Preprocessing

- Train/Test split: 80 / 20
- Categorical features encoded using One-Hot Encoding
- Numerical features passed without scaling

### Models

1. Baseline Model

   - Linear Regression
   - Used for interpretability and benchmarking

2. Improved Model
   - Random Forest Regressor
   - Captures non-linear relationships and feature interactions

### Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## Results

- Linear Regression provides a simple and interpretable baseline but struggles with non-linear patterns.
- Random Forest significantly improves predictive performance.
- Feature importance analysis shows that smoking status, BMI, and age are the most influential factors.

---

## Visualizations

The project includes:

- Distribution of insurance charges
- Charges comparison between smokers and non-smokers
- Actual vs. predicted values
- Feature importance (Random Forest)

All visualizations are displayed simultaneously in a single figure.

---

## Repository Structure

insurance-charges-prediction-ml/
│
├── insurance_ml_project.py
├── insurance.csv
├── README.md
└── venv/

---

## How to Run

1. Clone the repository

```bash
git clone https://github.com/your-username/insurance-charges-prediction-ml.git
cd insurance-charges-prediction-ml
Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate
Install dependencies
python3 -m pip install pandas numpy matplotlib seaborn scikit-learn
Run the project
python3 insurance_ml_project.py
```

## Conclusion
This project demonstrates that non-linear machine learning models outperform linear baselines in predicting medical insurance costs. Lifestyle-related features, particularly smoking status, play a dominant role in cost estimation.

