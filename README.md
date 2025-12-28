# Credit Card Fraud Detection

## Problem Statement
Credit card fraud causes significant financial losses for banks and customers.
This project aims to detect fraudulent credit card transactions using machine learning techniques.

## Dataset
The dataset contains anonymized credit card transactions made by European cardholders.
It is highly imbalanced, with fraudulent transactions representing a very small fraction of the data.

## Approach
- Data loading and preprocessing
- Handling missing values
- Feature scaling using StandardScaler
- Logistic Regression with class-weighted learning to handle imbalance
- Model evaluation using precision, recall, and ROC-AUC
- Prediction on new transaction samples
- Saving trained model and scaler for reuse

## Results
The model achieved good recall for fraudulent transactions while maintaining a reasonable false positive rate.
ROC-AUC was used as the primary evaluation metric due to severe class imbalance.

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Joblib

## Use Case
This model can be integrated into financial systems to flag potentially fraudulent transactions
and assist in reducing financial risk.
