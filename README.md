# Credit Card Fraud Detection 🔍

## Overview
A machine learning project to detect fraudulent credit card transactions using real-world data from Kaggle. The dataset contains 284,807 transactions with only 0.17% fraud cases, making this a classic imbalanced classification problem.

## Problem Statement
Given transaction features, predict whether a transaction is fraudulent or legitimate. Since fraud cases are extremely rare, standard accuracy metrics are insufficient — **Recall** is prioritized to minimize missed fraud cases.

## Dataset
- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Features:** 28 PCA-transformed features (V1-V28), Amount, Time
- **Target:** Class (0 = Normal, 1 = Fraud)

## Approach

### 1. Exploratory Data Analysis (EDA)
- Analyzed class imbalance (99.83% normal vs 0.17% fraud)
- Visualized transaction amount distribution for fraud vs normal
- Feature scaling with StandardScaler

### 2. Handling Imbalanced Data
Used **SMOTE** (Synthetic Minority Oversampling Technique) to balance the training data from 492 fraud cases to 227,451.

### 3. Model Comparison

| Model | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Logistic Regression | 0.83 | 0.64 | 0.72 |
| Logistic Regression + SMOTE | 0.06 | 0.92 | 0.11 |
| **Random Forest + SMOTE** | **0.87** | **0.83** | **0.85** |
| Gradient Boosting + SMOTE | 0.11 | 0.91 | 0.19 |

### 4. Best Model: Random Forest
Random Forest with SMOTE achieved the best balance between Precision and Recall, correctly identifying 83% of fraudulent transactions while maintaining high precision.

## Why Recall Matters Here
In fraud detection, missing a fraudulent transaction (false negative) is far more costly than incorrectly flagging a legitimate one (false positive). Therefore, **Recall** is our primary metric.

## Tech Stack
- Python 3.12
- pandas, numpy
- scikit-learn
- imbalanced-learn (SMOTE)
- matplotlib, seaborn

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```
Download the dataset from Kaggle and place `creditcard.csv` in the project root, then run `01_EDA.ipynb`.
