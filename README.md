# Credit Default Prediction

## Overview
This project predicts the likelihood of credit default using a structured dataset. It applies preprocessing, feature engineering, and advanced machine learning with LightGBM to classify whether a customer will default on credit. Feature selection with Recursive Feature Elimination (RFE) is also used to identify the most important predictors.

## Key Insights
- Handled missing values and categorical encoding for features such as home ownership, purpose, and employment duration.
- RFE identified the most relevant features for predicting credit default.
- LightGBM achieved strong performance using selected features.

## Tech Stack
- Python (pandas, numpy, matplotlib)
- scikit-learn (preprocessing, feature selection, metrics)
- LightGBM (classification)
