# Liver Disease Prediction Project

This project involves building a predictive model for diagnosing liver disease using a dataset of patient health metrics. The primary focus is to preprocess the data, handle class imbalances, evaluate multiple machine learning algorithms, and select the best-performing model.

## Table of Contents
- [Dataset Description](#dataset-description)
- [Project Workflow](#project-workflow)
- [Libraries Used](#libraries-used)
- [Insights from Exploratory Data Analysis (EDA)](#insights-from-exploratory-data-analysis-eda)
- [Challenges in Building Predictive Models](#challenges-in-building-predictive-models)
- [Model Comparison](#model-comparison)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [How to Run the Code](#how-to-run-the-code)
- [Future Improvements](#future-improvements)

## Dataset Description

The dataset used in this project is **Indian Liver Patient Dataset**. It can be downloaded from Kaggle using the following link:  
[Indian Liver Patient Dataset on Kaggle](https://www.kaggle.com/uciml/indian-liver-patient-records)

The dataset consists of the following columns:

1. **Age**: Age of the patient.
2. **Gender**: Gender of the patient (Male/Female).
3. **Total_Bilirubin**: Total bilirubin levels in the blood.
4. **Direct_Bilirubin**: Direct bilirubin levels in the blood.
5. **Alkaline_Phosphotase**: Alkaline phosphatase enzyme levels.
6. **Alamine_Aminotransferase**: Alamine aminotransferase enzyme levels.
7. **Aspartate_Aminotransferase**: Aspartate aminotransferase enzyme levels.
8. **Total_Proteins**: Total protein levels in the blood.
9. **Albumin**: Albumin levels in the blood.
10. **Albumin_and_Globulin_Ratio**: Ratio of albumin to globulin.
11. **Dataset**: Target variable, indicating if the patient has liver disease (1) or not (2).

## Project Workflow

1. **Data Preprocessing**:
   - Handled missing values by removing incomplete rows.
   - Converted categorical data (Gender) to numerical format.
   - Addressed class imbalance using SMOTE (Synthetic Minority Oversampling Technique).
   - Removed duplicate entries.
   - Performed feature scaling where necessary.

2. **Exploratory Data Analysis (EDA)**:
   - Generated histograms for individual features to understand their distributions.
   - Created a correlation heatmap to identify relationships between variables.
   - Visualized outliers using box plots.

3. **Model Development**:
   - Evaluated multiple machine learning algorithms:
     - Logistic Regression
     - Random Forest
     - Decision Tree
     - Support Vector Classifier
     - K-Nearest Neighbors (KNN)
     - Naive Bayes
     - Bagging Classifier
   - Performed hyperparameter tuning using RandomizedSearchCV for the best-performing model.

4. **Evaluation**:
   - Compared models based on metrics like accuracy, precision, recall, F1-score, and AUC-ROC.
   - Visualized model performance before and after SMOTE using bar charts.

5. **Model Saving**:
   - Saved the final model using `pickle` for future predictions.

## Libraries Used

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `imblearn`
- `pickle`

## Insights from Exploratory Data Analysis (EDA)

1. **Age**: Most patients are in the 30-60 age range.
2. **Gender**: The dataset has more males than females.
3. **Bilirubin Levels**: Highly skewed, with most values being low.
4. **Enzymes**: Right-skewed distributions with significant outliers.
5. **Proteins and Albumin**: Fairly normal distributions centered around clinically typical ranges.
6. **Target Variable**: Imbalanced, with more cases of liver disease.

## Challenges in Building Predictive Models

1. **Data Quality Issues**:
   - Missing values and outliers.
   - Imbalanced dataset.
2. **Feature Selection**:
   - High correlation among certain features.
   - Scaling needed for models like SVM and KNN.
3. **Model Selection**:
   - Avoiding overfitting in complex models like Decision Trees.
   - Selecting the best metric for imbalanced datasets.
4. **Computational Complexity**:
   - Hyperparameter tuning required significant resources.

## Model Comparison

| Classifier               | Accuracy Before SMOTE | Accuracy After SMOTE |
|--------------------------|-----------------------|----------------------|
| Logistic Regression      | 62.93%                | 73.28%               |
| Random Forest            | 65.52%                | 70.69%               |
| Decision Tree            | 58.62%                | 60.34%               |
| Support Vector Classifier| 62.93%                | 61.21%               |
| KNN                      | 62.07%                | 62.93%               |
| Naive Bayes              | 58.62%                | 61.21%               |
| Bagging                  | 63.79%                | 60.34%               |

## Hyperparameter Tuning

Performed hyperparameter optimization for Random Forest using `RandomizedSearchCV`. Best parameters:

```json
{
  "n_estimators": 100,
  "max_features": 0.6,
  "max_depth": 8,
  "max_samples": 0.75
}
