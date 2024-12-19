# Liver Disease Prediction

## Project Overview
This project involves analyzing liver disease data to predict the likelihood of liver disease in patients. The goal is to build a predictive model that uses clinical parameters to identify whether an individual has liver disease. The model leverages machine learning algorithms to classify patients based on their health metrics.

## Features
- **Data Cleaning and Preprocessing**
- **Exploratory Data Analysis (EDA)**
- **Visualizations (Histograms, Boxplots, Correlation Heatmap, etc.)**
- **Outlier Detection and Removal**
- **Predictive Modeling using Machine Learning Algorithms**
- **Hyperparameter Tuning**
- **Feature Importance Analysis**
- **Custom Function for Predicting Liver Disease for New Patients**

## Technologies Used
- **Python Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Imbalanced-learn
- **Visualization**: Seaborn, Matplotlib
- **Machine Learning Models**:
  - Logistic Regression
  - Random Forest
  - Decision Tree
  - Support Vector Classifier
  - K-Nearest Neighbors
  - Naive Bayes
  - Bagging Classifier
- **Data Processing**: Label Encoding, SMOTE (Synthetic Minority Over-sampling Technique)
- **Evaluation Metrics**: Accuracy, Classification Report, Confusion Matrix

## Dataset
The dataset contains clinical information about patients, including:
- `Age`, `Gender`
- Liver enzyme levels like `Total_Bilirubin`, `Direct_Bilirubin`, `Alkaline_Phosphotase`, `Alamine_Aminotransferase`, `Aspartate_Aminotransferase`
- Biomarkers like `Total_Protiens`, `Albumin`, and `Albumin_and_Globulin_Ratio`
- The target variable: `Dataset`, where 1 indicates liver disease and 2 indicates a healthy individual.

## Key Steps
### 1. Data Preprocessing
- Cleaned missing values and handled categorical data using Label Encoding.
- Applied SMOTE to handle class imbalance.
- Removed outliers from key features like:
  - `Total_Bilirubin`
  - `Direct_Bilirubin`
  - `Albumin`

### 2. Exploratory Data Analysis (EDA)
- Analyzed data distribution using histograms and box plots.
- Examined correlations between liver enzymes and biomarkers.
- Created a correlation heatmap to understand feature relationships.

### Correlation Heatmap
![Correlation Heatmap](https://github.com/omkarnitturkar/Liver-Patient-Analysis-/blob/main/Heatmap_Liver.png)

### 3. Model Training and Evaluation
- Trained multiple machine learning models and compared their performance.
- Used hyperparameter tuning to optimize model parameters.
- Identified the top 3 most important features affecting liver disease.

### Performance Comparison
![Model Performance Comparison](https://github.com/omkarnitturkar/Liver-Patient-Analysis-/blob/main/Comparison_result_liver.png)

