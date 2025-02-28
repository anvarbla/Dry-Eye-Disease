
# Dry Eye Disease Analysis

## Overview
This project focuses on analyzing a dataset related to dry eye disease using machine learning techniques. The dataset is preprocessed, features are engineered, and various classification models, including Logistic Regression, Random Forest, and K-Nearest Neighbors (KNN), are applied. Additionally, hyperparameter tuning is performed using Optuna to optimize the performance of the Random Forest model.

## Dataset
The dataset is loaded from:
```
https://raw.githubusercontent.com/anvarbla/Dry-Eye-Disease/refs/heads/main/Preprocessed_Dry_Eye_Dataset.csv
```

### Feature Engineering
1. **One-Hot Encoding**:
   - Categorical features such as "Sleep quality," "Stress level," and "Age Group" are one-hot encoded.
   - "BMI Category" and "Sleep duration group" are also encoded to ensure numerical representation.
2. **BMI Calculation**:
   - BMI is derived from height and weight.
   - BMI categories are defined: Underweight, Normal Weight, Overweight, and Obesity.
3. **Min-Max Scaling**:
   - Features such as "Heart rate," "Daily steps," "Physical activity," "Average screen time," "Systolic_BP," and "Diastolic_BP" are normalized.

## Oversampling
To address class imbalance, oversampling is applied to the training dataset to balance the number of samples in the "Dry Eye Disease" and "No Dry Eye Disease" classes.

## Machine Learning Models
### Logistic Regression
- Features are standardized using `StandardScaler`.
- The model is trained on an 80/20 split of the dataset.
- Performance is evaluated using accuracy and classification report.

### Random Forest Classifier
- The model is trained on both the original dataset and the oversampled dataset.
- Performance metrics include MAE, MSE, accuracy, and classification report.
- Optuna is used for hyperparameter tuning to find the best set of hyperparameters.

### K-Nearest Neighbors (KNN)
- The model is trained on the oversampled dataset with `n_neighbors=3`.
- Performance is evaluated using accuracy, precision, recall, and F1-score.

## Hyperparameter Optimization with Optuna
- Optuna is used to optimize `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features` for Random Forest.
- The best parameters are used to retrain the final Random Forest model.

## Results
- The models are evaluated on the test set, and accuracy, precision, recall, and other metrics are reported.
- Random Forest, with optimized hyperparameters, achieves the best performance.

## Dependencies
- `pandas`
- `scikit-learn`
- `optuna`
- `matplotlib`

## Usage
1. Install dependencies:
   ```sh
   pip install pandas scikit-learn optuna matplotlib
   ```
2. Run the script to preprocess data, train models, and evaluate performance.
   ```sh
   python main.py
   ```

## Future Improvements
- Explore deep learning approaches for classification.
- Implement additional feature selection techniques to improve model performance.
- Expand dataset with additional health-related parameters.

---

This README provides an overview of the project, detailing data preprocessing, model training, evaluation, and optimization using Optuna.

