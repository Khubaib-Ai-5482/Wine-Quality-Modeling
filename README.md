# Wine Quality Prediction using Random Forest Classifier

This project predicts wine quality based on physicochemical properties using a Random Forest Classifier. It includes preprocessing, hyperparameter tuning, model evaluation, and feature importance visualization.

## Dataset
- File: `WineQT.csv`
- Contains various physicochemical properties of wine samples
- Target column: `quality` (categorical/numerical quality score)
- `Id` column is dropped as it is not useful for prediction

## Data Preprocessing
- All categorical columns are encoded using LabelEncoder
- Dataset split into training (80%) and testing (20%) sets with stratification to preserve target distribution

## Model
- **Algorithm**: Random Forest Classifier
- **Hyperparameter Tuning**: Performed using GridSearchCV with 5-fold cross-validation
- **Tuned Parameters**:
    - n_estimators: 200, 300
    - max_depth: 6, 8, 10
    - min_samples_split: 2, 5
    - min_samples_leaf: 1, 2
    - class_weight: balanced
- The best model is selected automatically based on highest accuracy

## Evaluation
- **Accuracy** on training and testing sets
- **Classification Report**: Includes precision, recall, F1-score for each quality class
- **Confusion Matrix**: Visualized with a heatmap for easy interpretation

## Feature Importance
- The top contributing features to wine quality prediction are identified using the Random Forest feature_importances_
- Feature importance is visualized using a horizontal bar chart

## Usage
1. Load the dataset:
```python
df = pd.read_csv("WineQT.csv")
