#  Regularized Regression & Gradient Boosting on Boston Housing Dataset

## 📋 Project Overview
This project demonstrates various regression techniques including Linear Regression, Lasso & Ridge Regularization, and Gradient Boosting implementation on the famous Boston Housing dataset to predict median house prices.

## 🎯 Key Features
- **Multiple Regression Techniques**: Linear, Lasso, Ridge, and Gradient Boosting
- **Regularization Implementation**: Understanding L1 (Lasso) and L2 (Ridge) regularization
- **Manual Gradient Boosting**: Step-by-step implementation of gradient boosting
- **Model Evaluation**: Performance comparison using R² scores
- **Feature Engineering**: Data preprocessing and exploration

## 📊 Dataset Information
The Boston Housing dataset contains 506 samples with 13 features:
- **crim**: Per capita crime rate by town
- **zn**: Proportion of residential land zoned for lots over 25,000 sq.ft
- **indus**: Proportion of non-retail business acres per town
- **chas**: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- **nox**: Nitric oxides concentration (parts per 10 million)
- **rm**: Average number of rooms per dwelling
- **age**: Proportion of owner-occupied units built prior to 1940
- **dis**: Weighted distances to five Boston employment centres
- **rad**: Index of accessibility to radial highways
- **tax**: Full-value property-tax rate per $10,000
- **ptratio**: Pupil-teacher ratio by town
- **black**: 1000(Bk - 0.63)² where Bk is the proportion of blacks by town
- **lstat**: Percent lower status of the population

**Target variable**: 
- **medv**: Median value of owner-occupied homes in $1000's

## 🛠️ Technologies Used
- Python 3.x
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- Jupyter Notebook
- 
## 🚀 Code Explanation

```python
# Import necessary libraries
from sklearn.linear_model import Lasso, Ridge, LinearRegression  # Regression models
import numpy as np  # Numerical computing
import pandas as pd  # Data manipulation
import seaborn as sns  # Statistical visualization
import matplotlib.pyplot as plt  # Plotting library
from sklearn.model_selection import train_test_split  # Data splitting

# Load the Boston housing dataset
df = pd.read_csv("Boston.csv")

# Explore dataset structure
df.head()  # Display first 5 rows
df.info()  # Show dataset information and data types
df.isnull().sum()  # Check for missing values

# Data preprocessing
df.drop("Unnamed: 0", axis=1, inplace=True)  # Remove unnecessary index column

# Prepare features and target variable
x = df.drop("medv", axis=1)  # Feature matrix (all columns except medv)
y = df["medv"]  # Target vector (median home values)

# Split data into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)

# Linear Regression Baseline
Li = LinearRegression()
Li.fit(x_train, y_train)  # Train the model
print("test", Li.score(x_test, y_test))  # Test R² score: 0.7439
print("train", Li.score(x_train, y_train))  # Train R² score: 0.7360

# Lasso Regression (L1 Regularization) with high alpha
La = Lasso(alpha=100)  # Strong regularization
La.fit(x_train, y_train)
print("test", La.score(x_test, y_test))  # Test R²: 0.1707 (too much regularization)
print("train", La.score(x_train, y_train))  # Train R²: 0.2354

# Ridge Regression (L2 Regularization) with high alpha
Ra = Ridge(alpha=100)  # Strong regularization
Ra.fit(x_train, y_train)
print("test", Ra.score(x_test, y_test))  # Test R²: 0.7221
print("train", Ra.score(x_train, y_train))  # Train R²: 0.7083

# Manual Gradient Boosting Implementation
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# First weak learner (shallow decision tree)
tree_reg1 = DecisionTreeRegressor(max_depth=3)
tree_reg1.fit(x_train, y_train) 
y1 = y_train - tree_reg1.predict(x_train)  # Calculate residuals
print(tree_reg1.score(x_test, y_test))  # R²: 0.6539

# Second weak learner trained on residuals
tree_reg2 = DecisionTreeRegressor(max_depth=4)
tree_reg2.fit(x_train, y1)
y2 = y1 - tree_reg2.predict(x_train)  # Calculate new residuals
print(tree_reg2.score(x_test, y_test))  # R²: -7.5844 (negative - learning residuals)

# Third weak learner trained on residuals
tree_reg3 = DecisionTreeRegressor(max_depth=5)
tree_reg3.fit(x_train, y2)
y3 = y2 - tree_reg3.predict(x_train)  # Calculate new residuals
print(tree_reg3.score(x_test, y_test))  # R²: -7.2961

# Combine predictions from all trees (gradient boosting)
y_pred = sum(tree.predict(x_test) for tree in (tree_reg1, tree_reg2, tree_reg3))
print(r2_score(y_test, y_pred))  # Combined R²: 0.7891 (improved!)

# Fourth weak learner for further improvement
tree_reg4 = DecisionTreeRegressor(max_depth=5)
tree_reg4.fit(x_train, y3)
print(tree_reg4.score(x_test, y_test))  # R²: -7.1738

# Final combined prediction
y_pred = sum(tree.predict(x_test) for tree in (tree_reg1, tree_reg2, tree_reg3, tree_reg4))
print(r2_score(y_test, y_pred))  # Final R²: 0.7749
```

## 📈 Results & Analysis

### Model Performance Comparison:
1. **Linear Regression**: R² = 0.7439 (baseline)
2. **Lasso (α=100)**: R² = 0.1707 (over-regularized)
3. **Ridge (α=100)**: R² = 0.7221 (good regularization)
4. **Gradient Boosting**: R² = 0.7749 (best performance)

### Key Insights:
- Regularization helps prevent overfitting but requires careful tuning of alpha
- Gradient boosting significantly improves prediction accuracy by combining weak learners
- The manual gradient boosting implementation demonstrates the algorithm's iterative nature
- Residual learning allows each subsequent model to correct errors of previous models

## 🎓 Learning Outcomes
This project provides hands-on experience with:
- Regularization techniques (Lasso vs Ridge)
- The bias-variance tradeoff in practice
- Manual implementation of gradient boosting
- Model evaluation using R² scores
- Feature importance analysis in regression problems

---
