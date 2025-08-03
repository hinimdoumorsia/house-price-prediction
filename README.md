#  House Price Prediction – Machine Learning Project

This project was conducted as part of an academic assignment aiming to predict house prices from synthetic features, including noise handling, outliers, and advanced modeling techniques.

---

## Team Members

- **HINIMDOU Morsia Guitdam** — *Team Leader, responsible for coordination, modeling, and project structure*  
- **KHALD Adam**  
- **DARRAJ Mohamed Amine**  
- **MOKHASS Yassine**  
- **Asmae Elhakioui**

---

##  Objective

Predict the **price of a house** (target variable: `Price`) based on features such as:

- Area  
- Property age  
- Security  
- Location  
- Number of bedrooms  
- Amenities

---

##  Main Project Steps

### 1. Data Generation
- Synthetic data simulated using `numpy`  
- Added random noise and 2% **outliers**

### 2. Preprocessing
- Handling missing values  
- Data standardization  
- Splitting into training, validation, and test sets

### 3. Modeling
- Linear regression (normal equation)  
- Mini-batch gradient descent with:  
  - Gradient clipping  
  - L2 regularization (Ridge Regression)  
- Grid search for hyperparameter tuning

### 4. Statistical Analysis
- Univariate analysis: histograms, boxplots, skewness  
- Bivariate analysis: correlation matrix, ANOVA

### 5. Visualization
- Convergence curves  
- Predictions vs actual values  
- Interactive charts

---

## Model Complexity and Performance Analysis

- Analysis of computational complexity for each modeling approach  
- Impact of regularization and gradient clipping on convergence speed  
- Comparison of training times for normal equation vs mini-batch gradient descent  
- Assessment of overfitting and underfitting behavior through learning curves

---

##  Evaluation Metrics

- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  
- Coefficient of Determination (R²)  
- Analysis of residuals distribution and error patterns

---

## Repository Structure

There are three folders named **R**, **Julia**, and **Python**.  
Each folder contains:  
- Source code  
- A folder named **plots**  
- A file named `README.txt`

---
