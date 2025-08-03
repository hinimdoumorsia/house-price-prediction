# House Price Prediction and Analysis with Julia

This project demonstrates a workflow to generate, preprocess, and analyze a dataset for predicting house prices using both analytical methods and gradient descent. It also includes univariate and bivariate analyses of the dataset.

## Requirements

Ensure you have the following Julia packages installed:
- `Random`
- `LinearAlgebra`
- `Statistics`
- `DataFrames`
- `CSV`
- `StatsPlots`
- `Plots`
- `GLM`

Install missing packages using `Pkg.add("PackageName")`.

## Features

### 1. **Data Generation**
   - Creates a synthetic dataset of house prices based on features such as surface area, age, security, localization, number of bedrooms, and equipment quality.
   - Introduces random noise and outliers to simulate real-world conditions.

### 2. **Data Preprocessing**
   - Normalizes features for compatibility with machine learning models.
   - Splits the dataset into training (80%) and testing (20%) sets.
   - Implements Ridge Regularization to handle potential multicollinearity in the dataset.

### 3. **Prediction Models**
   - **Normal Equation**: A closed-form solution for linear regression.
   - **Mini-batch Gradient Descent**: An iterative optimization technique with early stopping based on convergence criteria.

### 4. **Evaluation Metrics**
   - **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values.
   - **R² (Coefficient of Determination)**: Indicates how well the model explains the variance in the data.
   - **Precision**: Evaluates the model’s prediction accuracy relative to the data variance.

### 5. **Univariate Analysis**
   - Generates histograms, density plots, and boxplots for each feature.
   - Computes descriptive statistics (mean, standard deviation, skewness, etc.).
   - Detects asymmetry in the distribution of numerical features.

### 6. **Bivariate Analysis**
   - Examines relationships between features and the target variable (price) using scatter plots and correlation coefficients.
   - Conducts ANOVA for categorical variables to assess their impact on price.
   - Produces a heatmap for the correlation matrix of key features.

## Usage Instructions

1. **Run the Data Generation Script**
   - The script generates a synthetic dataset with predefined relationships between features and the target variable (price).

2. **Train the Models**
   - Train models using both the normal equation and mini-batch gradient descent.
   - Evaluate the models on the test set using MSE, R², and precision metrics.

3. **Univariate and Bivariate Analysis**
   - Perform univariate analysis to understand the distribution of each feature.
   - Conduct bivariate analysis to explore relationships between features and the target variable.

4. **Save and Visualize Results**
   - Generated plots (histograms, boxplots, scatter plots, heatmaps) are saved as PNG files.

## Example Outputs

- **Training Results:**
  - MSE, R², and precision for both models.
  - Model coefficients for comparison between methods.

- **Analysis Results:**
  - Summary statistics for each feature.
  - Correlation matrix and p-values for categorical variables.

- **Visualizations:**
  - Histograms, density plots, and boxplots for univariate analysis.
  - Scatter plots and heatmaps for bivariate analysis.
