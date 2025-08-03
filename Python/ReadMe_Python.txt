Data Description

Features:

Surface: Surface area of the house (in square meters).

Age: Age of the property (in years).

Security: Security rating (1 to 5).

Localization: Localization score (1 to 5).

Bedrooms: Number of bedrooms.

Equipment: Equipment score (1 to 5).

Target:

Price: Predicted price of the house.

Synthetic Adjustments:

Noise is added to make predictions realistic.

2% of the data contains outliers.

Key Scripts

1. Data Generation

Generates synthetic data and introduces outliers:

Uses numpy to create features.

Adds random noise and outliers to simulate real-world scenarios.

2. Preprocessing

Splits the dataset into training, validation, and test sets.

Standardizes the features for gradient descent optimization.

Introduces outliers and handles missing data.

3. Modeling

a. Linear Regression (Normal Equation):

Computes model parameters using the closed-form solution.

b. Mini-Batch Gradient Descent:

Implements a custom mini-batch gradient descent algorithm.

Includes gradient clipping to avoid large updates.

Ridge regression (L2 regularization) for better generalization.

c. Hyperparameter Tuning:

Performs grid search over learning rate, batch size, and regularization strength using cross-validation.

4. Visualization

Scatter plots and histograms for data exploration.

Regression lines and predicted vs actual value plots for model evaluation.

5. Statistical Analysis

Univariate Analysis:

Generates histograms and boxplots for each feature.

Provides skewness insights.

Bivariate Analysis:

Correlation matrix for numeric features.

ANOVA tests for categorical features.