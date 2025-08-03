# Real Estate Data Analysis Project

## Overview
This project implements a comprehensive analysis of real estate data, including both univariate and bivariate analyses, along with the implementation of linear regression using both gradient descent and normal equation methods.

## Project Structure
```

├── scripts/
│   ├── data_generation.r
│   ├── univariate_analysis.r
│   └── bivariate_analysis.r
└── README.md
```

## Dependencies
Required R packages:
- ggplot2
- dplyr
- gridExtra
- car
- corrplot



## Installation
1. Install R dependencies:
```R
install.packages(c("dplyr", "gridExtra", "ggplot2", "car", "corrplot"))
```



## Features

### 1. Data Generation
- Generates synthetic real estate data with the following features:
  - Surface area (30-300 m²)
  - Building age (0-49 years)
  - Security rating (1-5)
  - Location rating (1-5)
  - Number of bedrooms (1-5)
  - Equipment rating (1-5)
  - Price (calculated based on features)
- Includes controlled outlier generation (2% of data)

### 2. Univariate Analysis
For each variable, provides:
- Basic statistical summaries
- Histograms with density plots
- Box plots
- Skewness analysis
- Frequency distributions for categorical variables

### 3. Bivariate Analysis
Includes:
- Scatter plots with correlation coefficients
- Box plots by category
- ANOVA tests for categorical variables
- Correlation matrix
- Statistical summaries of relationships with price

### 4. Linear Regression
Implements two methods:
- Mini-batch gradient descent
- Normal equation (analytical solution)

Performance metrics:
- Mean Squared Error (MSE)
- R-squared (R²)
- Model precision

## Usage

### Data Generation and Model Training
```R
# Load and prepare data
source("scripts/data_generation.r")

# Run analyses
source("scripts/univariate_analysis.r")
source("scripts/bivariate_analysis.r")
```


# Run analyses
include("univariate_analysis.jl")
include("bivariate_analysis.jl")
```

## Results
The analysis provides:
- Detailed visualization of variable distributions
- Correlation analysis between variables
- Price prediction model using linear regression
- Model performance metrics and comparisons

