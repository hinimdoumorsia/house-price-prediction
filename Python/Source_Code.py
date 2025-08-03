import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Set a random seed for reproducibility
np.random.seed(42)

# Number of rows in the dataset
n_rows = 10000

# Generate features
surface = np.random.uniform(30, 300, n_rows)  # Surface area in square meters
age = np.random.randint(0, 50, n_rows)        # Age of the property in years
security = np.random.randint(1, 6, n_rows)   # Security rating (1 to 5)
localization = np.random.randint(1, 6, n_rows)  # Localization score (1 to 5)
num_bedrooms = np.random.randint(1, 6, n_rows)  # Number of bedrooms
equipment = np.random.randint(1, 6, n_rows)  # Equipment score (1 to 5)

# Define the true relationship for Price
price = (
    2000 * surface
    - 500 * age
    + 1500 * security
    + 3000 * localization
    + 10000 * num_bedrooms
    + 2000 * equipment
    + np.random.normal(0, 10000, n_rows)  # Add some noise
)

# Create the DataFrame
data = pd.DataFrame({
    'Surface': surface,
    'Age': age,
    'Security': security,
    'Localization': localization,
    'Bedrooms': num_bedrooms,
    'Equipment': equipment,
    'Price': price
})

# Introduce outliers
n_outliers = int(0.02 * n_rows)  # 2% of the data as outliers
outlier_indices = np.random.choice(data.index, n_outliers, replace=False)

# Modify the outliers
data.loc[outlier_indices, 'Surface'] *= np.random.uniform(2, 5, n_outliers)  # Increase surface size significantly
data.loc[outlier_indices, 'Price'] *= np.random.uniform(2, 5, n_outliers)    # Inflate prices drastically
data.loc[outlier_indices, 'Age'] *= np.random.uniform(1.5, 3, n_outliers)    # Increase age slightly for some properties

# Display dataset information
print("Dataset with Outliers:")
print(data.loc[outlier_indices].head())
print(data.describe())

# Features and target variable
X = data.drop(columns=['Price'])  # Features
y = data['Price']                 # Target variable

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split the training set into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
# Add a bias column (intercept term) to the feature matrix
X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]
X_test_bias = np.c_[np.ones(X_test.shape[0]), X_test]

# Compute theta using the normal equation
theta_normal = np.linalg.inv(X_train_bias.T @ X_train_bias) @ (X_train_bias.T @ y_train)

# Predict on the test set
y_test_pred_normal = X_test_bias @ theta_normal

# Evaluate the model

mse_normal = mean_squared_error(y_test, y_test_pred_normal)
r2_normal = r2_score(y_test, y_test_pred_normal)

print("Normal Equation Results:")
print(f"MSE: {mse_normal:.2f}")
print(f"R²: {r2_normal:.4f}")

# Check for NaN values
print(X_train.isnull().sum())  # For features
print(y_train.isnull().sum()) 

# Combine features into a single matrix
X = np.column_stack([surface, age, security, localization, num_bedrooms, equipment])
y = price

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mini-Batch Gradient Descent Implementation
class MiniBatchGradientDescentRegressor:
    def __init__(self, learning_rate=0.001, iterations=2000, batch_size=32, clip_gradients=True, max_gradient_value=10):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.batch_size = batch_size
        self.clip_gradients = clip_gradients
        self.max_gradient_value = max_gradient_value

    def fit(self, X, y):
        self.m = len(y)
        self.X = np.c_[np.ones(X.shape[0]), X]  # Add intercept term (bias)
        self.y = y
        self.theta = np.zeros(self.X.shape[1])

        for _ in range(self.iterations):
            indices = np.random.permutation(self.m)
            X_shuffled = self.X[indices]
            y_shuffled = self.y[indices]
            for i in range(0, self.m, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                predictions = X_batch.dot(self.theta)
                errors = predictions - y_batch
                gradients = (1 / len(y_batch)) * X_batch.T.dot(errors)

                if self.clip_gradients:
                    # Clip gradients to avoid large updates
                    gradients = np.clip(gradients, -self.max_gradient_value, self.max_gradient_value)

                self.theta -= self.learning_rate * gradients

        return self

    def predict(self, X):
        X_b = np.c_[np.ones(X.shape[0]), X]  # Add intercept term (bias)
        return X_b.dot(self.theta)

# Create and train the model
learning_rate = 0.001
iterations = 1000
batch_size = 64
model = MiniBatchGradientDescentRegressor(learning_rate, iterations, batch_size)

# Fit the model on the training data
model.fit(X_train, y_train)

# Predictions
y_pred_test = model.predict(X_test)

# Check if any NaNs are present in predictions
if np.any(np.isnan(y_pred_test)):
    print("Error: Predictions contain NaN values.")
else:
    # Evaluate the model
    mse_test = mean_squared_error(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)

    # Print results
    print(f"=== Mini-Batch Gradient Descent (Test) ===")
    print(f"MSE: {mse_test:.4f}, MAE: {mae_test:.4f}, RMSE: {rmse_test:.4f}, R²: {r2_test:.4f}")

    # Visualize predictions vs. actual values for the first feature (Surface)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test[:, 0], y_test, color='green', label='True Data (Test)', alpha=0.5)
    plt.scatter(X_test[:, 0], y_pred_test, color='blue', label='Predicted Data (Test)', alpha=0.5)
    plt.xlabel('Surface (m²)')
    plt.ylabel('Price (in Euros)')
    plt.title('Mini-Batch Gradient Descent Regression: Predictions vs True Values')
    plt.legend()
    plt.grid(True)
    plt.show()


# Minibatch avec Regularisaion , Standarisation et CrossValidation Grid Search



# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Mini-Batch Gradient Descent with L2 Regularization (Ridge Regression)
class MiniBatchGradientDescentRegressor:
    def __init__(self, learning_rate=0.001, iterations=2000, batch_size=32, lambda_reg=0.1, clip_gradients=True, max_gradient_value=10):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.clip_gradients = clip_gradients
        self.max_gradient_value = max_gradient_value

    def fit(self, X, y):
        self.m = len(y)
        self.X = np.c_[np.ones(X.shape[0]), X]  # Add intercept term (bias)
        self.y = y
        self.theta = np.zeros(self.X.shape[1])

        for _ in range(self.iterations):
            indices = np.random.permutation(self.m)
            X_shuffled = self.X[indices]
            y_shuffled = self.y[indices]
            for i in range(0, self.m, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                predictions = X_batch.dot(self.theta)
                errors = predictions - y_batch
                gradients = (1 / len(y_batch)) * X_batch.T.dot(errors)

                # Add L2 regularization (Ridge)
                gradients += (self.lambda_reg / self.m) * self.theta

                if self.clip_gradients:
                    # Clip gradients to avoid large updates
                    gradients = np.clip(gradients, -self.max_gradient_value, self.max_gradient_value)

                self.theta -= self.learning_rate * gradients

        return self

    def predict(self, X):
        X_b = np.c_[np.ones(X.shape[0]), X]  # Add intercept term (bias)
        return X_b.dot(self.theta)

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Hyperparameter tuning (learning rate, batch size, regularization strength)
learning_rates = [0.0001, 0.001, 0.01]
batch_sizes = [16, 32, 64]
lambda_regs = [0, 0.01, 0.1, 1]

best_model = None
best_mse = float('inf')
best_params = {}

# Grid search for hyperparameter tuning
for lr in learning_rates:
    for batch_size in batch_sizes:
        for lambda_reg in lambda_regs:
            model = MiniBatchGradientDescentRegressor(learning_rate=lr, batch_size=batch_size, lambda_reg=lambda_reg)
            fold_mse = []

            for train_index, val_index in kf.split(X_train_scaled):
                X_train_cv, X_val_cv = X_train_scaled[train_index], X_train_scaled[val_index]
                y_train_cv, y_val_cv = y_train_scaled[train_index], y_train_scaled[val_index]

                model.fit(X_train_cv, y_train_cv)
                y_val_pred = model.predict(X_val_cv)
                fold_mse.append(mean_squared_error(y_val_cv, y_val_pred))

            avg_mse = np.mean(fold_mse)
            print(f"Learning Rate: {lr}, Batch Size: {batch_size}, Lambda: {lambda_reg} -> Avg MSE: {avg_mse:.4f}")

            # Store the best model
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_model = model
                best_params = {'learning_rate': lr, 'batch_size': batch_size, 'lambda_reg': lambda_reg}

# Train the best model on the entire training set
best_model.fit(X_train_scaled, y_train_scaled)
# Test the best model
y_test_pred = best_model.predict(X_test_scaled)
y_test_pred_rescaled = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()

# Evaluate the best model on the test set
mse_test = mean_squared_error(y_test, y_test_pred_rescaled)
mae_test = mean_absolute_error(y_test, y_test_pred_rescaled)
r2_test = r2_score(y_test, y_test_pred_rescaled)
rmse_test = np.sqrt(mse_test)

print(f"Best Model (Test): MSE: {mse_test:.4f}, MAE: {mae_test:.4f}, RMSE: {rmse_test:.4f}, R²: {r2_test:.4f}")

# Plotting the regression line between surface and predicted price
plt.figure(figsize=(10, 6))
plt.plot(X_test[:, 0], y_test_pred_rescaled, color='blue', label='Predicted Data (Test)', linewidth=2)
plt.xlabel('Surface (m²)')
plt.ylabel('Price (in Euros)')
plt.title(f'Mini-Batch Gradient Descent with Regularization\nBest Model (lr={best_params["learning_rate"]}, batch_size={best_params["batch_size"]}, lambda={best_params["lambda_reg"]})')
plt.legend()
plt.grid(True)
plt.show()

# Predict using the best model
y_test_pred = best_model.predict(X_test_scaled)
y_test_pred_rescaled = scaler_y.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()

# Plot Prediction vs Actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred_rescaled, color='green', alpha=0.5, label="Predicted vs Actual")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label="Perfect Prediction")
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Predicted vs Actual Prices')
plt.legend()
plt.grid(True)
plt.show()
#analyse univariée
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

my_data = pd.read_csv('/content/house_price_data.csv')
my_data

my_data.columns

# Analyse univariée pour chaque variable
for col in my_data.columns:
  print(f"Analyse de la variable : {col}")

  # Statistiques descriptives
  print(my_data[col].describe())

  # Type de la colonne
  col_type = my_data[col].dtype

  # Choix du graphique en fonction du type de données
  if col_type in ['int64', 'float64']:
    # Histogramme
    plt.figure(figsize=(8, 6))  # Ajuster la taille de la figure
    sns.histplot(my_data[col], bins=10, kde=True)
    plt.title(f'Histogramme de {col}')
    plt.xlabel(col)
    plt.ylabel('Fréquence')
    plt.show()

    # Boîte à moustaches
    plt.figure(figsize=(8, 6))  # Ajuster la taille de la figure
    sns.boxplot(y=my_data[col]) # Utiliser y pour les boxplots verticaux
    plt.title(f'Boîte à moustaches de {col}')
    plt.ylabel(col)
    plt.show()

    # Conclusion (exemple, à adapter)
    if my_data[col].skew() > 0.5:
      print(f"La distribution de {col} est positivement asymétrique (skew > 0.5).")
    elif my_data[col].skew() < -0.5:
      print(f"La distribution de {col} est négativement asymétrique (skew < -0.5).")
    else:
      print(f"La distribution de {col} est relativement symétrique.")

  elif col_type == 'object':
      # Diagramme en barres pour les variables catégorielles
      plt.figure(figsize=(10, 6))
      my_data[col].value_counts().plot(kind='bar')
      plt.title(f'Diagramme en barres de {col}')
      plt.xlabel(col)
      plt.ylabel('Fréquence')
      plt.xticks(rotation=45, ha='right') # Rotation des labels de l'axe x
      plt.show()
      print(f"La variable {col} est catégorielle. Les valeurs les plus fréquentes sont {my_data[col].value_counts().head()}.")
  else:
    print(f"Type de données inconnu pour la colonne {col}, impossible de générer un graphique.")
  print("-" * 30) # Séparation entre les analyses

  

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

def analyse_bivariee(df):
    # Configuration des plots
    fig, axes = plt.subplots(3, 2, figsize=(15, 20))
    fig.suptitle('Analyse Bivariée - Prix des Maisons', fontsize=16)

    # Variables numériques
    num_vars = ['Surface', 'Age', 'Bedrooms']

    # 1. Analyse des variables numériques
    for i, var in enumerate(num_vars):
        # Scatter plot
        axes[i,0].scatter(df[var], df['Price'])
        axes[i,0].set_xlabel(var)
        axes[i,0].set_ylabel('Prix')
        axes[i,0].set_title(f'Prix vs {var}')

        # Calcul de corrélation
        corr = df[var].corr(df['Price'])
        axes[i,0].text(0.05, 0.95, f'Corrélation: {corr:.2f}',
                      transform=axes[i,0].transAxes)

        # Boxplot
        df.boxplot(column='Price', by=var, ax=axes[i,1])
        axes[i,1].set_title(f'Distribution des Prix par {var}')

    plt.tight_layout()

    # 2. Analyse des variables catégorielles
    cat_vars = ['Security', 'Localization', 'Equipment']
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 15))

    for i, var in enumerate(cat_vars):
        # Boxplot par catégorie
        data = [df[df[var] == cat]['Price'] for cat in df[var].unique()]
        axes2[i//2, i%2].boxplot(data, labels=df[var].unique())
        axes2[i//2, i%2].set_xlabel(var)
        axes2[i//2, i%2].set_ylabel('Prix')
        axes2[i//2, i%2].set_title(f'Prix par {var}')
        axes2[i//2, i%2].tick_params(axis='x', rotation=45)

        # Test ANOVA
        f_stat, p_value = stats.f_oneway(*data)
        axes2[i//2, i%2].text(0.05, 0.95, f'p-value ANOVA: {p_value:.4f}',
                             transform=axes2[i//2, i%2].transAxes)

    plt.tight_layout()

    # 3. Matrice de corrélation
    corr_matrix = df[num_vars + ['Price']].corr()
    fig3 = plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center')
    plt.colorbar()
    plt.title('Matrice de Corrélation')

    # 4. Statistiques récapitulatives
    stats_summary = pd.DataFrame({
        'Variable': num_vars,
        'Corrélation_avec_Prix': [df[var].corr(df['Price']) for var in num_vars]
    })

    return stats_summary

df = pd.read_csv('house_price_data.csv')
resultats = analyse_bivariee(df)
plt.show()
