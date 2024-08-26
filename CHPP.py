# Import Libraries
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create the 'plots' directory if it doesn't exist
if not os.path.exists('./plots'):
    os.makedirs('./plots')

# Load and explore the California housing dataset
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['PRICE'] = housing.target

# Display the first few rows of the dataset
print("Data Head:")
print(data.head())

# Display descriptive statistics of the dataset
print("\nDescriptive Statistics:")
print(data.describe())

# Check for any missing values in the dataset
print("\nMissing Values:")
print(data.isnull().sum())

# Feature Engineering: Generate polynomial features (degree=2) for the dataset
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(data.drop('PRICE', axis=1))

# Convert polynomial features into a DataFrame and add the target variable 'PRICE'
poly_data = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(data.columns[:-1]))
poly_data['PRICE'] = data['PRICE']

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(poly_data.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
plt.show()

# Data Preprocessing: Split the data into training and testing sets
X = poly_data.drop('PRICE', axis=1)  # Features
y = poly_data['PRICE']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features for better model performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Selection and Cross-Validation: Evaluate different regression models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor()
}

cv_results = {}
for name, model in models.items():
    print(f"Running cross-validation for {name}...")
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    mean_mse = np.mean(-scores)  # Calculate the mean MSE
    cv_results[name] = mean_mse
    print(f"{name} - Cross-Validation MSE: {mean_mse}")

# Hyperparameter Tuning: Use GridSearchCV to find the best hyperparameters for selected models
param_grid = {
    'Random Forest': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
}

best_models = {}
for name in ['Random Forest', 'Gradient Boosting']:
    print(f"Performing Grid Search for {name}...")
    grid_search = GridSearchCV(models[name], param_grid[name], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_  # Store the best model
    print(f"Best parameters for {name}: {grid_search.best_params_}")

# Model Evaluation: Evaluate the best models on the test set and save scatter plots
for name, model in best_models.items():
    print(f"Evaluating model: {name}...")
    y_pred = model.predict(X_test)  # Predict on the test set
    mse = mean_squared_error(y_test, y_pred)  # Calculate the MSE
    r2 = r2_score(y_test, y_pred)  # Calculate the R2 score
    print(f"{name} - Test MSE: {mse}, R2: {r2}")

    # Scatter plot: Actual vs Predicted Prices
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title(f'Actual vs Predicted Prices - {name}')
    
    # Save the scatter plot to the 'plots' directory
    plot_file = f'./plots/{name}_scatter_plot.png'
    try:
        plt.savefig(plot_file)
        print(f"{name} scatter plot saved successfully at {plot_file}.")
    except Exception as e:
        print(f"Failed to save {name} scatter plot. Error: {e}")
    finally:
        plt.close()
