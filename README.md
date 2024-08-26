# California Housing Price Prediction

This project involves predicting California housing prices using various machine learning models. The dataset used is the California Housing dataset, and the models implemented include Linear Regression, Ridge Regression, Lasso Regression, Random Forest, and Gradient Boosting.


## Overview

The goal of this project is to build a predictive model that estimates housing prices in California based on various features such as the average number of rooms, average occupancy, and latitude/longitude. We employ different regression models to assess their performance and fine-tune them using hyperparameter tuning.

## Features

- **Data Preprocessing**: The dataset is transformed using polynomial features to capture non-linear relationships.
- **Model Training**: Multiple regression models are trained and evaluated using cross-validation.
- **Hyperparameter Tuning**: GridSearchCV is employed to find the best hyperparameters for the Random Forest and Gradient Boosting models.
- **Visualization**: Scatter plots of actual vs predicted prices are generated and saved.

## Installation

To run this project, you need to have Python 3.12 installed, along with the required libraries. You can install the necessary dependencies using `pip`:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```
## Project Description

The California Housing Price Prediction project aims to develop a robust predictive model to estimate housing prices in California using a variety of machine learning techniques. The project leverages the California Housing dataset, which includes features such as the average number of rooms, population, median income, and geographical location (latitude and longitude) of various districts.

The key objectives of the project are:
- **Data Exploration and Visualization**: To understand the distribution of the dataset and the relationships between different features.
- **Feature Engineering**: To create polynomial features that capture non-linear interactions between the variables.
- **Model Implementation**: To apply different regression techniques, including Linear Regression, Ridge Regression, Lasso Regression, Random Forest, and Gradient Boosting, to predict house prices.
- **Hyperparameter Tuning**: To optimize the performance of the models using GridSearchCV.
- **Model Evaluation**: To assess model performance using metrics like Mean Squared Error (MSE) and R-squared (R²) values.
- **Visualization**: To compare the predicted prices with the actual prices through scatter plots.

The project not only showcases the application of different machine learning models but also emphasizes the importance of model tuning and evaluation. The final output includes the best-performing models along with visualizations that provide insights into the model predictions.


## Results
The models were evaluated based on their Mean Squared Error (MSE) and R-squared (R²) values. The best models were further fine-tuned, and their performance was visualized through scatter plots.

## Visualizations
The scatter plots comparing the actual and predicted housing prices can be found in the ./plots directory.

## Contributing
Contributions are welcome! If you find any bugs or have suggestions for improvements, please feel free to open an issue or submit a pull request.
