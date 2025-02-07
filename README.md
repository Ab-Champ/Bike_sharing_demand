# Predict Bike Sharing Demand with AutoGluon

## Project Overview

In this project, I have participated in the **Kaggle Bike Sharing Demand Competition**. The objective was to predict the demand for bike rentals in a city based on a variety of factors such as the weather, the time of day, the season, and more. To achieve this, I am using the **AutoGluon** library, which simplifies the machine learning process by automating tasks like model selection, hyperparameter tuning, and ensembling. This project allows for the application of machine learning concepts, feature engineering, and model optimization to compete in a global leaderboard.

### Objective

- **Predict the demand for bike rentals** using data from various sources (weather, time, season, etc.).
- **Optimize the model** by adding more features and tuning hyperparameters to improve prediction accuracy.
- Submit predictions for ranking and compete with other participants.
- Document the process and evaluate which techniques provided the most significant improvement in performance.

## Kaggle Competition

This project is part of the **Bike Sharing Demand competition**. The goal of the competition is to predict the number of bike rentals for a bike-sharing service based on weather and other features. The dataset includes columns such as:
- **Season**: The season in which the observation was made.
- **Year**: The year in which the observation was made.
- **Month**: The month when the data was recorded.
- **Hour**: The hour of the day (from 0 to 23).
- **Weather data**: Information on temperature, humidity, and precipitation levels.

The dataset also includes the target column, **count**, which represents the number of bikes rented in a given hour.

## Key Steps of the Project

### Data Exploration and Preprocessing
- Load the data into a DataFrame and perform basic statistical analysis and visualizations to understand feature distributions and relationships.
- Handle missing values and apply necessary transformations to the data (e.g., encoding categorical variables and scaling numerical features).

### Train Initial Model Using AutoGluon
- Use **AutoGluon** to automatically select the best model for the dataset, tune hyperparameters, and create an ensemble model for final predictions.
- Submit the model’s predictions to Kaggle for ranking.

### Feature Engineering and Model Improvement
- Enhance the model by performing **feature engineering**:
  - Add new features based on existing ones (e.g., creating interaction terms between time and weather).
  - Apply advanced feature selection techniques to identify the most important predictors.

### Hyperparameter Tuning
- Tune multiple hyperparameters of the AutoGluon model, such as:
  - The number of trees in Random Forest and Gradient Boosting models.
  - The learning rate and other settings for deep learning models.
  - The number of layers and neurons for neural network models.

### Final Submission
- After optimizing the model, make final predictions and submit them to Kaggle for competition ranking.

### Report
- Write a detailed report covering:
  - The steps taken to improve the model.
  - A discussion of feature engineering strategies and their impact.
  - A comparison of the models tried and their final performances.
  - Insights into the effectiveness of various techniques used to improve prediction accuracy.

## Dependencies

- **Python 3.7**
- **AutoGluon 0.2.0**
- **MXNet 1.8**
- **Pandas >= 1.2.4**
- **Matplotlib/Seaborn** (for data visualization)

## License

This project is licensed under the **MIT License**.
