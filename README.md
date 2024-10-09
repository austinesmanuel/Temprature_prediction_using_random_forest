# Predicting Average Temperature for a City Based on Geographic Features

## Problem Statement

The goal of this project is to predict the average daily temperature (`TAVG`) for City D using geographic and weather data from three neighboring cities (A, B, and C). The data includes features such as latitude, longitude, elevation, precipitation, snow depth, and temperatures (maximum, minimum, and average) for each of these three cities. By using this information, we aim to build a machine learning model capable of predicting City D's average temperature based on its geographical proximity and weather-related features of the neighboring cities.

This is an important task for weather forecasting, environmental monitoring, and broader climate research, as understanding how geographical and weather patterns influence temperatures can lead to more accurate predictions.

## Dataset Overview

The dataset can be accessed from Kaggle: [Shaastra Techathon AI-ML Challenge](https://www.kaggle.com/competitions/shaastra-techathon-ai-ml-challenge-2/data).

The dataset consists of geographical and weather data from three cities (A, B, and C) and requires predictions for City D. The provided features include:

- **Date**: The day on which the measurements were recorded (`DD-MM-YYYY`).
- **Geographical Features**:
  - `LATITUDE_A`, `LATITUDE_B`, `LATITUDE_C`: Latitude of cities A, B, and C.
  - `LONGITUDE_A`, `LONGITUDE_B`, `LONGITUDE_C`: Longitude of cities A, B, and C.
  - `ELEVATION_A`, `ELEVATION_B`, `ELEVATION_C`: Elevation of cities A, B, and C.
  
- **Weather Features**:
  - `PRCP_A`, `PRCP_B`, `PRCP_C`: Daily precipitation in cities A, B, and C.
  - `SNWD_A`, `SNWD_B`, `SNWD_C`: Snow depth in cities A, B, and C.
  - `TMAX_A`, `TMAX_B`, `TMAX_C`: Maximum temperature in cities A, B, and C.
  - `TMIN_A`, `TMIN_B`, `TMIN_C`: Minimum temperature in cities A, B, and C.
  - `TAVG_A`, `TAVG_B`, `TAVG_C`: Average temperature in cities A, B, and C (Target: City D's TAVG).

### Dataset Purpose

- The dataset provides geographical and temperature data for three cities (A, B, and C) to predict the average temperature for City D.
- The target variable is the average temperature (`TAVG`) of City D, which we aim to predict using the information from the other cities.
- 
## Solution Code Overview
This repository contains a Python script that builds a machine learning pipeline using the **Random Forest Regressor** from the scikit-learn library. The script performs the following tasks:

1. Loads and processes the dataset.
2. Handles missing values by applying **K-Nearest Neighbors (KNN) Imputation**.
3. Preprocesses date information and standardizes the features.
4. Splits the data into training and testing sets.
5. Trains the **Random Forest** model and evaluates its performance using various metrics.
6. Saves the preprocessed data.

## Requirements

The following Python libraries are required to run the script:
- **pandas**: For data manipulation.
- **numpy**: For numerical computations.
- **scikit-learn**: For machine learning algorithms, imputation methods, and performance metrics.

Install the dependencies by running:
```bash
pip install pandas numpy scikit-learn
```

## Dataset

The script is designed to handle two datasets:
1. **train.csv**: The training dataset that contains features and target values.
2. **test.csv**: The test dataset, which is used to evaluate the model after training.

Both datasets should have a **DATE** column, which will be split into **day**, **month**, and **year** columns for feature extraction.

## Code Breakdown

### 1. Data Loading

The first step is to load the dataset using `pandas.read_csv()`. This loads the data into a Pandas DataFrame for manipulation.

```python
df = pd.read_csv('/content/train.csv')
```

A serial number column is inserted at the beginning of the DataFrame to preserve the original order of rows. This will be useful for sorting the data after missing values have been imputed.

```python
df.insert(0, 'Serial Number', range(1, len(df) + 1))
```

### 2. Date Preprocessing

The **DATE** column is transformed into day, month, and year components using the `pd.to_datetime()` function. This step helps in capturing more granular temporal patterns in the dataset.

```python
df['Date'] = pd.to_datetime(df['DATE'], dayfirst=True)
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
```

Once the day, month, and year columns are created, the original **DATE** column is no longer needed and is dropped. The data is then sorted based on the newly created **Year**, **Month**, and **Day** columns to maintain chronological order.

```python
df.drop('DATE', axis=1, inplace=True)
df.sort_values(by=['Year', 'Month', 'Day'], inplace=True)
```

### 3. Handling Missing Data

The script uses the **K-Nearest Neighbors (KNN) Imputer** to handle missing values. This technique fills in missing data points by averaging the values of the nearest neighbors, based on other feature similarities.

```python
imputer = KNNImputer(n_neighbors=5)
df_train_imputed_numeric = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
```

After imputation, the data is sorted back to its original order using the serial number column, and the serial number is dropped from the DataFrame.

```python
df_train_imputed_numeric.sort_values(by='Serial Number', inplace=True)
df_train_imputed_numeric.drop('Serial Number', axis=1, inplace=True)
```

### 4. Saving Preprocessed Data

The imputed training data is saved as `knn_and_sorted_train.csv`, preserving the processed dataset for later use.

```python
df_train_imputed_numeric.to_csv('/content/knn_and_sorted_train.csv', index=False, index_label='INDEX')
```

### 5. Test Data Preprocessing

The test dataset undergoes the same preprocessing steps as the training data, including adding a serial number column, splitting the date into day, month, and year, and preparing it for model evaluation.

```python
df_test = pd.read_csv('/content/test.csv')
df_test.insert(0, 'Serial Number', range(1, len(df_test) + 1))
# Additional preprocessing similar to the training data
```

### 6. Feature Scaling

Before fitting the model, the numerical features are standardized using the **StandardScaler** from scikit-learn to ensure that each feature contributes equally to the model. This helps prevent features with larger values from dominating those with smaller values.

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 7. Splitting Data

The data is split into training and testing sets using the `train_test_split()` function. This allows for evaluating the model on unseen data to check its generalization capability.

```python
X_train, X_test, y_train, y_test = train_test_split(df_train_imputed_numeric, y, test_size=0.2, random_state=42)
```

### 8. Random Forest Model Training

The **Random Forest Regressor** is used as the machine learning model. It is a robust ensemble method that fits multiple decision trees on various sub-samples of the dataset and averages the predictions to improve accuracy and control overfitting.

```python
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
```

### 9. Model Evaluation

The model is evaluated using two key metrics:

- **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted values.
- **Mean Absolute Error (MAE)**: Measures the average absolute difference between actual and predicted values.

These metrics are computed using scikit-learn’s `mean_squared_error` and `mean_absolute_error` functions.

```python
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f'MSE: {mse}, MAE: {mae}')
```

### 10. Saving Processed Data and Results

Finally, the imputed and sorted training data is saved to a CSV file, and the model’s performance metrics (MSE and MAE) are printed.

```python
df_train_imputed_numeric.to_csv('knn_and_sorted_train.csv', index=False)
```
