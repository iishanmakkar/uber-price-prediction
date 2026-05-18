# Uber Fare Prediction Using Machine Learning

## Overview

This project predicts Uber ride fares using Machine Learning with a Linear Regression model.  
The model is trained on Uber ride data and predicts fare amount based on travel distance.

The project demonstrates:
- Data preprocessing
- Feature engineering
- Data visualization
- Linear Regression
- Model evaluation
- Fare prediction
- Model saving

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

---

# Dataset

The dataset contains Uber ride details such as:
- Fare Amount
- Pickup Latitude
- Pickup Longitude
- Dropoff Latitude
- Dropoff Longitude

Dataset file used:
```bash
uber.csv
```

---

# Project Workflow

## 1. Import Libraries

Required libraries are imported for:
- Data analysis
- Visualization
- Machine Learning

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

---

## 2. Load Dataset

The dataset is loaded using Pandas.

```python
df = pd.read_csv('uber.csv')
```

---

## 3. Data Cleaning

The dataset is cleaned by:
- Removing missing values
- Removing invalid fare amounts

```python
df = df.dropna()
df = df[df['fare_amount'] > 0]
```

---

## 4. Feature Engineering

A new feature called `distance` is created using pickup and dropoff coordinates.

```python
def distance(lat1, lon1, lat2, lon2):
    return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

df['distance'] = distance(
    df['pickup_latitude'],
    df['pickup_longitude'],
    df['dropoff_latitude'],
    df['dropoff_longitude']
)
```

---

## 5. Train-Test Split

The dataset is divided into:
- 80% Training Data
- 20% Testing Data

```python
X = df[['distance']]
y = df['fare_amount']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## 6. Model Training

A Linear Regression model is trained using the training dataset.

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

---

## 7. Data Visualization

The project includes:
- Histogram of fare distribution
- Scatter plot of distance vs fare
- Regression prediction graph

Example visualization:

```python
plt.scatter(df['distance'], df['fare_amount'])
plt.xlabel("Distance")
plt.ylabel("Fare Amount")
plt.title("Distance vs Fare")
plt.show()
```

---

## 8. Fare Prediction

The trained model predicts Uber fare using travel distance.

```python
distance = float(input("Enter distance in km: "))

predicted_fare = model.predict([[distance]])

print("Predicted fare:", predicted_fare[0])
```

### Example Output

```bash
Enter distance in km: 5
Predicted fare: 14.25
```

---

## 9. Model Evaluation

The model is evaluated using:
- RMSE (Root Mean Squared Error)
- R² Score

```python
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2 Score:", r2)
```

---

## 10. Save Model

The trained model is saved using Joblib.

```python
import joblib

joblib.dump(model, 'linear_regression_model.pkl')
```

Saved model file:

```bash
linear_regression_model.pkl
```

---

# Project Structure

```bash
├── uber.csv
├── Untitled18.ipynb
├── linear_regression_model.pkl
└── README.md
```

---

# How to Run the Project

## Step 1: Install Dependencies

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

---

## Step 2: Run Jupyter Notebook

```bash
jupyter notebook
```

Open:
```bash
Untitled18.ipynb
```

Run all cells.

---

# Learning Outcomes

This project helps understand:
- Data Cleaning
- Feature Engineering
- Linear Regression
- Machine Learning Workflow
- Data Visualization
- Model Evaluation

---

# Future Improvements

Possible enhancements:
- Add more ride features
- Use advanced regression algorithms
- Deploy as web application
- Add real-time fare prediction

---

# Author

Developed using Python and Machine Learning.
