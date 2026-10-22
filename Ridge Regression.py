import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import time


# Load the data
df = pd.read_csv('housing_cleaned.csv')

# Extract Features and Targets
X = df.drop(columns=["longitude", "latitude", "median_house_value"])
y = df['median_house_value']

# One hot encode values of ocean proximity
X = pd.get_dummies(X, columns=['ocean_proximity'], dtype=int)

# Split the train and the test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applying polynomial features to improve model's accuracy
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Values Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

# Model Training
alpha_candidates = np.logspace(-3, 3, 100)
model = RidgeCV(alphas=alpha_candidates, cv=5)
model.fit(X_train_scaled, y_train)

# Check for accuracy
y_predict = model.predict(X_test_scaled)
model_accuracy = r2_score(y_test, y_predict)
print(f"The model accuracy is {model_accuracy:.2%}")