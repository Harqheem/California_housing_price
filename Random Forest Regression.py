import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
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

# Find the best hyperparameters that could be used
params = {'min_samples_split' : [2,10, 30, 50, 100, 200, 300, 700] ,
          'max_depth': [2, 4, 8, 16, 32, 64, None],
          'n_estimators': [10, 50, 100, 500]}
random_search = RandomizedSearchCV(
                estimator=RandomForestRegressor(random_state=42),
                param_distributions=params,
                n_iter=20,
                cv=5,
                scoring="r2",
                n_jobs=-1,
                random_state=42)

random_search.fit(X_train, y_train)

# Model Training
print(f"Best Hyperparameters: {random_search.best_params_}") 
best_model = random_search.best_estimator_

# Check for accuracy
y_predict = best_model.predict(X_test)
model_accuracy = r2_score(y_test, y_predict)
print(f"The model accuracy is {model_accuracy:.2%}")