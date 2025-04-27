# src/train.py

import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('project csv new.csv')  # <-- Your dataset filename

# Define Features (X) and Target (y)
X = df.drop('shelf_life', axis=1)
y = df['shelf_life']

# Preprocessing: Scale and Encode
numeric_features = ['storage_temp', 'storage_humid']
categorical_features = ['material_type', 'product_state', 'storage_cond']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(), categorical_features)
])

# Model Definition
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42))
])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set MLflow tracking
mlflow.set_tracking_uri("file:///tmp/mlruns")
mlflow.set_experiment("Shelf_Life_Predictor")

with mlflow.start_run():
    # Train the model
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Calculate Metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Log metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)

    # Log the trained model
    mlflow.sklearn.log_model(model, "model")

    print(f"✅ Model saved in run {mlflow.active_run().info.run_id}")
