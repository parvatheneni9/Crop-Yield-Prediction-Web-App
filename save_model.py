from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pickle
import pandas as pd

# Dummy data – replace with real dataset
data = pd.DataFrame({
    'Year': [2020, 2021, 2022],
    'Rainfall': [500, 450, 470],
    'Pesticides': [10, 15, 12],
    'Temperature': [27.0, 25.0, 26.0],
    'Area': [1000, 900, 950],
    'Crop': ['rice', 'maize', 'wheat'],
    'Yield': [3000, 2500, 2700]
})

X = data[['Year', 'Rainfall', 'Pesticides', 'Temperature', 'Area', 'Crop']]
y = data['Yield']

# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('crop', OneHotEncoder(handle_unknown='ignore'), ['Crop'])
    ],
    remainder='passthrough'  # Leave the rest of the columns untouched
)

# Create pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor())
])

# Train
model.fit(X, y)

# Save the model pipeline
with open('static/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print(" Model pipeline saved to static/model.pkl")


