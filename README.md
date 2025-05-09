# Crop Yield Prediction Web App

A Flask-based machine learning web application that predicts agricultural crop yield based on key environmental and farming parameters.

##  Features

  - Predict crop yield using a trained Decision Tree Regressor.
  - Input parameters include:
  - Year
  - Average Rainfall (mm/year)
  - Pesticides Used (Tonnes)
  - Average Temperature (°C)
  - Cultivation Area (sq km)
  - Crop Type
  - Clean and responsive user interface.
  - Model saved and loaded via `pickle`.

## Tech Stack

- Backend: Flask
- Frontend: HTML, CSS
- Machine Learning: scikit-learn (Decision Tree Regressor, OneHotEncoder, Pipeline)
- Model Serialization: `pickle`



