Solar Power Generation Prediction System
A machine learning system that predicts solar radiation (W/m²) using historical weather data, built with XGBoost.

Problem Statement
Solar radiation is the primary driver of solar power generation, but it's highly variable and depends on time of day, season, and weather conditions. This project builds a regression model to predict solar radiation from weather features, which can serve as a key input for solar power output forecasting.

Dataset
Source: NREL Solar Radiation Prediction dataset (Kaggle)
32,686 readings taken at ~5 minute intervals over 4 months
Features: temperature, pressure, humidity, wind speed/direction, sunrise/sunset times
Target: solar radiation (W/m²)
Approach
Data Cleaning: Parsed date/time fields, verified no missing values
Feature Engineering:
Cyclical encoding of hour and day-of-year (sin/cos transforms) to capture daily and seasonal cycles correctly
Daytime indicator flag from sunrise/sunset times
Temperature × Humidity interaction term
Model: XGBoost Regressor
Validation: Time-based train/test split (80/20) — no shuffling, to simulate real forecasting conditions
Tuning: Hyperparameter search using TimeSeriesSplit cross-validation (to avoid data leakage from random shuffling)
Results
RMSE: 132.81 W/m²
R² Score: 0.75
Feature importance analysis showed that time-of-day (encoded cyclically) is by far the strongest predictor (~80% importance), followed by humidity (~8%). This aligns with the physical reality that solar radiation is fundamentally driven by the sun's position in the sky.

Limitations & Honest Notes
This model predicts instantaneous radiation from weather conditions alone (no use of recent radiation history), which caps achievable accuracy since short-term cloud cover and atmospheric noise aren't fully captured by point-in-time weather readings.
R² of ~0.75 represents a realistic ceiling for this feature set; adding recent radiation trends (lag features) would likely improve accuracy further but changes the problem framing to short-term forecasting rather than weather-based prediction.
Tech Stack
Python, pandas, NumPy
XGBoost, scikit-learn
Matplotlib (visualization)
How to Run
Clone this repo
Install dependencies: pip install -r requirements.txt
Open solar_prediction.ipynb in Jupyter or Colab
Run all cells in order
Files
solar_prediction.ipynb — full analysis and model training notebook
solar_radiation_model.pkl — saved trained model
SolarPrediction.csv — dataset used
