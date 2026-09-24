# Gold Price Forecasting with Machine Learning & Macroeconomic Data

## Overview

A Python machine learning project for **next-day gold price forecasting** using historical gold prices and macroeconomic/financial indicators, including the U.S. Dollar Index, Treasury yield, EFFR, CPI, PCE, and GDP.

The pipeline builds lag, rolling, return, and calendar features, then compares XGBoost, Random Forest, LightGBM, and a validation-error-weighted ensemble using chronological training, validation, and test splits. Models predict the next-day logarithmic return, which is converted into a gold price forecast and evaluated with MAE, RMSE, and R².

## Data

The main variables used in the project include:

- **Gold** — Historical gold price
- **USD** — U.S. Dollar Index
- **Yield** — U.S. 10-Year Treasury Yield
- **EFFR** — Effective Federal Funds Rate
- **CPI** — Consumer Price Index
- **PCE** — Personal Consumption Expenditures
- **GDP** — Gross Domestic Product

The merged dataset contains daily observations from **September 2001 to December 2025**.

## Feature Engineering

The project generates multiple time-series features, including:

### Lag Features
- Gold price lags: 1, 3, 5, 10, 20, and 60 days
- USD Index lags
- Treasury Yield lags
- EFFR, CPI, PCE, and GDP lags

### Rolling Statistics
- Moving averages over 5, 10, 20, 60, and 120 days
- Rolling standard deviations
- USD and Treasury Yield moving averages

### Return and Change Features
- Daily and multi-day gold returns
- USD returns
- Treasury Yield changes
- EFFR changes
- CPI, PCE, and GDP changes

### Momentum Features
- 20-day and 60-day gold momentum
- Gold price relative to moving averages

### Time Features
- Year
- Month
- Quarter
- Day of week

The prediction target is defined as the **next-day logarithmic return**:

`Target = log(Gold(t+1) / Gold(t))`

The predicted return is then converted back into the next-day gold price.

## Models

Three machine learning regression models are trained and compared:

- **XGBoost Regressor**
- **Random Forest Regressor**
- **LightGBM Regressor**

An additional **Adaptive Error Weighted Ensemble (AEWE)** combines predictions from the three models.

The ensemble weights are calculated using the inverse of each model's validation MAE.

## Dataset Split

To preserve the temporal structure of the data, the dataset is split chronologically:

| Dataset | Period |
|---|---|
| Training | 2001–2022 |
| Validation | 2023 |
| Testing | 2024–2025 |

No random train-test split is used.

## Results

Performance is evaluated using **MAE, RMSE, and R²** on the predicted gold price.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| XGBoost | 28.37 | 38.52 | 0.9965 |
| Random Forest | 66.40 | 72.79 | 0.9874 |
| **LightGBM** | **24.43** | **34.64** | **0.9971** |
| AEWE | 36.11 | 44.49 | 0.9953 |

Among the evaluated models, **LightGBM achieved the best test-set performance**, with an MAE of approximately **24.43** and an R² of approximately **0.9971**.

## AEWE Ensemble

The AEWE method assigns weights according to validation error:

| Model | Weight |
|---|---:|
| XGBoost | 0.3490 |
| Random Forest | 0.2826 |
| LightGBM | 0.3684 |

Although the ensemble provides a combined prediction from all three models, LightGBM individually achieved the strongest performance on the current test set.

## Project Structure

```text
gold-price-forecasting/
│
├── data.py
├── feature.py
├── train.py
│
├── total.csv
├── total_featured.csv
│
├── gold_daily_filled.csv
├── dollar_index.csv
├── Yield.csv
├── EFFR.csv
├── CPI.csv
├── PCE.csv
├── GDP.csv
│
├── validation_results.csv
├── single_model_test_results.csv
├── aewe_weights.csv
├── aewe_test_predictions_price.csv
└── model_results_summary_with_aewe.csv
```

### Main Scripts

**`data.py`**  
Performs preprocessing and daily-date completion for economic data.

**`feature.py`**  
Creates lag, rolling, return, momentum, and time-based features and generates the next-day prediction target.

**`train.py`**  
Trains XGBoost, Random Forest, and LightGBM models, calculates AEWE ensemble weights, evaluates model performance, and exports prediction results.

## Installation

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm
```

## How to Run

After preparing `total.csv`, run:

```bash
python feature.py
```

This generates:

```text
total_featured.csv
```

Then train and evaluate the models:

```bash
python train.py
```

The script generates model evaluation results, ensemble weights, and test-set predictions.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- Time-Series Feature Engineering
- Machine Learning Regression

## Key Highlights

- Integrated gold price data with multiple macroeconomic and financial indicators
- Engineered lag, rolling, return, volatility, and momentum features
- Used chronological train/validation/test splits to preserve time-series order
- Compared three tree-based machine learning models
- Implemented an adaptive error-weighted ensemble strategy
- Achieved a test **R² of 0.9971** with LightGBM

## Author

**Rui Tian**

Master's Student, Southern Methodist University