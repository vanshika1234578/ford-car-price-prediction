# Ford Car Price Prediction

A machine learning project that predicts the used-car selling price of Ford vehicles from their specifications, using Linear Regression and comparing two categorical encoding methods.

## Dataset
`ford.csv` has 17,966 records and 9 columns (from the Kaggle "Ford Car Price Prediction" dataset by adhurimquku).

| Column | Description |
|--------|-------------|
| model | Ford model (Fiesta, Focus, ...) |
| year | Year of registration |
| price | Selling price (target variable) |
| transmission | Manual, Automatic, Semi-Auto |
| mileage | Distance driven |
| fuelType | Petrol, Diesel, Hybrid, ... |
| tax | Road tax |
| mpg | Miles per gallon |
| engineSize | Engine size in litres |

## Workflow
The full analysis is in `ford_car_price_prediction.ipynb`.

1. **EDA**: value counts, price distribution, correlation heatmap, box plots of price against year, engine size, transmission, fuel type and model
2. **Encoding**: two approaches compared. One-hot encoding of `model`, `transmission` and `fuelType`, and label encoding of the same columns
3. **Scaling**: `StandardScaler` on the numeric columns
4. **Modelling**: Linear Regression, 67/33 train/test split (`random_state=42`)

## Results
| Encoding | MAE | RMSE | R-squared |
|----------|-----|------|-----------|
| **One-hot** | **1,377** | **1,903** | **0.840** |
| Label | 1,765 | 2,469 | 0.731 |

One-hot encoding gives the more accurate model, at the cost of many more columns.

## Tech stack
Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, Jupyter

## How to run
```bash
git clone https://github.com/vanshika1234578/ford-car-price-prediction.git
cd ford-car-price-prediction
pip install -r requirements.txt
jupyter notebook ford_car_price_prediction.ipynb
```
Keep `ford.csv` in the same folder as the notebook.
