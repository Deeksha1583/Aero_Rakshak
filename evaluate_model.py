import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

test = pd.read_csv("Dataset/PM_test.csv")
truth = pd.read_csv("Dataset/PM_truth.csv")

print("Test Shape:", test.shape)
print("Truth Shape:", truth.shape)

# Load model
xgb_model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

test.columns = test.columns.str.strip()
test_sorted = test.sort_values(['id', 'cycle'])

# Feature engineering
sensor_cols = [col for col in test_sorted.columns if col.startswith('s')]

for col in sensor_cols:
    test_sorted[col+"_mean"] = test_sorted.groupby('id')[col].rolling(10).mean().reset_index(0,drop=True)
    test_sorted[col+"_std"] = test_sorted.groupby('id')[col].rolling(10).std().reset_index(0,drop=True)
    test_sorted[col+"_diff"] = test_sorted.groupby('id')[col].diff()

# test_sorted['cycle_norm'] = test_sorted['cycle'] / test_sorted.groupby('id')['cycle'].transform('max')

test_sorted.bfill(inplace=True)
test_last = test_sorted.groupby('id').last().reset_index()

X_test = test_last.reindex(columns=feature_columns, fill_value=0)
X_test_scaled = scaler.transform(X_test)

# Predict
final_pred = xgb_model.predict(X_test_scaled)
final_pred = np.clip(final_pred, 0, 125)

y_true = truth.iloc[:, 1].values

# Evaluation
rmse = np.sqrt(np.mean((y_true - final_pred)**2))

print("\n----Test Performance----")
print("MAE:", mean_absolute_error(y_true, final_pred))
print("RMSE:", rmse)
print("R2:", r2_score(y_true, final_pred))

# Sample Output
results = pd.DataFrame({
    "Engine_ID": test_last['id'],
    "Actual_RUL": y_true,
    "Predicted_RUL": final_pred
})

print("\nSample Predictions:")
print(results.head(10))