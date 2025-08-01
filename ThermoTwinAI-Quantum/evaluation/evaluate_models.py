# evaluation/evaluate_models.py
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import numpy as np

def evaluate_model(y_true, y_pred, name="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    corr, _ = pearsonr(y_true, y_pred)

    print(f"📌 {name} Evaluation")
    print(f"   MAE     = {mae:.6f}")
    print(f"   RMSE    = {rmse:.6f}")
    print(f"   Corr(R) = {corr:.4f}")
