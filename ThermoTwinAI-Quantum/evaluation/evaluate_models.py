# evaluation/evaluate_models.py
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
import os

def evaluate_model(y_true, y_pred, name="Model", plot=False):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    corr, _ = pearsonr(y_true, y_pred)

    print(f"📌 {name} Evaluation")
    print(f"   MAE     = {mae:.6f}")
    print(f"   RMSE    = {rmse:.6f}")
    print(f"   Corr(R) = {corr:.4f}")

    if plot:
        os.makedirs("plots", exist_ok=True)
        plt.figure()
        plt.plot(y_true, label="True")
        plt.plot(y_pred, label="Predicted")
        plt.legend()
        plt.title(name)
        plt.tight_layout()
        fname = f"plots/{name.replace(' ', '_').lower()}_pred.png"
        plt.savefig(fname)
        plt.close()
        print(f"   Plot saved to {fname}")
