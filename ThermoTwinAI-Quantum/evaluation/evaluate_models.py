# evaluation/evaluate_models.py
import numpy as np
import os

try:  # Optional dependency for plotting
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional
    plt = None


def evaluate_model(y_true, y_pred, name: str = "Model", plot: bool = False):
    """Print common regression metrics and optionally plot predictions."""

    # Ensure 1D arrays for stable metric calculations
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    # Metrics computed with numpy to avoid external dependencies
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    if y_true.std() == 0 or y_pred.std() == 0:
        corr = 0.0
    else:
        corr = np.corrcoef(y_true, y_pred)[0, 1]

    print(f"📌 {name} Evaluation")
    print(f"   MAE     = {mae:.6f}")
    print(f"   RMSE    = {rmse:.6f}")
    print(f"   Corr(R) = {corr:.4f}")

    if plot and plt is not None:
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
