# evaluation/evaluate_models.py
import numpy as np
import os

try:  # Optional dependency for plotting
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional
    plt = None


def evaluate_model(
    y_true,
    y_pred,
    name: str = "Model",
    plot: bool = False,
    lower=None,
    upper=None,
):
    """Print regression metrics, optionally plot and return them.

    When ``lower`` and ``upper`` bounds are provided, coverage and sharpness of
    the predictive interval are also reported.
    """

    # Ensure 1D arrays for stable metric calculations
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    # Metrics computed with numpy to avoid external dependencies
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + 1e-8
    r2 = 1 - ss_res / ss_tot
    if y_true.std() == 0 or y_pred.std() == 0:
        corr = 0.0
    else:
        corr = np.corrcoef(y_true, y_pred)[0, 1]

    print(f"📌 {name} Evaluation")
    print(f"   MAE     = {mae:.6f}")
    print(f"   RMSE    = {rmse:.6f}")
    print(f"   MAPE    = {mape:.6f}")
    print(f"   R²      = {r2:.4f}")
    print(f"   Corr(R) = {corr:.4f}")

    if lower is not None and upper is not None:
        lower = np.ravel(lower)
        upper = np.ravel(upper)
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        sharpness = np.mean(upper - lower)
        print(f"   Coverage = {coverage:.4f}")
        print(f"   Sharpness = {sharpness:.6f}")

    if plot and plt is not None:
        os.makedirs("plots", exist_ok=True)
        plt.figure()
        plt.plot(y_true, label="True")
        plt.plot(y_pred, label="Predicted")
        if lower is not None and upper is not None:
            plt.fill_between(
                np.arange(len(y_true)), lower, upper, color="gray", alpha=0.2
            )
        plt.legend()
        plt.title(name)
        plt.tight_layout()
        fname = f"plots/{name.replace(' ', '_').lower()}_pred.png"
        plt.savefig(fname)
        plt.close()
        print(f"   Plot saved to {fname}")

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "R2": float(r2),
        "Corr": float(corr),
    }
