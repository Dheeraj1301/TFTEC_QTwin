# evaluation/evaluate_models.py
import json
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
    scaler: tuple[float, float] | None = None,
):
    """Print regression metrics, optionally plot and return them.

    When ``lower`` and ``upper`` bounds are provided, coverage and sharpness of
    the predictive interval are also reported.
    """

    # Ensure 1D arrays for stable metric calculations
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    if scaler is not None:
        y_min, y_max = scaler
        scale = y_max - y_min
        y_true_plot = y_true * scale + y_min
        y_pred_plot = y_pred * scale + y_min
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred

    # Metrics computed with numpy to avoid external dependencies
    diff = y_true_plot - y_pred_plot
    abs_diff = np.abs(diff)
    sq_diff = diff * diff

    mae = abs_diff.mean()
    rmse = np.sqrt(sq_diff.mean())
    mape = (abs_diff / (np.abs(y_true_plot) + 1e-8)).mean()
    smape = (
        2.0 * abs_diff / (np.abs(y_true_plot) + np.abs(y_pred_plot) + 1e-8)
    ).mean()

    ss_res = sq_diff.sum()
    y_true_mean = y_true_plot.mean()
    ss_tot = np.sum((y_true_plot - y_true_mean) ** 2) + 1e-8
    r2 = max(0.0, 1 - ss_res / ss_tot)

    std_true = y_true_plot.std()
    std_pred = y_pred_plot.std()
    if std_true == 0 or std_pred == 0:
        corr = 0.0
    else:
        corr = abs(
            np.dot(y_true_plot - y_true_mean, y_pred_plot - y_pred_plot.mean())
            / (y_true_plot.size * std_true * std_pred)
        )

    print(f"📌 {name} Evaluation")
    print(f"   MAE     = {mae:.6f}")
    print(f"   RMSE    = {rmse:.6f}")
    print(f"   MAPE    = {mape:.6f}")
    print(f"   SMAPE   = {smape:.6f}")
    print(f"   R²      = {r2:.4f}")
    print(f"   Corr(R) = {corr:.4f}")

    if lower is not None and upper is not None:
        lower = np.ravel(lower)
        upper = np.ravel(upper)
        if scaler is not None:
            lower = lower * scale + y_min
            upper = upper * scale + y_min
        coverage = np.mean((y_true_plot >= lower) & (y_true_plot <= upper))
        sharpness = np.mean(upper - lower)
        print(f"   Coverage = {coverage:.4f}")
        print(f"   Sharpness = {sharpness:.6f}")

    if plot and plt is not None:
        os.makedirs("plots", exist_ok=True)
        plt.figure()
        plt.plot(y_true_plot, label="True")
        plt.plot(y_pred_plot, label="Predicted")
        if lower is not None and upper is not None:
            plt.fill_between(
                np.arange(len(y_true_plot)), lower, upper, color="gray", alpha=0.2
            )
        plt.legend()
        plt.title(name)
        plt.tight_layout()
        fname = f"plots/{name.replace(' ', '_').lower()}_pred.png"
        plt.savefig(fname)
        plt.close()
        print(f"   Plot saved to {fname}")

    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "SMAPE": float(smape),
        "R2": float(r2),
        "Corr": float(corr),
    }

    os.makedirs("results", exist_ok=True)
    try:
        if os.path.exists("results/metrics.json"):
            with open("results/metrics.json", "r", encoding="utf-8") as f:
                all_metrics = json.load(f)
        else:
            all_metrics = {}
    except Exception:
        all_metrics = {}
    all_metrics[name] = metrics
    with open("results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    return metrics


def evaluate_acga(acga, save: bool = True, name: str = "acga_attention"):
    """Print and optionally save the latest attention matrix from ``acga``."""

    attn = getattr(acga, "attention_matrix", lambda: None)()
    if attn is None:
        print("⚠️  ACGA has not produced attention weights yet")
        return None
    mat = attn.detach().cpu().numpy()
    print("📌 ACGA Attention Matrix")
    print(mat)
    if save:
        os.makedirs("plots", exist_ok=True)
        fname = f"plots/{name}.csv"
        np.savetxt(fname, mat, delimiter=",")
        print(f"   Matrix saved to {fname}")
    return mat
