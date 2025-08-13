"""Feature importance analysis using SHAP."""

from __future__ import annotations

import numpy as np
import torch


def compute_feature_importance(model: torch.nn.Module, X: np.ndarray, top_k: int | None = None):
    """Compute feature importance for the classical portion of ``model``.

    Parameters
    ----------
    model:
        Trained PyTorch model.
    X:
        Input data as a NumPy array of shape ``(n_samples, seq, features)``.
    top_k:
        Optionally return only the top ``k`` features.
    """

    try:  # pragma: no cover - optional dependency
        import shap
    except Exception as exc:  # pragma: no cover - analysis is optional
        raise ImportError("SHAP is required for feature importance analysis") from exc

    model.eval()
    background = torch.tensor(X[: min(100, len(X))], dtype=torch.float32)
    explainer = shap.DeepExplainer(model, background)
    shap_vals = explainer.shap_values(background)[0]  # (samples, features)
    importance = np.mean(np.abs(shap_vals), axis=0)
    ranking = np.argsort(-importance)

    if top_k is not None:
        ranking = ranking[:top_k]
        importance = importance[ranking]

    return ranking, importance


if __name__ == "__main__":  # pragma: no cover - manual use
    print("Run from a notebook or script to analyse feature importance.")
