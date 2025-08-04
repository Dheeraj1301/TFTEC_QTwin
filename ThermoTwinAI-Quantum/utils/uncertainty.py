import numpy as np
import torch


def mc_dropout_predict(model: torch.nn.Module, x: torch.Tensor, n_samples: int = 30):
    """Perform Monte Carlo Dropout predictions.

    Returns the mean prediction, lower/upper 95% confidence bounds and the
    predictive standard deviation.
    """

    device = next(model.parameters()).device
    x = x.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(model(x, mc_dropout=True).cpu().numpy())
    preds = np.stack(preds, axis=0)
    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)
    lower = mean_pred - 1.96 * std_pred
    upper = mean_pred + 1.96 * std_pred
    return mean_pred, lower, upper, std_pred
