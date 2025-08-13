from typing import Optional

import numpy as np


def augment_time_series(
    df: np.ndarray,
    add_noise: bool = True,
    add_scale: bool = True,
    add_seasonal: bool = True,
    add_warp: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Augment a multivariate time series using several techniques.

    Parameters
    ----------
    df:
        Array of shape ``(n_samples, n_features)`` containing the time
        series to augment. The first column is assumed to be the primary
        target feature (e.g. CoP).
    add_noise, add_scale, add_seasonal, add_warp:
        Toggles controlling the respective augmentation techniques.
    seed:
        Optional seed for deterministic behaviour.

    Returns
    -------
    np.ndarray
        Augmented time series including the original samples.
    """

    rng = np.random.default_rng(seed)
    data = np.array(df, dtype=float)
    augmented = [data]

    length = len(data)
    if length < 2:
        return data

    # Window slicing: generate overlapping sub-sequences from the original
    slice_len = max(8, length // 3)
    step = max(1, slice_len // 2)
    for start in range(0, length - slice_len + 1, step):
        window = data[start : start + slice_len].copy()

        if add_noise:
            amp = window.max(axis=0) - window.min(axis=0)
            sigma = rng.uniform(0.01, 0.03)
            noise = rng.normal(0, sigma * amp, size=window.shape)
            window += noise

        if add_scale:
            scale = rng.uniform(0.95, 1.05)
            window *= scale

        if add_seasonal:
            t = np.linspace(0, 2 * np.pi, len(window))
            amp = 0.02 * (window.max(axis=0) - window.min(axis=0))
            bias = amp * np.sin(t)[:, None]
            # Apply bias to a random half of the window
            s = rng.integers(0, len(window) // 2)
            e = s + len(window) // 2
            window[s:e] += bias[s:e]

        if add_warp:
            orig = np.linspace(0, 1, len(window))
            warp = orig + rng.normal(0, 0.02, size=len(window))
            warp = np.clip(np.sort(warp), 0, 1)
            for c in range(window.shape[1]):
                window[:, c] = np.interp(orig, warp, window[:, c])

        augmented.append(window)

    return np.vstack(augmented)
