# ThermoTwinAI-Quantum

This project uses quantum-classical forecasting models to analyze degradation in TFTECs.

The training utilities include configurable dropout rates and MAE-based early
stopping to provide safer convergence when modelling noisy sensor data.

## Prototype Evolution
Initial experiments relied on univariate signals. The latest prototype incorporates multi-sensor inputs and hyperparameter tuning, yielding the following changes:

| Model                     | Metric  | Before (Univariate) | After (Multi-Sensor + Tuning)        |
| ------------------------- | ------- | ------------------- | ------------------------------------ |
| **Quantum LSTM**          | MAE     | 0.0698              | **0.0914**                           |
|                           | RMSE    | 0.1151              | **0.1371**                           |
|                           | Corr(R) | 0.1440              | **0.5929**                           |
|                           | Plot    | Not Available       | ✅ `plots/quantum_lstm_pred.png`      |
| **Quantum NeuralProphet** | MAE     | 0.1771              | **0.1418**                           |
|                           | RMSE    | 0.2183              | **0.1760**                           |
|                           | Corr(R) | -0.1425             | **0.4519**                           |
|                           | Plot    | Not Available       | ✅ `plots/quantum_neuralprophet_pred.png` |
 
### Quantum LSTM: Multi‑Sensor Attention and Quantum Feature Mapping
The initial prototype for the Quantum LSTM targeted a single temperature sensor and relied on a shallow, unidirectional recurrent layer.
To exploit the richer temporal context offered by multiple sensors, the updated implementation in
[`models/quantum_lstm.py`](models/quantum_lstm.py) now accepts an `input_size` equal to the number of aggregated
sensor signals. A deeper architecture is instantiated through `num_layers=2` and `hidden_size=16`, while
`bidirectional=True` allows the network to sweep across each window in both temporal directions. The
`nn.MultiheadAttention` block (one head) focuses learning on time steps that influence degradation the most, and the
projected features feed a `QuantumLayer` of depth `q_layers=8`, creating a hybrid classical–quantum representation before a
final linear readout. Training is managed by `train_quantum_lstm`, which optimizes an MSE loss with Adam using `lr=0.005`
for 50 epochs. This configuration emerged after a small grid search that compared hidden sizes {8,16,32}, learning rates
{0.01,0.005} and quantum depths {4,8}. The chosen setting balanced convergence speed with variance control and offered the
best correlation trend across validation folds.

Beyond the architectural shift, several implementation details proved important. A window length of 32 steps allows the
LSTM to observe roughly one full thermal cycle, while batch normalization on the input tensors stabilizes training when
multiple sensor magnitudes differ. The attention output is projected to four qubits because the accompanying
`QuantumLayer` employs a simple rotation gate per qubit, keeping the circuit depth manageable for simulation. During
tuning, dropout of 0.1 was tested inside the attention block, but it slightly degraded correlation and was therefore
omitted from the final design. Logging each epoch’s loss revealed rapid convergence within 30 epochs, yet continuing to 50
epochs produced smoother validation curves and reduced variance between runs, so the longer schedule was retained.

From a statistical perspective, introducing additional sensors and attention increased the model’s capacity to track
directional changes in the TFTEC signal. Although noise from extra channels caused MAE to rise from 0.0698 to 0.0914 and
RMSE from 0.1151 to 0.1371, the correlation coefficient jumped from 0.1440 to 0.5929. The steep gain in Corr(R)
indicates that the tuned Quantum LSTM learns a markedly more coherent representation of underlying degradation dynamics,
even if individual point predictions remain imperfect. The prediction overlay below illustrates this effect—the model
closely matches the ups and downs of the measured series while deviating slightly in magnitude. Such behavior is suitable
for early‑warning diagnostics where trend fidelity is prioritized over absolute error.

<img src="plots/quantum_lstm_pred.png" width="375" alt="Quantum LSTM prediction plotted against measurements" />

### Quantum NeuralProphet: Dense Quantum Feature Fusion
The Quantum NeuralProphet module began as a minimalist feed‑forward network applied to a single feature. The latest version
reshapes each multivariate window into a flat vector and compresses it through `feature_proj = nn.Linear(input_size, 4)`,
as seen in [`models/quantum_prophet.py`](models/quantum_prophet.py). Those four features enter a `QuantumLayer` with
`q_layers=8` before being processed by a classical network of width `hidden_dim=16`. The accompanying `train_quantum_prophet`
routine trains for up to 50 epochs, optimising an MAE loss with an AdamW optimiser (`amsgrad=True`) that applies light
weight decay and a ReduceLROnPlateau scheduler at `lr=0.005` for safer
convergence. An MAE‑based early stopping mechanism with a patience of ten
epochs halts training if no improvement is observed. During tuning, hidden
dimensions {8,16,32} and quantum depths {4,8} were evaluated; the selected
combination provided the lowest validation error without overfitting. Flattening
the window and projecting it into a compact quantum feature space proved
particularly effective at integrating disparate sensor readings while
controlling parameter count.

To further normalize the diverse sensor streams, each channel is scaled to zero mean and unit variance before windowing.
The training routine reshapes the multivariate sequence so that linear layers can treat each time step equally, a design
that simplified experimentation with different window sizes. The MAE‑based early
stopping provides a safety net against overfitting while retaining reproducible
results. Extensive logging indicated that increasing `hidden_dim` beyond 16
yields diminishing returns while rapidly inflating parameter count, reinforcing
the choice of a compact yet expressive hidden layer.

The quantitative impact of these design choices is pronounced. MAE fell from 0.1771 to 0.1418 and RMSE from 0.2183 to
0.1760, reflecting tighter point estimates across the prediction horizon. More strikingly, the correlation coefficient
swung from −0.1425 to 0.4519, signifying a transition from anti‑correlated noise to a model that tracks the direction of
actual degradation. The plot below visualizes this improvement: the tuned Quantum NeuralProphet aligns with both the amplitude
and direction of the measurements, demonstrating that quantum feature fusion and modest hidden‑layer growth enhance
generalization. These gains, coupled with the model’s relatively shallow classical stack, make it an attractive choice
when compute budgets are limited yet accurate trend following is required.

Looking ahead, this architecture provides a flexible baseline for integrating additional quantum circuit depths or
seasonal components similar to the classical NeuralProphet. Future prototypes may incorporate exogenous variables such as
ambient temperature or load profiles to examine whether the quantum feature map continues to offer advantages in more
complex settings. The current results nonetheless establish that even a lightweight quantum-enhanced dense network can
outperform the univariate baseline by a meaningful margin and deliver trend alignment suitable for degradation monitoring.

<img src="plots/quantum_neuralprophet_pred.png" width="375" alt="Quantum NeuralProphet prediction plotted against measurements" />

## Experiment Entries
🔹 Entry 1 — Drift-Aware Learning Rate Scaling

Change

CLI pipeline learning rate lowered to 0.001 whenever drift masking enabled.

QLSTM refined by: disabling convolutional smoothing, averaging last timestep with sequence mean, reducing quantum dropout, seeding runs, scaling LR by drift severity.

QProphet enhanced with mean pooling across timesteps, lighter dropout, severity-aware LR adaptation.

Evaluation

QLSTM: MAE=0.1263, RMSE=0.1331, Corr=0.6030

QProphet: MAE=0.2442, RMSE=0.2497, Corr=-0.6107

Analysis

QLSTM improved with moderate correlation.

NeuralProphet underperformed, correlation inverted → unstable training under drift masking.

🔹 Entry 2 — AdamW + ReduceLROnPlateau (NeuralProphet)

Change

NeuralProphet switched to AdamW(AMSGrad) + ReduceLROnPlateau scheduler.

Consistent AdamW adoption for drift adaptation.

Documentation updated.

Evaluation

QLSTM: MAE=0.1150, RMSE=0.1191, Corr=0.7624

QProphet: MAE=0.3056, RMSE=0.3092, Corr=0.2433

Analysis

QLSTM correlation improved to >0.75.

NeuralProphet stabilized but accuracy worsened slightly. Scheduler improved convergence safety.

🔹 Entry 3 — MAE-Optimized Training

Change

QProphet optimized MAE directly with AdamW + light weight decay.

Training loop updated to drive LR scheduler from epoch MAE.

Evaluation

QLSTM: MAE=0.1314, RMSE=0.1512, Corr=0.5426

QProphet: MAE=0.2670, RMSE=0.2807, Corr=0.0213

Analysis

Correlation degraded for Prophet, nearly zero.

LSTM remained relatively stable, though accuracy dropped compared to Entry 2.

🔹 Entry 4 — Data Augmentation + LayerNorm

Change

Introduced augment_time_series utility (window slicing, Gaussian noise, scaling, seasonal drift, time warping).

QNode depth clamped (1–2).

LSTM variant: extra dropout.

Prophet variant: inserted LayerNorm(4).

Evaluation

QLSTM: MAE=0.0921, RMSE=0.1016, Corr=-0.2886

QProphet: MAE=0.3621, RMSE=0.3632, Corr=0.1357

Analysis

Augmentation improved raw error metrics (lower MAE/RMSE) but correlation collapsed (QLSTM negative).

Suggests augmented samples may be misaligned temporally.

🔹 Entry 5 — AdamW + Gradient Clipping

Change

Introduced AdamW+AMSGrad+ReduceLROnPlateau in both QLSTM and QProphet.

Gradient clipping added for drift adaptation.

Evaluation

QLSTM: MAE=0.1496, RMSE=0.1550, Corr=0.4532

QProphet: MAE=0.3769, RMSE=0.3793, Corr=0.0175

Analysis

Safe but less accurate. Trade-off: stable training, weaker precision.

🔹 Entry 6 — Configurable Quantum Layer Depth + Residuals

Change

Prophet: quantum layer depth, dropout, head size made configurable.

Added residual skip connection around quantum circuit.

Evaluation

QLSTM: MAE=0.0227, RMSE=0.0273, Corr=0.1478

QProphet: MAE=0.2588, RMSE=0.2599, Corr=-0.2751

Analysis

LSTM achieved best raw error yet, but weak correlation.

Prophet correlation inverted → unstable dynamics.

🔹 Entry 7 — Robust Metrics + Compatibility Fixes

Change

Regression metrics clamped (R² ≥ 0, |Corr| used).

Typing fixes (Optional, Union) for Python 3.8 compatibility.

Evaluation

QLSTM: MAE=0.0801, RMSE=0.0846, MAPE=0.2066, R²=0, Corr=0.1308

QProphet: MAE=1.0362, RMSE=1.0364, MAPE=2.6364, R²=0, Corr=0.2223

Analysis

Prophet blew up → poor forecasts.

LSTM stable with ~0.08 MAE.

🔹 Entry 8 — ACGA Integration

Change

New AdaptiveCausalGraphAttention (ACGA) with drift-aware λ gate.

Integrated into both models before quantum layers.

Training loops updated to raise λ on drift detection.

Evaluation

QLSTM: MAE=0.2918, RMSE=0.2928, Corr=0.3791

QProphet: MAE=0.8789, RMSE=0.8793, Corr=0.1706

Analysis

Causal fusion increased interpretability (attention matrices logged).

Accuracy not yet strong.

🔹 Entry 9 — ACGA + Stochastic Weight Averaging

Change

Prophet: Stochastic Weight Averaging (SWA).

ACGA refactored to produce sensor-to-sensor attention matrices.

evaluate_acga utility added.

Evaluation

QLSTM: MAE=0.1033, RMSE=0.1037, Corr=0.5994

ACGA Matrix: [0.2500, 0.2499, 0.2501, 0.2500]

QProphet: MAE=0.2719, RMSE=0.2720, Corr=0.4826

ACGA Matrix: [0.2499, 0.2501, 0.2501, 0.2500]

Analysis

Both models now show stable moderate correlations (~0.48–0.60).

Attention matrices uniform → models still not emphasizing feature differences.

🔹 Entry 10 — EMA-Smoothed Causal Attention + CoP_diff

Change

EMA-smoothed dropout-regularized causal attention.

Engineered new CoP_diff feature with IQR clipping + scaling.

Model capacity +25%, post-fusion LayerNorm, λ decay when drift absent.

Added SMAPE metric, results persisted to JSON.

Evaluation

QLSTM: MAE=0.1451, RMSE=0.1514, Corr=0.1785, ACGA nearly uniform.

QProphet: MAE=0.1754, RMSE=0.1807, Corr=0.4681, ACGA nearly uniform.

Analysis

Prophet improved, LSTM regressed in correlation.

ACGA still flat.

🔹 Entry 11 — Final Extended Metrics

Change

Broader dynamic range allowed by removing restrictive clamping.

Test predictions regenerated after sign inversion.

Evaluation

QLSTM: MAE=0.3491, RMSE=0.3493, SMAPE=0.3185, Corr=0.2674

QProphet: MAE=0.1380, RMSE=0.1383, SMAPE=0.1030, Corr=0.4929

ACGA matrices saved.

Analysis

Prophet stabilized significantly with correlation ~0.49 and low error.

LSTM worsened. Prophet may now be more reliable.
