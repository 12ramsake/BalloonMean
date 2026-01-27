
# Balloon Mean: Differentially Private Robust Mean Estimation

This repository contains the reference implementation and simulation code for the Balloon Mean, a differentially private multivariate mean estimator based on iterated clipping of Mahalanobis balls (“balloons”).

Structure
.
├── ellipse_mean_stable.py        # BalloonMean / BalloonUpdate (main implementation)
├── coin_press_2.py               # COINPRESS 
├── instance_opt_method.py        # IOM / random-rotation baseline
├── Huber_GDP.py                  # Robust private baselines (Yu et al.)
│
├── Comparison_Study.ipynb        # Main comparison simulations
├── Sensitivity_main_param.ipynb  # Sensitivity to tuning parameters
├── Sensitivity_tau_schedule.ipynb# Sensitivity to τ schedules
├── Sensitivity_Initial_Mean.ipynb# Sensitivity to initialization
│
└── README.md



This code is written in Python and primarily uses PyTorch.

pip install torch numpy scipy pandas matplotlib joblib


GPU is optional but recommended for large-scale simulations.

# Example data
X = torch.randn(1000, 20)

mu_hat = private_ellipse_iteration(
    S=X,
    rho_total=1.0,
    r=2.0,
    R_min=1.0,
    R_max=100.0,
    iters=2,
    beta=1.01,
    thresholds=(0.9, 0.9),
    Sigma_unknown=True
)

print(mu_hat)


Key parameters:

rho_total — total privacy budget (zCDP)

iters — number of balloon iterations

beta — geometric grid parameter in AboveThreshold

thresholds — target mass captured at each iteration

R_min, R_max — radius range

Sigma_unknown — whether covariance is privately estimated

Tuning parameters

The Balloon Mean depends on a small number of interpretable parameters controlling:

robustness (τ / threshold schedule),

grid size (β),

number of iterations M and radii

The sensitivity notebooks explore these choices and provide recommended defaults.

Reproducing experiments

The experiments in the paper are produced using:

Comparison_Study.ipynb – method comparisons,

Sensitivity_main_param.ipynb – main tuning parameters,

Sensitivity_tau_schedule.ipynb – τ schedules,

Sensitivity_Initial_Mean.ipynb – initialization effects.

These notebooks generate the figures and CSV outputs used in the experimental section.


