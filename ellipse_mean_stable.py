import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# -----------------------------
# Utility functions
# -----------------------------
# make a matrix symmetric
def symmetrize(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.transpose(-1, -2))

# project to PSD
def psd_project(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    "nearPD"-ish: project to PSD cone by eigenvalue clipping.
    """
    A = symmetrize(A)
    evals, evecs = torch.linalg.eigh(A)
    # lower bounded by eps
    evals = torch.clamp(evals, min=eps)
    # Using a broadcasting trick - Q diag \lambda here the last thing is just Q'
    # evals has shape (..., d). unsqueeze(-2) -> (..., 1, d) so it can
    # broadcast across the columns of evecs (..., d, d), effectively forming
    # Q diag(evals) without explicitly constructing the diagonal matrix.
    # transpose(-1, -2) swaps the last two dimensions (matrix axes) to give Q^T,
    # while leaving any leading batch dimensions unchanged.
    return (evecs * evals.unsqueeze(-2)) @ evecs.transpose(-1, -2)

# same as above but now we're taking the sqrt 
def matrix_sqrt_psd(A: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Symmetric PSD square root via eigh.
    """
    A = symmetrize(A)
    evals, evecs = torch.linalg.eigh(A)
    evals = torch.clamp(evals, min=eps)
    sqrt_evals = torch.sqrt(evals)
    return (evecs * sqrt_evals.unsqueeze(-2)) @ evecs.transpose(-1, -2)

# same as above but now we're taking the inverted sqrt 
def inv_sqrt_psd(A: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Symmetric PSD inverse square root via eigh.
    """
    A = symmetrize(A)
    evals, evecs = torch.linalg.eigh(A)
    evals = torch.clamp(evals, min=eps)
    inv_sqrt_evals = 1.0 / torch.sqrt(evals)
    return (evecs * inv_sqrt_evals.unsqueeze(-2)) @ evecs.transpose(-1, -2)

#  randn like is gaussian noise!
def gaussian_like(x: torch.Tensor, std: float) -> torch.Tensor:
    return torch.randn_like(x) * std

# Projection functions
def ball_projection(X: torch.Tensor, radius: float, center: torch.Tensor) -> torch.Tensor:
    """
    Project each row of X onto the closed Euclidean ball B(center, radius).
    """
    diff = X - center.unsqueeze(0)  # (n,d)
    norms = torch.linalg.norm(diff, dim=1).clamp_min(1e-12)
    radius_t = radius if torch.is_tensor(radius) else norms.new_tensor(radius)
    radius_t = radius_t.to(device=norms.device, dtype=norms.dtype)    
    scale = torch.clamp(radius_t / norms, max=1.0)
    return center.unsqueeze(0) + diff * scale.unsqueeze(1)


# -----------------------------
# Winsorized g_r and U-stat cov estimator 
# - we are not using this in this paper but in a later paper we will
# -----------------------------
def winsorize_gr(delta: torch.Tensor, r: float) -> torch.Tensor:
    """
    Vectorized version of:
      g_win(x,r) = min(1, r^2/||x||) * x
    """
    norms = torch.linalg.norm(delta, dim=-1).clamp_min(1e-12)
    factor = torch.minimum(torch.ones_like(norms), (r * r) / norms)
    return delta * factor.unsqueeze(-1)

# -----------------------------
# - we are not using this in this paper but in a later paper we will
# -----------------------------
def estimate_Kg_win_mh(
    S: torch.Tensor,
    MH: torch.Tensor,
    r: float,
    batch_pairs: int = 200_000,
) -> torch.Tensor:
    """
    Estimate K_hat = (2/(n(n-1))) sum_{i<j} g(MH * (S_j-S_i)/sqrt(2)) g(.)^T

    NOTE: This is O(n^2) pairs; we do it in chunks.
    """
    device = S.device
    n, d = S.shape

    # all pairs i<j
    pairs = torch.combinations(torch.arange(n, device=device), r=2)  # (m,2)
    m = pairs.shape[0]

    K = torch.zeros((d, d), device=device, dtype=S.dtype)

    sqrt2_inv = 1.0 / math.sqrt(2.0)

    # chunk over pairs to control memory
    for start in range(0, m, batch_pairs):
        end = min(m, start + batch_pairs)
        ij = pairs[start:end]
        i = ij[:, 0]
        j = ij[:, 1]
        deltas = (S[j] - S[i]) * sqrt2_inv                 # (b,d)
        deltas = deltas @ MH.transpose(0, 1)               # apply MH
        G = winsorize_gr(deltas, r=r)                      # (b,d)
        # sum of outer products: G^T G
        K = K + G.transpose(0, 1) @ G

    K = K * (2.0 / (n * (n - 1)))
    return K

# -----------------------------
# - we are not using this in this paper but in a later paper we will
# -----------------------------
def add_symmetric_gaussian_noise_upper(
    A: torch.Tensor,
    std_upper: float
) -> torch.Tensor:
    """
    Add i<=j iid N(0, std_upper^2) then symmetrize exactly like the R code:
      Kg_est[upper] += noise
      Kg_est[lower] = t(Kg_est)[lower]
    :contentReference[oaicite:6]{index=6}
    """
    d = A.shape[0]
    out = A.clone()
    iu = torch.triu_indices(d, d, offset=0, device=A.device)
    noise = torch.randn((iu.shape[1],), device=A.device, dtype=A.dtype) * std_upper
    out[iu[0], iu[1]] = out[iu[0], iu[1]] + noise
    out = symmetrize(out)
    return out

# -----------------------------
# - we are not using this in this paper but in a later paper we will
# -----------------------------
def private_Kg(
    S: torch.Tensor,
    r: float,
    MH: torch.Tensor,
    rho: float = 1.0,
    batch_pairs: int = 200_000
) -> torch.Tensor:
    """
    PyTorch analog of:
      sens <- 4*r^2/n
      Kg_est <- estimate_Kg_win_mh_rcpp(...)
      add N(0, sens/sqrt(2*rho)) to upper triangle
    :contentReference[oaicite:7]{index=7}
    """
    n, d = S.shape
    sens = 4.0 * (r * r) / n
    Kg_est = estimate_Kg_win_mh(S, MH=MH, r=r, batch_pairs=batch_pairs)
    std_upper = sens / math.sqrt(2.0 * rho)
    Kg_noised = add_symmetric_gaussian_noise_upper(Kg_est, std_upper=std_upper)
    MH_inv = torch.linalg.inv(MH)
    Kg_original_scale = MH_inv @ Kg_noised @ MH_inv.transpose(-1, -2)
    return Kg_original_scale

#############################################################
# -----------------------------
# BalloonUpdate is here, I had it named something else before 
# -----------------------------
# Note that Sigma=EV D EV^t
def dp_above_threshold_ellipse(
    S: torch.Tensor,
    D: torch.Tensor,
    EV: torch.Tensor,
    center: torch.Tensor,
    R_min: float = 1.0,
    beta: float = 1.001,
    threshold: float = 0.95,
    rho_thresh: float = 1.0,
    rho_query: float = 1.0,
    max_iter: int = 10_000
) -> Optional[float]:
    """
    Here D is eigenvalues (axis-length^2 up to scaling constant)
    """
    n, d = S.shape
    b_thresh = 1.0 / (n * math.sqrt( rho_thresh))
    b_query = 1.0 / (n * math.sqrt( rho_query))

    noisy_T = threshold + float(torch.randn((), device=S.device, dtype=S.dtype) * b_thresh)

    S_C = S - center.unsqueeze(0)
    ROT = S_C @ EV  # (n,d) equals t(EV)%*%S_C in R up to transpose conventions
    D_inv_vals = 1.0 / D
    # Ellipse_Points = sum_i (ROT_i * sqrt(D_inv_i))^2
    Ellipse_Points = torch.sum((ROT * torch.sqrt(D_inv_vals).unsqueeze(0)) ** 2, dim=1)  # (n,)

    # radii[i] = R_min + beta^i - 1
    idx = torch.arange(max_iter + 1, device=S.device, dtype=S.dtype)
    radii = R_min + (beta ** idx) - 1.0

    noise_q = torch.randn((max_iter + 1,), device=S.device, dtype=S.dtype) * b_query

    for i in range(max_iter + 1):
        r_i = float(radii[i])
        count_i = int((Ellipse_Points <= (r_i * r_i)).sum().item())
        noisy_qi = (count_i / n) + float(noise_q[i])
        if noisy_qi >= noisy_T:
            return r_i

    return None

# For whitened data
def dp_above_threshold_ball(
    S: torch.Tensor,
    center: torch.Tensor,
    R_min: float = 1.0,
    beta: float = 1.001,
    threshold: float = 0.95,
    rho_thresh: float = 1.0,
    rho_query: float = 1.0,
    max_iter: int = 10_000
) -> Optional[float]:
    """
    Matches R dp_above_threshold_ball. :contentReference[oaicite:9]{index=9}
    """
    n, d = S.shape
    b_thresh = 1.0 / (n * math.sqrt( rho_thresh))
    b_query = 1.0 / (n * math.sqrt( rho_query))

    noisy_T = threshold + float(torch.randn((), device=S.device, dtype=S.dtype) * b_thresh)

    S_C = S - center.unsqueeze(0)
    norms = torch.linalg.norm(S_C, dim=1)  # (n,)

    for i in range(max_iter + 1):
        r_i = R_min + (beta ** i) - 1.0
        count_i = int((norms <= r_i).sum().item())
        noisy_qi = (count_i / n) + float(torch.randn((), device=S.device, dtype=S.dtype) * b_query)
        if noisy_qi >= noisy_T:
            return float(r_i)

    return None


@dataclass
class PrivateEllipseMeanResult: 
    mu: torch.Tensor          # (d,)
    Sigma: torch.Tensor       # (d,d)
    R: float                  # final radius in whitened space

# Some of these parameters are for a more complicated ellipse mean, 
# which is presented later in another paper potentially
# PyTorch version of BalloonMean! - one iteration
def private_ellipse_mean(
    S: torch.Tensor,
    rho_total: float,
    r: float = 1.0,
    init: Optional[torch.Tensor] = None,
    Sigma_dp: Optional[torch.Tensor] = None,
    R_min: Optional[float] = None,
    beta: float = 1.001,
    threshold: float = 0.95,
    rho_at_ratio: float = 0.5,
    iter_num: int = 1,
    Sigma_unknown: bool = True,
    max_iter: int = 10_000,
    batch_pairs: int = 200_000 # you can ignore for now
) -> PrivateEllipseMeanResult:
    """
    PyTorch version of BalloonMean! - one iteration
    """
    device = S.device
    dtype= S.dtype
    n, d = S.shape
    # Set inital values if not provided:
    if init is None:
        init = torch.randn((d,), device=device, dtype=dtype)   # rough analogue
    if R_min is None:
        R_min = math.sqrt(d)/10

    # budget split this is unknown Sigma
    if (Sigma_dp is None and Sigma_unknown):
        rho_cov = 0.5 * rho_total
        rho_width = 0.2 * rho_total
        rho_center = 0.3 * rho_total

        MH = torch.eye(d, device=device, dtype=S.dtype)  #  passes diag(d) on first call
        Sigma_dp = private_Kg(S, r=r, MH=MH, rho=rho_cov, batch_pairs=batch_pairs)
        Sigma_dp = psd_project(Sigma_dp, eps=1e-6)

    elif (iter_num == 2 and Sigma_unknown):
        rho_cov = 0.5 * rho_total
        rho_width = 0.2 * rho_total
        rho_center = 0.3 * rho_total

        MH = matrix_sqrt_psd(torch.linalg.inv(psd_project(Sigma_dp)), eps=1e-10)  # matrix_square_root(solve(Sigma_dp))
        Sigma_dp = private_Kg(S, r=r, MH=MH, rho=rho_cov, batch_pairs=batch_pairs)
        Sigma_dp = psd_project(Sigma_dp, eps=1e-6)
    # budget split this is known Sigma
    else:
        rho_width = 0.5 * rho_total
        rho_center = 0.5 * rho_total

    Sigma_dp = 0.5 * (Sigma_dp + Sigma_dp.T)
    avg_var = torch.trace(Sigma_dp) / d
    I = torch.eye(d, device=Sigma_dp.device, dtype=Sigma_dp.dtype)

    # This tau is for stabilizing the covariance, not the paper tau - we don't use this in the current paper
    if Sigma_unknown:
        tau0 = 1e-2
        tau = (tau0 / max(float(rho_total), 1e-6)) * avg_var

        for _ in range(5):  # a few cheap tries
            Sigma_reg = Sigma_dp + tau * I
            Sig = inv_sqrt_psd(Sigma_reg.double(), eps=1e-12).to(dtype=S.dtype)
            opnorm = float(torch.linalg.svdvals(Sig)[0])
            if opnorm <= 50.0:   # target stability
                break
            tau = tau * 10.0     # increase ridge
        Sig_sq=matrix_sqrt_psd(Sigma_reg.double(), eps=1e-12).to(dtype=S.dtype)
    else:
        Sig = inv_sqrt_psd(Sigma_dp.double(), eps=1e-12).to(dtype=S.dtype) 
        Sig_sq=matrix_sqrt_psd(Sigma_dp.double(), eps=1e-12).to(dtype=S.dtype)
    # S_rot = S @ Sig
    # center = init @ Sig
    S_rot  = S @ Sig.T
    center = init @ Sig.T

    # AboveThreshold on ball in whitened space
    r_star = dp_above_threshold_ball(
        S_rot,
        center=center,
        R_min=float(R_min),
        beta=float(beta),
        threshold=float(threshold),
        rho_query=float(rho_at_ratio * rho_width),
        rho_thresh=float((1.0 - rho_at_ratio) * rho_width),
        max_iter=int(max_iter)
    )
    if r_star is None:
        r_star = float(r)

    # project to ball and then private mean
    X_proj = ball_projection(S_rot, radius=r_star, center=center)

    mean_proj = X_proj.mean(dim=0)
    noise_std = mean_proj.new_tensor(r_star / (n * math.sqrt(2.0 * rho_center)))  # R: final_radius/(n*sqrt(2*rho_center)) :contentReference[oaicite:14]{index=14}
    mean_noised = mean_proj + torch.randn_like(mean_proj) * noise_std

    # map back to original space: p_mean = Sig_sq %*% mean_r
    mu_hat = (Sig_sq @ mean_noised.unsqueeze(1)).squeeze(1)

    return PrivateEllipseMeanResult(mu=mu_hat, Sigma=Sigma_dp, R=float(r_star))


# -----------------------------
# Iteration version
# -----------------------------
def private_ellipse_iteration(
    S: torch.Tensor,
    rho_total: float,
    r: float = 2.0,
    R_min: float = 1.0,
    R_max: float = 1000,
    iters: int = 2,
    split: Optional[torch.Tensor] = None,
    init: Optional[torch.Tensor] = None,
    beta: float = 1.01,
    thresholds: Optional[Tuple[float, ...]] = None,
    rho_at_ratio: float = 0.5,
    Sigma_unknown: bool = True,
    Sigma: Optional[torch.Tensor] = None,
    max_iter: int = 10_000,
    batch_pairs: int = 200_000) -> torch.Tensor:
    """
    PyTorch version of BalloonMean! - iterations
    """
    device = S.device
    dtype=S.dtype
    n, d = S.shape

    rho_init = 0.1 * rho_total
    rho_body = 0.9 * rho_total

    if init is None:
        init0 = torch.zeros((d,), device=device, dtype=dtype)
    else:
        init0 = init.to(device=device, dtype=dtype)   

    if Sigma_unknown: 
        init_proj = ball_projection(S, radius=R_max, center=init0).mean(dim=0)
        init_noise_std_val = R_max / (n * math.sqrt(2.0 * rho_init))  # can be float OR tensor
        init_noise_std = torch.as_tensor(init_noise_std_val, device=init_proj.device, dtype=init_proj.dtype)
        prev_mean = init_proj +  torch.randn_like(init_proj)* init_noise_std
    else:
        # Need to rotate if we know Sigma - this is \tilde mu_1
        Sig = inv_sqrt_psd(Sigma.double(), eps=1e-12).to(dtype=S.dtype) 
        Sig_sq=matrix_sqrt_psd(Sigma.double(), eps=1e-12).to(dtype=S.dtype)
        S_rot  = S @ Sig.T
        center = init0 @ Sig.T
        init_proj = ball_projection(S_rot, radius=R_max, center=center).mean(dim=0)
        init_noise_std_val = R_max / (n * math.sqrt(2.0 * rho_init))  
        init_noise_std = torch.as_tensor(init_noise_std_val, device=init_proj.device, dtype=init_proj.dtype)
        prev_mean = init_proj +  torch.randn_like(init_proj)* init_noise_std
        prev_mean =  (Sig_sq @ prev_mean.unsqueeze(1)).squeeze(1)


    # split budget evenly amongst tensors
    if split is None:
        split = torch.full((iters,), 1.0 / iters, device=device, dtype=dtype)
    else:
        split = split.to(device=device, dtype=dtype)
        split = split / split.sum()

    if thresholds is None:
        thresholds = tuple([0.9] * iters)
    
    if Sigma_unknown:
        Sigma_dp = None
    else:
        Sigma_dp= Sigma
    cur_r = float(r)

    for i in range(1, iters + 1):
        res = private_ellipse_mean(
            S,
            rho_total=float(rho_body * float(split[i - 1])),
            r=float(cur_r),
            init=prev_mean,
            Sigma_dp=Sigma_dp,
            R_min=float(R_min),
            beta=float(beta),
            threshold=float(thresholds[i - 1]),
            rho_at_ratio=float(rho_at_ratio),
            iter_num=int(i),
            max_iter=int(max_iter),
            batch_pairs=int(batch_pairs)
        )
        if Sigma_unknown:
            Sigma_dp = res.Sigma
        prev_mean = res.mu
        cur_r = float(min(res.R, float(R_max)))

    return prev_mean

