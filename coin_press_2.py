import numpy as np
import torch
from typing import Optional, Tuple, Dict
import math 

def cov_nocenter(X):
    cov = torch.mm(X.t(), X)/X.size(0)
    return cov

def cov(X):
    X = X - X.mean(0)
    cov = torch.mm(X.t(), X)/X.size(0)
    return cov

'''
PSD projection
'''
def psd_proj_symm(S):
    U, D, V_t = torch.svd(S)
    D = torch.clamp(D, min=0, max=None).diag()
    A = torch.mm(torch.mm(U, D), U.t()) 
    return A


def cov_est_step(X, A, rho,beta):
    """
    One step of multivariate covariance estimation, scale cov a.
    """
    # print(X.shape)
    if len(X.shape)>2:
        # print(X.shape)
        X=torch.squeeze(X)
        # print('after' ,X.shape)
    n, d = X.shape

    #Hyperparameters
    gamma = gaussian_tailbound(d, beta)
    # Was here before
    # gamma = gaussian_tailbound(d, 0.1)
    eta = 0.5*(2*(math.sqrt(d/n)) + (math.sqrt(d/n))**2)
    nu=0.0
    # actual params in paper
    # eta = 2*(math.sqrt(d/n) + math.sqrt(2*math.log(2/beta)/n)) + (math.sqrt(d/n) + math.sqrt(2*math.log(2/beta)/n))**2
    # nu = (gamma**2 / (n*math.sqrt(rho))) * (2*math.sqrt(d) + 2*d**(1/16) * math.log(d)**(1/3) + (6*(1 + (math.log(d)/d)**(1/3))*math.sqrt(math.log(d)))/(math.sqrt(math.log(1 + (math.log(d)/d)**(1/3)))) + 2*math.sqrt(2*math.log(1/beta)))

    #truncate points
    # W = torch.mm(X, A)
    W = torch.mm(X, A.t())
    # W_norm = np.sqrt((W**2).sum(-1, keepdim=True))
    W_norm = torch.linalg.norm(W, dim=-1, keepdim=True)
    norm_ratio = gamma / W_norm
    large_norm_mask = (norm_ratio < 1).squeeze()
    
    W[large_norm_mask] = W[large_norm_mask] * norm_ratio[large_norm_mask]
    
    # noise
    Y = torch.randn(d, d,device=X.device,dtype=X.dtype)
    noise_var = (gamma**4/(rho*n**2))
    Y *= torch.sqrt(noise_var)    
    #can also do Y = torch.triu(Y, diagonal=1) + torch.triu(Y).t()
    Y = torch.triu(Y)
    Y = Y + Y.t() - Y.diagonal().diag_embed() # Don't duplicate diagonal entries
    Z = (torch.mm(W.t(), W))/n
    #add noise    
    Z = Z + Y
    #ensure psd of Z
    Z = psd_proj_symm(Z)
    
    U = Z + (nu+eta)*torch.eye(d,device=X.device,dtype=X.dtype)
    inv = torch.inverse(U)
    inv_sqrt=compute_sqrt_mat(inv)
    # invU, invD, invV = inv.svd()
    # inv_sqrt = torch.mm(invU, torch.mm(invD.sqrt().diag_embed(), invV.t()))
    A = torch.mm(inv_sqrt, A)
    return A, Z



def compute_sqrt_mat(A):
    U, D, V = A.svd()
    inv_sqrt = torch.mm(U, torch.mm(D.sqrt().diag_embed(), V.t()))
    return inv_sqrt


def cov_est(X, rho,d,u,t,beta=0.1):
    """
    Multivariate covariance estimation.
    Returns: zCDP estimate of cov.
    """
    A = torch.eye(d,device=X.device,dtype=X.dtype) / math.sqrt(u)
    assert len(rho) == t
    # looping
    for i in range(t-1):
        A, Z = cov_est_step(X, A, rho[i], beta/(4*(t-1)))
    A_t, Z_t = cov_est_step(X, A, rho[-1], beta/4)
    # final covariance
    cov = torch.mm(torch.mm(A.inverse(), Z_t), A.inverse().t())
    return cov






def gaussian_tailbound(d,b):
    return ( d + 2*( d * math.log(1/b) )**0.5 + 2*math.log(1/b) )**0.5

def mahalanobis_dist(M, Sigma):
    Sigma_inv = torch.inverse(Sigma)
    U_inv, D_inv, V_inv = Sigma_inv.svd()
    Sigma_inv_sqrt = torch.mm(U_inv, torch.mm(D_inv.sqrt().diag_embed(), V_inv.t()))
    M_normalized = torch.mm(Sigma_inv_sqrt, torch.mm(M, Sigma_inv_sqrt))
    return torch.norm(M_normalized - torch.eye(M.size()[0]), 'fro')

''' 
Functions for mean estimation
'''

##    X = dataset
##    c,r = prior knowledge that mean is in B2(c,r)
##    t = number of iterations
##    Ps = 
def multivariate_mean_iterative(X, c, r, t, Ps,beta=0.1):
    for i in range(t-2):
        c, r = multivariate_mean_step(X, c, r, Ps[i],beta/(4*(t-1)))
    c, r = multivariate_mean_step(X, c, r, Ps[t-1],beta/4)
    return c

def multivariate_mean_step(X, c, r, p,beta):
    n, d = X.shape

    ## Determine a good clipping threshold
    gamma = gaussian_tailbound(d,beta)
    clip_thresh = min((r**2 + 2*r*3 + gamma**2)**0.5 , r + gamma) #3 in place of sqrt(log(2/beta))
        
    ## Round each of X1,...,Xn to the nearest point in the ball B2(c,clip_thresh)
    # in case
    c = c.to(device=X.device, dtype=X.dtype)
    x = X - c
    mag_x = torch.linalg.norm(x, axis=1)
    mag_x = mag_x.to(device=X.device, dtype=X.dtype)
    if torch.is_tensor(clip_thresh):
        clip_t = clip_thresh.to(device=mag_x.device, dtype=mag_x.dtype)
    else:
        clip_t = mag_x.new_tensor(clip_thresh)
    outside_ball = (mag_x > clip_t)
    x_hat = (x.T / mag_x).T
    if torch.sum(outside_ball)>0:
        X[outside_ball] = c + (x_hat[outside_ball] * clip_t)
    
    ## Compute sensitivity
    delta = 2*clip_t/float(n)
    # print(delta)
    # print(p)
    sd = delta/(2*p)**0.5
    
    ## Add noise calibrated to sensitivity
    # Y = np.random.normal(0, sd, size=d)
    sd_t = sd.to(device=X.device, dtype=X.dtype) if torch.is_tensor(sd) else X.new_tensor(sd)
    Y = torch.randn(d, device=X.device, dtype=X.dtype) * sd_t
    c = torch.sum(X, axis=0)/float(n) + Y
    r = ( 1/float(n) + sd**2 )**0.5 * gaussian_tailbound(d,0.01)
    return c, r


def overall_mean(X, u,c,r,t,Ps,beta):
    # n, d = X.shape
    # print('before', X.shape)
    if len(X.shape)>2:
        # print(X.shape)
        X=torch.squeeze(X)
    if len(X.shape)<2:
        X=X.view(X.shape[0], 1)
    n, d =X.shape
    row_diff = torch.diff(X, axis=0)[::2]
    Sigma=cov_est(row_diff, Ps,d,u,t,beta)/2
    # Stabilize Sigma before whitening
    Sigma = 0.5 * (Sigma + Sigma.T)
    d = Sigma.shape[0]
    avg_var = torch.trace(Sigma) / d
    tau = 1e-2 * avg_var  # try 1e-2; increase if rho is small

    Sigma_reg = Sigma + tau * torch.eye(d, device=Sigma.device, dtype=Sigma.dtype)

    sqrt_mat = compute_sqrt_mat(Sigma_reg)
    adj = torch.linalg.inv(sqrt_mat)
    whitened = X @ adj
    # multivariate_mean_iterative(X, c, r, t, Ps)
    mean_est = multivariate_mean_iterative(whitened,c,r,t,Ps,beta)
    mean_est = mean_est @ sqrt_mat
    return [mean_est, Sigma]


def overall_mean_S_known(X,c,r,t,Ps,beta,Sigma):
    # n, d = X.shape
    # print('before', X.shape)
    if len(X.shape)>2:
        # print(X.shape)
        X=torch.squeeze(X)
    if len(X.shape)<2:
        X=X.view(X.shape[0], 1)
    n, d =X.shape
    # row_diff = torch.diff(X, axis=0)[::2]
    # Sigma=cov_est(row_diff, Ps,d,u,t,beta)/2
    # Stabilize Sigma before whitening - don't when known... 
    # Sigma = 0.5 * (Sigma + Sigma.T)
    # d = Sigma.shape[0]
    # avg_var = torch.trace(Sigma) / d
    # tau = 1e-2 * avg_var  # try 1e-2; increase if rho is small

    # Sigma_reg = Sigma + tau * torch.eye(d, device=Sigma.device, dtype=Sigma.dtype)

    sqrt_mat = compute_sqrt_mat(Sigma)
    adj = torch.linalg.inv(sqrt_mat)
    whitened = X @ adj
    # multivariate_mean_iterative(X, c, r, t, Ps)
    mean_est = multivariate_mean_iterative(whitened,c,r,t,Ps,beta)
    mean_est = mean_est @ sqrt_mat
    return [mean_est, Sigma]

# def overall_mean(X, u,c,r,t,Ps,beta):
#     # n, d = X.shape
#     # print('before', X.shape)
#     if len(X.shape)>2:
#         # print(X.shape)
#         X=torch.squeeze(X)
#     if len(X.shape)<2:
#         X=X.view(X.shape[0], 1)
#     n, d =X.shape
#     row_diff = torch.diff(X, axis=0)[::2]
#     Sigma=cov_est(row_diff, Ps,d,u,t,beta)/2

#     # U, S, Vh =torch.linalg.svd(Sigma)
#     # D = torch.diag(torch.sqrt(S))
#     # sqrt_mat = U @ D @ Vh
#     sqrt_mat = compute_sqrt_mat(Sigma)
#     adj = torch.linalg.inv(sqrt_mat)

#     # print('after', X.shape)
#     whitened=X @ adj
#     # multivariate_mean_iterative(X, c, r, t, Ps)
#     mean_est = multivariate_mean_iterative(whitened,c,r,t,Ps)
#     mean_est = mean_est @ sqrt_mat
#     return [mean_est, Sigma]


def COINPRESS(
    X: torch.Tensor,
    rho: float,
    c: torch.Tensor,
    r: float,
    u: float,
    t: int,
    nm: torch.Tensor,
    beta: float,
    Sigma_unknown: bool = True,
    Sigma: Optional[torch.Tensor] = None):
    # nm=torch.tensor([1/3.0, 1/2.0, 1.0])
    # if t!=3: 
    #     nm=torch.ones(t)
    # divide by 2 for mu
    if Sigma_unknown: 
        Ps= rho*(nm/nm.sum())/2
        return overall_mean(X,u,c,r,t,Ps,beta)
    else:
        Ps= rho*(nm/nm.sum())
        return overall_mean_S_known(X,c,r,t,Ps,beta,Sigma)


