import multiprocessing as mp
import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import hadamard


def quantile_binary_search(x, m, u, p, T=10, l=1):
    left = l
    right = u
    i = 0
    while i < T:
        mid = (left+right)/2.
        count = len([xi for xi in x if xi<=mid])
        count_noisy = count + np.random.normal(0, np.sqrt(T/(2.*p))) 
        if count_noisy <= m:
            left = mid# + 1
        else:
            right = mid
        i = i+1
    return (left+right)/2.

def clipped_mean(x, n, d, u, p, l=0, T=10, threshold=None):
    p1 = p * 0.25
    p2 = p * 0.75
    x_norm = np.linalg.norm(x, axis=1)
    if threshold is not None:
        C = threshold
    else:
        m = int(n - 2.*np.sqrt(d/(2.*p2)) - np.sqrt(T/(2*p1)))
        C = quantile_binary_search(x_norm,m,u,p1,T=T,l=0)
    x_clipped = []
    for i in range(len(x)):
        xi_norm = x_norm[i]
        scale = min(C/xi_norm,1.0)
        x_clipped.append(scale*np.array(x[i]))
    mean = np.mean(x_clipped,axis=0)
    noisy_mean = mean + np.random.normal(0, 2.*C/np.sqrt(2.*p2)/n, size=d)
    return noisy_mean


def random_hadamard(d,t=1):
    H = hadamard(d)/np.sqrt(d)
    rand = np.random.uniform(low=0,high=1,size=d)
    flipped = (rand < 0.5)
    diagonal = np.ones(d)
    diagonal[flipped] = -1
    D = np.diag(diagonal)
    HD = np.matmul(H,D)
    # rotate t-1 more times
    for i in range(t-1):
        rand = np.random.uniform(low=0,high=1,size=d)
        flipped = (rand < 0.5)
        diagonal = np.ones(d)
        diagonal[flipped] = -1
        D = np.diag(diagonal)
        HD = np.matmul(HD,np.matmul(H,D))
    return HD
    

def random_rotation_mean(x,d,u,p,T=10,prop=0.25):
    n = len(x)
    HD = random_hadamard(d,t=3)
    HD_inv = np.transpose(HD)
    x_hat = np.matmul(HD,x.T).T
    ps = prop*p/d
    up = u/np.sqrt(d)
    c_tilde = quantile_binary_search_mp(x_hat,d,0.5*n,up,ps,T=T,l=-up)
    x_shifted = x_hat - c_tilde
    y_tilde = clipped_mean(x_shifted,n,d,u,(1-prop)*p,T=T)
    y_hat = np.matmul(HD_inv,y_tilde+c_tilde)
    return y_hat


def quantile_binary_search_mp_wrapper(args):
    return quantile_binary_search(*args)


def quantile_binary_search_mp(x,d,m,u,p,T=10,l=1,para='0'):
    if para=='0':
        num_cores = mp.cpu_count()
        processed_list = Parallel(n_jobs=num_cores)(delayed(quantile_binary_search)(x[:,o],m,u,p,T=T,l=l) for o in range(d))
    elif para=='1':
        processed_list = None
        with mp.Pool(4) as pool:
            processed_list = pool.map(quantile_binary_search_mp_wrapper, [(x[:,o],m,u,p,T,l) for o in range(d)])
    else:
        processed_list = [quantile_binary_search(x[:,o],m,u,p,T=T,l=l) for o in range(d)]
    return processed_list
