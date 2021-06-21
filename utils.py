import matplotlib.pyplot as plt
from matplotlib import rc
plt.style.use('ggplot')
import os
import numpy as np
import torch
from torch.nn.functional import softmax
from scipy.special import softmax as np_softmax
import numpy as np
from torch.distributions import uniform, cauchy, normal, relaxed_bernoulli, negative_binomial
import time
import pickle

# Loss functions and statistics

def sparse_dense_mul(s, d):
  i = s._indices()
  v = s._values()
  dv = d[i[0,:], i[1,:]]  # get values from relevant entries of dense matrix
  return torch.sparse.FloatTensor(i, v * dv, s.size())

def yield_zero_column(n,m):
    i = 0
    while i <= n:
        yield torch.zeros((m,1)), 0

def circular_variance(phases, reduce=True):
    num_phases = phases.shape[-1]
    xx = torch.cos(phases)
    yy = torch.sin(phases)
    if reduce:
        return (1 - (torch.sqrt(xx.sum(-1)**2 + yy.sum(-1)**2)) / num_phases).mean()
    else:
        return (1 - (torch.sqrt(xx.sum(-1)**2 + yy.sum(-1)**2)) / num_phases)

def c_x(x, connectivity, reduce=True):
    x_bar = x.mean(0).unsqueeze(0)
    num = (connectivity * torch.einsum('id,jd->ij',(x - x_bar) ,(x - x_bar) )).sum((0,1))
    den = (connectivity * ((x - x_bar)**2).unsqueeze(1).sum(-1)).sum((0,1))
    if reduce:
        return (num / den + 1e-6).mean()
    else:
        return (num / den + 1e-6)

def laplacian(connectivity,sym_norm=True, batch=True):
    A = connectivity
    D = torch.diag_embed(torch.abs(A).sum(1))
    D_mask = torch.diag_embed(torch.ones_like(A.sum(1)))
    L = D-A
    D2neg = torch.where(D_mask.bool(), D**(-.5),torch.zeros_like(D))
    D2neg = torch.where(torch.isinf(D2neg), torch.zeros_like(D2neg),D2neg)
    if sym_norm:
        if batch:
            return torch.bmm(D2neg,torch.bmm(L,D2neg))
        else:
            return torch.matmul(D2neg, torch.matmul(L,D2neg))   
    else:
        return L
def p_neg(omega, connectivity, reduce=True, weighted=False):
    num_units = omega.shape[-1]
    sign_omega = torch.sign(omega)
    sign_prods = torch.einsum('bi,bj->bij',sign_omega, sign_omega)
    if weighted:
        return (((sign_prods < 0)*1) * connectivity).mean()
    num_links= (connectivity > 0).sum((1,2))
    opp_connections = (sign_prods * connectivity) < 0
    if reduce:
        return ((opp_connections.sum((1,2))) / num_links.float()).mean()
    else:
        return ((opp_connections.sum((1,2))) / num_links.float())

def omega_energy(omega, connectivity, reduce=True):
    L = laplacian(connectivity)
    if reduce:
        return -1*torch.einsum('bi,bi->b', omega, torch.einsum('bij,bi->bj',L, omega)).mean()
    else:
        return -1*torch.einsum('bi,bi->b', omega, torch.einsum('bij,bi->bj',L, omega)).mean()

def save_object(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_object(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def nearly_square(n):
    factors = []
    for i in range(1, n + 1):
        if n % i == 0:
            factors.append(i)

    squarest_loc = np.argsort([np.abs((n / a) - a) for a in factors])[0]
    return (factors[squarest_loc], n / factors[squarest_loc])
    
    
