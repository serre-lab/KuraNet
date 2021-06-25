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

def circular_moments(phases, masks=None):
    num_groups = masks.shape[0]
    group_size = masks.sum(1)
    max_loss = 1 + .5 * num_groups * (1. / np.arange(1, num_groups + 1)**2)[:int(num_groups / 2.)].sum()
    group_size = torch.where(group_size == 0, torch.ones_like(group_size), group_size)
    T = phases.shape[0]
    
    #burn_in_steps = int(burn_in_prop * T)
    masked_phases = phases.unsqueeze(1) * masks.unsqueeze(0)
    xx = torch.where(masks.bool(), torch.cos(masked_phases), torch.zeros_like(masked_phases))
    yy = torch.where(masks.bool(), torch.sin(masked_phases), torch.zeros_like(masked_phases))
    go = torch.sqrt((xx.sum(-1))**2 + (yy.sum(-1))**2) / group_size
    synch = 1 - go.sum(-1)/num_groups
    mean_angles = torch.atan2(yy.sum(-1),xx.sum(-1))
    desynch = 0
    for m in np.arange(1, int(np.floor(num_groups/2.))+1):
#         K_m = 1 if m < int(np.floor(num_groups/2.)) + 1 else -1
        desynch += (1.0 / (2* num_groups * m**2)) * (torch.cos(m*mean_angles).sum(-1)**2 + torch.sin(m*mean_angles).sum(-1)**2)

    loss = (synch + desynch) / max_loss
    return loss.mean()
    
def cohn_loss(phases, masks=None, eps=1e-5):
    """
    phases: shape [steps, ...]
    masks(one hot indicator): shape [num_groups, ...]
    split: if return synch and desynch separately
    burn_in_prop: proportion of steps to be considered, steps before will not be added into loss
    """
    
    num_groups = masks.shape[0]
    group_size = masks.sum(1)
    group_size = torch.where(group_size == 0, torch.ones_like(group_size), group_size)
    T = phases.shape[0]
    
    masked_phases = phases.unsqueeze(1) * masks.unsqueeze(0) # [steps, num_groups, ...]
    xx = torch.where(masks.bool(), torch.cos(masked_phases), torch.zeros_like(masked_phases))
    yy = torch.where(masks.bool(), torch.sin(masked_phases), torch.zeros_like(masked_phases))
    go = torch.sqrt((xx.sum(-1))**2 + (yy.sum(-1))**2) / group_size
    synch = 1 - go.sum(-1)/num_groups

    mean_xx = xx.sum(-1) / group_size
    mean_yy = yy.sum(-1) / group_size
    mean_angles = torch.atan2(mean_yy, mean_xx)
    phase_diffs = (mean_angles.unsqueeze(2) - mean_angles.unsqueeze(1))
    desynch = ((-1*torch.log(torch.abs(2*torch.sin(.5*(phase_diffs))) + eps)) + np.log(2)).sum((1,2)) / (num_groups)**2
    
    loss = synch + desynch
    
    return loss.mean()

def sparse_dense_mul(s, d):
  i = s._indices()
  v = s._values()
  dv = d[i[0,:], i[1,:]]  # get values from relevant entries of dense matrix
  return torch.sparse.FloatTensor(i, v * dv, s.size())

def yield_zero_column(n,m):
    i = 0
    while i <= n:
        yield torch.zeros((m,1)), 0

def circular_variance(phases, masks=None):
    num_phases = phases.shape[-1]
    xx = torch.cos(phases)
    yy = torch.sin(phases)
    return (1 - (torch.sqrt(xx.sum(-1)**2 + yy.sum(-1)**2)) / num_phases).mean()

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
    
    
def make_masks(Y, num_classes, device):
    #TODO Explain that this is a limitation
    target = [Y[key] for key in Y.keys()][0].to(device)
    num_units = target.shape[0]
    masks = torch.FloatTensor(num_units, num_classes).zero_().to(device)
    masks.scatter_(1,target.unsqueeze(1),1).transpose_(1,0)
    return masks
 
