import matplotlib.pyplot as plt
from matplotlib import rc
plt.style.use('ggplot')
import os
import numpy as np
import torch
import time
import pickle

# Loss functions and statistics

def circular_moments(phases, masks=None):
    '''circular_moments : this loss measures both the degree of synchrony within prespecified groups and the degree
       of desynchrony between those groups. Synchrony is measured by circular variance. Desynchrony is measured by a sum
       of higher order circular moments. See Eq. 36 in Sepulchre, R., Paley, D. A., & Leonard, N. E. (2007). Stabilization
       of planar collective motion: All-to-all communication. IEEE Transactions on Automatic Control, 52(5), 811–824. https://doi.org/10.1109/TAC.2007.898077
	   
       Positionl arguments are 
	   
       *phases (tensor of size T x n) : the state of the n-dimensional Kuramoto dynamics over T time steps.
	   
       Keyword arguments are

       * masks (tensor of size g x n; default = None) : binary mask indicating which of the n phases is in each of the g groups. '''
    
	# Relevant sizes
    num_groups = masks.shape[0]
    group_size = masks.sum(1)
    group_size = torch.where(group_size == 0, torch.ones_like(group_size), group_size)
    T = phases.shape[0]
			
	# Loss is at least as large as the maxima of each individual loss (total desynchrony + total synchrony)
    loss_bound = 1 + .5 * num_groups * (1. /
	np.arange(1, num_groups + 1)**2)[:int(num_groups / 2.)].sum()
    
	# Calculate global order within each group
    masked_phases = phases.unsqueeze(1) * masks.unsqueeze(0)
    xx = torch.where(masks.bool(), torch.cos(masked_phases), torch.zeros_like(masked_phases))
    yy = torch.where(masks.bool(), torch.sin(masked_phases), torch.zeros_like(masked_phases))
    go = torch.sqrt((xx.sum(-1))**2 + (yy.sum(-1))**2) / group_size
    synch = 1 - go.sum(-1)/num_groups
	
	# Average angle within a group
    mean_angles = torch.atan2(yy.sum(-1),xx.sum(-1))
	
	# Calculate desynchrony between average group phases
    desynch = 0
    for m in np.arange(1, int(np.floor(num_groups/2.))+1):
#         K_m = 1 if m < int(np.floor(num_groups/2.)) + 1 else -1 # This is specified in Eq 36 of the cited paper and may have an effect on the values of the minimum though not its location
        desynch += (1.0 / (2* num_groups * m**2)) * (torch.cos(m*mean_angles).sum(-1)**2 + torch.sin(m*mean_angles).sum(-1)**2)

    # Total loss is average of invidual losses, averaged over time
    loss = (synch + desynch) / loss_bound
    return loss.mean()
    
def cohn_loss(phases, masks=None, eps=1e-5):

    '''cohn loss : this loss measures both the degree of synchrony within prespecified groups and the degree of desynchrony
       between those groups. Synchrony is measured by circular variance. Desynchrony is measured by a logarithmic potential
       defined on phase averages with a group. See Eq. 2.2 in Cohn, H. (1960). Global Equilibrium Theory of Charges on a
       Circle. The American Mathematical Monthly, 67(4), 338–343.
	   
       Positionl arguments are 
	   
       * phases (tensor of size T x n) : the state of the n-dimensional Kuramoto dynamics over T time steps.
	   
       Keyword arguments are
	   
       * masks (tensor of size g x n; default = None) : binary mask indicating which of the n phases is in each of the g groups.
       * eps (float, default=1e-5)                    : additive corretion for the logarithm. Range: (0, infty)'''

    
	# Relevant sizes
    num_groups = masks.shape[0]
    group_size = masks.sum(1)
    group_size = torch.where(group_size == 0, torch.ones_like(group_size), group_size)
    T = phases.shape[0]
    
	# Calculate global order within groups
    masked_phases = phases.unsqueeze(1) * masks.unsqueeze(0) # [steps, num_groups, ...]
    xx = torch.where(masks.bool(), torch.cos(masked_phases), torch.zeros_like(masked_phases))
    yy = torch.where(masks.bool(), torch.sin(masked_phases), torch.zeros_like(masked_phases))
    go = torch.sqrt((xx.sum(-1))**2 + (yy.sum(-1))**2) / group_size
    synch = 1 - go.sum(-1)/num_groups

    # Calcuate phase means
    mean_xx = xx.sum(-1) / group_size
    mean_yy = yy.sum(-1) / group_size
    mean_angles = torch.atan2(mean_yy, mean_xx)
    phase_diffs = (mean_angles.unsqueeze(2) - mean_angles.unsqueeze(1))
	
	# Calculate Cohn's desynchrony
    desynch = ((-1*torch.log(torch.abs(2*torch.sin(.5*(phase_diffs))) + eps)) + np.log(2)).sum((1,2)) / (num_groups)**2
	
    # Total loss is average of invidual losses, averaged over time
    loss = synch + desynch
    return loss.mean()

def circular_variance(phases, **kwargs):
    '''circular_variance : 1 minus the global order of a set of phases (averaged over time). Minimized
       when the phases synchronize. 
	
       Positional arguments are 
	   
       * phases (tensor of size T x n) : the state of the n-dimensional Kuramoto dynamics over T time steps.
	   
       Keyword arguments are
	   
       * **kwargs : dummy argument to interface with rest of code.'''
	# How many oscillators
    num_phases = phases.shape[-1]
	
	# Real and imaginary parts of each phasor
    xx = torch.cos(phases)
    yy = torch.sin(phases)
	
	# 1 minus global order
    return (1 - (torch.sqrt(xx.sum(-1)**2 + yy.sum(-1)**2)) / num_phases).mean()

def c_x(x, connectivity, reduce=True):
    '''c_x : weighted correlation between node feature vectors on a graph. Generalized from Eq. 2 in
        Brede, M. (2008). Synchrony-optimized networks of non-identical Kuramoto oscillators. Physics Letters, Section A: General,
        Atomic and Solid State Physics, 372(15), 2618–2622. https://doi.org/10.1016/j.physleta.2007.11.069
	
        Positional arguments are
	
        * x (tensor of shape n x d)            : the array of d-dimensional features for each of the n graph nodes. 
        * connectivity (tensor of shape n x n) : the graph's weight matrix. 
	
        Keyword arguments are
	
        * reduce (bool, default = True) : whether or not to average '''
	
	# Average value of the feature vector across nodes
    x_bar = x.mean(0).unsqueeze(0)
	
	# Weighted correlation
    num = (connectivity * torch.einsum('id,jd->ij',(x - x_bar) ,(x - x_bar) )).sum((0,1))
    den = (connectivity * ((x - x_bar)**2).unsqueeze(1).sum(-1)).sum((0,1))
    if reduce:
        return (num / den + 1e-6).mean()
    else:
        return (num / den + 1e-6)

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
    target = [Y[key] for key in Y.keys()][0].to(device)
    num_units = target.shape[0]
    masks = torch.FloatTensor(num_units, num_classes).zero_().to(device)
    masks.scatter_(1,target.unsqueeze(1),1).transpose_(1,0)
    return masks
 
