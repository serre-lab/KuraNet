import torch
from torch.nn.functional import softmax
from scipy.special import softmax as np_softmax
import numpy as np
from torch.autograd import Variable
from torch.distributions import uniform, cauchy, normal, relaxed_bernoulli
from utils import c_x, sparse_dense_mul
from torchdiffeq import odeint, odeint_adjoint
import ipdb

class KuraNet(torch.nn.Module):
    def __init__(self,feature_dim, num_hid_units=256, avg_deg=1.0,
                 symmetric=True, rand_inds=False, adjoint=False, solver_method='euler',
                 alpha=.1, gd_steps=50, burn_in_steps=100):
        super(KuraNet, self).__init__()

        '''KuraNet: object for running Kuramoto dynamics and predicting couplings from data. 

           Initialization positional arguments are

           * feature_dim (int) : the number of features per graph node
         
           Initialization keyword arguments are

           * num_hid_units (int, default=256)     : number of hidden neurons in each layer of the connectivity network. Range : [1, infty)
           * avg_deg (float, default=1.0)         : average degree of underlying graph. Range : (0,infty)
           * symmetric (bool, default=True)       : whether the graph with be undirected (True) or directed (False).
           * rand_inds (bool, default=False)      : whether or not to use random updates during ODE solution.
           * adjoint (bool, default=False)        : whether or not to solve adjoint system for parameter updates. See torchdiffeq documentation. 
           * solver_method (str, default='euler') : solver method for ODEs. Range: see torchdiffeq documentation.
           * alpha (float, default=.1)            : step size for ODE solver. Range: (0,infty)
           * gd_steps (int, deafult=50)           : number of steps to integrate for loss calculation. Range: [0, infty)
           * burn_in_steps (int, default=100)     : number of burn_in_steps to discard before integrating the loss. Total number of steps is gd plus burn_in. Range : [0,infty)
           '''
       
        # Initialize connectivity network
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(2*feature_dim, num_hid_units),
            torch.nn.BatchNorm1d(num_hid_units),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(num_hid_units, num_hid_units),
            torch.nn.BatchNorm1d(num_hid_units),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(num_hid_units,1,bias=False))

        # Set attributes
        self.symmetric = symmetric
        self.avg_deg = avg_deg
        self.rand_inds = rand_inds
        self.solver = odeint_adjoint if adjoint else odeint
        self.solver_method = solver_method
        self.set_grids(alpha, burn_in_steps, gd_steps)

    def set_grids(self,alpha, burn_in_steps, gd_steps):
        '''set_grids: discretize ODE time domain. We call the discretized intervals "grids" and place relevant grids in the attribute self.grids.
           Each grid consists of x_steps number of alpha-sized steps where x is either burn_in or gd.
           
           Positional arguments are
           
           * alpha (float)       : step size. Range: [0,infty)
           * burn_in_steps (int) : number of steps to discard before integrating loss. Range: [0,infty)
           * gd_steps (int)      : number of steps to integrate loss'''
 
        self.alpha = alpha
        self.gd_steps = gd_steps
        self.burn_in_steps=burn_in_steps

        burn_in_integration_grid = torch.cumsum(torch.tensor([self.alpha] * (self.burn_in_steps)), 0) # grid for burn in
        grad_integration_grid = torch.cumsum(torch.tensor([self.alpha] * (self.gd_steps + 1)), 0) # grid for loss
        self.grids = [burn_in_integration_grid, grad_integration_grid]

    def run(self,x):
        '''run: solve Kuramoto equation of motion and return a trajectory given the node features, x

        Positional arguments are:

        * x (tensor): Tensor containing node features

        Returns

        trajectory ((T_neg + burn_in_steps + grad_steps) x (k+1) x n-dim tensor): the full dynamical trajectory of length T_neg (maximal delay) + burn_in_steps + grad_steps '''

        # Preliminary
        num_units = x['omega'].shape[0]
        self.fixed_couplings=False
        tau = x['tau']
        T_neg = tau.max().long() + 1
        self.past_phase = torch.zeros((T_neg,num_units)).to(x['omega'].device)

        x = torch.cat([x['omega'], x['h'], x['tau']], dim=-1)
        if num_units == self.batch_size:
            couplings, _ = self.get_couplings(x)
            self.fixed = couplings
            self.fixed_couplings = True

        # Integration grids
        tneg_integration_grid = torch.cumsum(torch.tensor([self.alpha] * T_neg),0)

        # Initialize phase
        init_phase = torch.zeros((num_units,1)).to(x.device)

        # Solve ODE on all grids
        all_trajectories = []
        t = 0
        for g, grid in enumerate([tneg_integration_grid] + self.grids):
            self.tneg = True if g < 1 else False

            grid += t

            if g < 2 : # For burn_in and negative time, no gradient
                torch.set_grad_enabled(False) 
            else: 
                torch.set_grad_enabled(True) # Turn on gradient for last grid

            y = torch.cat([x, init_phase], dim=-1)
            
            # Solve ODE	
            trajectory = self.solver(self, y, grid, rtol=1e-3, atol=1e-5, method=self.solver_method) 
            all_trajectories.append(trajectory[...,-1])
            init_phase = trajectory[...,-1][-1,:].unsqueeze(1) 
            # Update time variable
            t += len(grid) * self.alpha
        
        return torch.cat(all_trajectories,0)

    def forward(self, t, y):

        '''forward: one solver update step. Used by self.solver object. Separates input, y, into its constituents, including the current phase and the node
           features. Then computes couplings as a function of these features. Finally returns the gradient at time t at the current phase. 

           Positional arguments are 
         
           * t (1-dim tensor)         : the current time step. The system can be made non-autonomous by introducing dependence on t (not implemented)
           * y ((k+1) x n-dim tensor) : the dynamical state of the system with k+1 (feature dim + phase) x n (network size) dimensions. Includes both current phase and node features.

           Returns

           * delta (n-dim tensor) : the gradient at the current phase/time. Fed into torchdiffeq backend. '''

        # Unpack input 
        phase = y[:,-1]
        x     = y[:,:-1] 
        omega = x[:,0]
        h = x[:,1]
        tau   = x[:,2]
        T_neg = tau.max().long() + 1 if self.tneg is False else 0

        # Track recent past
        self.past_phase = torch.cat([self.past_phase,phase.unsqueeze(0)],dim=0)
        self.past_phase = self.past_phase[1:,:]
         
        # Get couplings
        couplings, mask = self.get_couplings(x)
        # Random sampling 
        _x = x[mask]
        _phase = phase[mask]
        _omega = omega[mask]
        _h     = h[mask]
        _tau   = tau[mask]
        _past_phase   = self.past_phase[:,mask]

        # Compute interactions
        if T_neg > 1: # if nontrivial delays and in positive time
            delayed_phase = torch.gather(_past_phase, 0, (T_neg - 1 - _tau[None,:]).long())[0]
            phase_diffs = torch.sin(delayed_phase.unsqueeze(1) - _phase.unsqueeze(0))
        else:
            phase_diffs = torch.sin(_phase.unsqueeze(1) - _phase.unsqueeze(0))
 

        # Compute external field strengths
        ext_field = _h * torch.sin(_phase)

        local_delta = _omega + (couplings*phase_diffs).sum(0) + ext_field
        delta = torch.zeros_like(phase)
        delta[mask] = local_delta

        delta = torch.cat([torch.zeros_like(x), delta.unsqueeze(-1)], -1)

        return delta

    def set_batch_size(self, batch_size):
        '''set_batch_size: sets the dynamical batch size. This is the number of units sampled for updating if random sampling is enabled.

        Positional arguments are

        * batch_size (int) : the dynamic batch size'''
        self.batch_size = batch_size
       
    def get_couplings(self, x):

        '''get_couplings: return couplings as a function of node features, x. 

           Positional arguments are

           * x ( ( k x n-dim tensor) : the k-dim node features corresponding to each of the n nodes.

           Returns
 
           * couplings (n x n-dim tensor) : the n x n coupling/weight matrix.
           * mask (n-dim tensor)          : the indices used in sampling. Used later for plotting.'''
        num_units = x.shape[0]

        if self.rand_inds:
            mask = torch.randperm(num_units)[:self.batch_size].to(x.device).detach() # sample self.batch_size nodes
        else:
            mask = torch.tensor(np.arange(num_units)).to(x.device) # Otherwise mask comprises all network indices.

        if self.fixed_couplings:
            return self.fixed[mask,:][:,mask], mask
        else:
            # Subsampled node features
            _x = x[mask]
            _x = torch.cat([_x[:,None].repeat(1,self.batch_size,1),_x[None,:].repeat(self.batch_size,1,1)],dim=2) # all pairs

            #Infer couplings
            couplings = self.layers(_x.view(-1, _x.shape[-1])).squeeze().reshape(self.batch_size, self.batch_size)
            # Normalize for fixed degree
            couplings = (softmax(couplings.reshape(-1),dim=-1) * (self.avg_deg * self.batch_size)).reshape(self.batch_size, self.batch_size)

            # Symmetrize if necessary
            if self.symmetric:
                couplings = .5*(couplings + couplings.transpose(1,0))

            self.current_couplings = couplings.detach().clone()
            self.current_mask      = mask.detach().clone()
            self.current_cx        = c_x(x[mask], couplings).detach().cpu().numpy()

        return couplings, mask
