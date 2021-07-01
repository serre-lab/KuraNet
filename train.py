import os
import torch
import numpy as np
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
import time
from utils import circular_variance, c_x, save_object, circular_moments, cohn_loss, make_masks
from configparser import ConfigParser
import argparse
import copy
from models import KuraNet_xy, KuraNet_full
import ipdb

def optimize_connectivity_net(num_units, feature_dim, train_dls, test_dls, model_type='full', normalize='node', avg_deg=1.0, symmetric=True, num_classes=0,num_epochs=10,
                              batch_size=None, burn_in_steps=100, gd_steps=50, alpha=.1, solver_method='euler', adjoint=False, initial_phase='zero',
                              loss_type='circular_variance', optimizer='Adam', lr=.01, momentum=0.0, max_grad_norm=10.0, set_gain=False, gain=1.0,
                              num_hid_units=128, verbose=0, measure_cx=False, show_every=50,num_eval_batches=1,rand_inds=False,
                              pretrained=False,save_path='~',device='cpu'):

    ''' optimize_connectivity_net: Train an instance of Kuranet on the provided data. Initializes a model, 
        trains it, and evaluates it over several epochs. 

        Positional arguments are

        * num_units (int)    : the number of units/nodes in the underlying graph on which dynamics will be run. Range: [2,infty)
        * feature_dim (int)  : the dimension of the feature associated to each node. Range: [1,infty)
        * train_dls (dict)   : dictionary containing the training data for each type of node feature (which can come from different data sets).  
        * test_dls (dict)    : " testing data "

        Keyword arguments are

        * model_type (str, default='full')     : which KuraNet: 'full' gives the model will all intrinsic node features. 'xy' is reduced model (faster than 'full' with zero features)
        * normalize (bool, default=True)       : whether or not couplings are normalized. Type of normalization depends on model_type. 
        * avg_deg (float, default=1.0)         : the average degree of the underlying graph. Range : (0,infty)
        * num_classes (int, default=0)         : number of classes in the data set. If not a classification task (e.g. global synchrony), set to 0. Range: [0,infty)
        * num_epochs (int, default=10)         : number of training and testing epochs. Range : [1, infty)
        * batch_size (int/None, default=None)  : the dynamic batch size used by solver. If None, set to num_units (i.e. the whole graph is updated at once). Range : [1,num_units]
        * burn_in_steps (int, default=100)     : # of steps before the loss is integrated. Range : [0, infty)
        * grad_steps (int, default=50)         : # of steps during which the loss is integrated after burn in. Total timesteps are burn_in_steps + grad_steps.  Range : [0, infty)
        * alpha (float, default=.1)            : the step size for the ODE solver. Range : (0,infty)
        * solver_method (str, default='euler') : solver method for ODE. Range : see torchdiffeq documentation
        * adjoint (bool, default=False)        : whether or not to calculate parameter gradients by solving an adjoint system. See torchdiffeq documentation. 
        * optimizer (str, deafult='Adam')      : which optimizer to use for learning connectivity net parameters (Note: this is not the ODE solver). Range : {'SGD', 'Adam'}
        * lr (float, default=.01)              : optimizer learning rate. Range : (0,infty)
        * momentum (float, default=0.0)        : optimizer momentum. Relevant for SGD and potentially custom solvers. 
        * max_grad_norm (float, default=10.0)  : the maximum allowed euclidean norm for the gradient of all concatenated model parameters. Range: (0, infty)
        * num_hid_units (int, default=128)     : the number of hidden neurons in each of the connectivity net's layers. Range : (1,infty)
        * verbose (int, deafult=0)             : controls how much information is displayed during training. Currently binary for either no inffformation or all. Range : (0, infty)
        * show_every (int, deafult=50)         : period for displaying training information. Range : (1, infty)
        * measure_cx (bool, default=False)     : whether or not to measure the c_x statistic. 
        * num_eval_batches (int, default=1)    : how many testing batches to run each epoch.Low values speed up training but are worse for estimating testing error. Range : (1, infty)
        * rand_inds (bool, default=False)      : make True to use random sampling during ODE update. Usually only needed for evaluation on very large networks.
        * pretrained (bool, default=False)     : whether or not a model is loaded from save_path. 
        * save_path (str, default='~')         : path from which to load a pretrained model. 
        * device (str, default='cpu')          : device to which tensors are cast. Range : {'cpu', 'cuda'}
    
        Returns
        
        * lossh_train (list) : list containing losses for every gradient descent step across all epochs. 
        * lossh_test (list)  : list containing average loss for all testing batches evaluated within an epoch for each epoch.
        * kn (KuraNet)       : the trained KuraNet object.
        * cxh (list)         : list containing the statistic cx measured at each iteration.
        '''
    
    # If batch_size not speccified, set it as num_units as long as network is small. Otherwise, set to 100. 
    assert batch_size <= num_units, "batch_size can be at most num_units"

    # Initialize model
    KN_model = KuraNet_full if model_type == 'full' else KuraNet_xy
    kn = KN_model(feature_dim, avg_deg=avg_deg,
                 num_hid_units=num_hid_units, initial_phase=initial_phase,
                 rand_inds=rand_inds,normalize=normalize,
                 adjoint=adjoint, solver_method=solver_method,set_gain=set_gain, gain=gain,
                 alpha=alpha, gd_steps=gd_steps,
                 burn_in_steps=burn_in_steps).to(device) 
    kn.set_batch_size(batch_size)

    # load network if necessary
    if pretrained:
        kn.load_state_dict(torch.load(os.path.join(save_path,model.pt)))

     # Set loss function
    if loss_type == 'circular_variance':
        loss_func = circular_variance
    elif loss_type == 'circular_moments':
        loss_func = circular_moments
    elif loss_type == 'cohn_loss':
        loss_func = cohn_loss
    else:
        raise Exception('Loss type not recognized.')

    # Initialize optimizer
    if optimizer == 'Adam':
        opt = torch.optim.Adam(kn.parameters(), lr=lr)
    elif optimizer == 'SGD':
        opt = torch.optim.SGD(kn.parameters(), lr=lr, momentum=momentum)

    # Initialize observables and data objects
    lossh_train = []
    lossh_test  = []
    cxh         = []

    data_keys = [key for key in train_dls.keys()]
    train_dls = [train_dls[key] for key in data_keys]	
    test_dls = [test_dls[key] for key in data_keys]	

    # Begin training
    for e in range(num_epochs):
        print('Training. Epoch {}.'.format(e))
        # Train 
        kn.train()
        for i, batch in enumerate(zip(*train_dls)):
            X = {key : x.float().to(device) for (key, (x,_)) in zip(data_keys, batch)}
            Y = {key : y for (key, (_,y)) in zip(data_keys, batch)} 

            # This is only used for cluster synchrony experiments
            if num_classes > 0:
                masks = make_masks(Y,num_classes,device)
            else:
                masks = None
            start = time.time()
            opt.zero_grad()		

	    # Fix max delay for memory problems
            if 'tau' in X.keys():
                X['tau']  = torch.where(X['tau'] > 40.0, 40.0 * torch.ones_like(X['tau']),X['tau'])
            # Run model, get trajectory
            trajectory = kn.run(X)
    
            # Calculate and record loss. Update
            truncated_trajectory = trajectory[-gd_steps:,...]
            ll = loss_func(truncated_trajectory, masks=masks)
            lossh_train.append(ll.detach().cpu().numpy())
            ll.backward()
            norm=torch.nn.utils.clip_grad_norm_(kn.parameters(), max_norm=max_grad_norm, norm_type=2)
            opt.step()

            # Calculate statistic
            if measure_cx:
                cxh.append(kn.current_cx)

            # Logging
            stop = time.time()
            if verbose > 0 and (i % show_every) == 0:
                print('Training batch: {}. Time/Batch: {}. Loss: {}. Gradient norm: {}.'.format(i, np.round(stop - start,4), lossh_train[-1], min(max_grad_norm,norm)))
        # Testing 
        print('Testing. Epoch {}'.format(e))
        with torch.no_grad():
            kn.eval()
            lossh_test_epoch = []
            for j, batch in enumerate(zip(*test_dls)):
                X = {key : x.float().to(device) for (key, (x,_)) in zip(data_keys, batch)}
                Y = {key : y for (key, (_,y)) in zip(data_keys, batch)}
                # This is only used for cluster synchrony experiments
                if num_classes > 0:
                    masks = make_masks(Y,num_classes,device)
                else:
                    masks = None
            
                start = time.time()
                opt.zero_grad()		
    
    	        # Fix max delay for memory problems
                if 'tau' in X.keys():
                    X['tau']  = torch.where(X['tau'] > 40.0, 40.0 * torch.ones_like(X['tau']),X['tau'])
                start = time.time()
                # Fix max delay for memory problems

                trajectory = kn.run(X)
                truncated_trajectory = trajectory[-gd_steps:,...]
            
                ll = loss_func(truncated_trajectory, masks=masks)
                lossh_test_epoch.append(ll.detach().cpu().numpy())

                stop = time.time()
                if verbose > 0 and (j % show_every) == 0:
                    print('Testing batch: {}. Time/Batch: {}. Loss: {}.'.format(j, np.round(stop - start,4), lossh_test_epoch[-1]))
                if j > num_eval_batches: break
 
            lossh_test.append(np.mean(lossh_test_epoch))
           
    return lossh_train, lossh_test, kn, cxh

if __name__=='__main__':
    ''' Main script for training a single model over potentially multiple random initializations. Loads experimental parameters
        from the experiments.cfg config file and then trains a model with the funtion optimize_connectivity_net. Shell parameters are
        
        * name (str)      : the header in experiments.cfg corresponding to the experiment to be run.
        * num_seeds (int) : the number of random seeds to search over. Only the model corresponding to the best seed will be saved. Range: 1+
        * best_seed (int) : a specific seed to be run. If num_seeds > 0, best_seed should be set to negative to indicate that no best seed has been prespecified. Range: -infty - infty.
        '''

    # Parse shell arguments
    meta_parser = argparse.ArgumentParser()
    meta_parser.add_argument('--name', type=str, default='DEFAULT')
    meta_parser.add_argument('--num_seeds', type=int, default=10)
    meta_parser.add_argument('--best_seed', type=int, default=-1)
    meta_args = meta_parser.parse_args()

    if meta_args.best_seed >= 0 and meta_args.num_seeds > 0:
        raise ValueError('If you want to search over multiple random seeds, set argument best_seed to -1!')

    # Load experimental parameters
    config = ConfigParser(allow_no_value=True)
    config.read('experiments.cfg')
    config_dict = {}
    for (key, val) in config.items(meta_args.name):
        config_dict[key] = val

    exp_name          = config_dict['exp_name']
    save_dir          = config_dict['save_dir']
    data_base_dir     = config_dict['data_base_dir']
    data_names        = config_dict['data_names'].split(',')
    dist_names        = config_dict['dist_names'].split(',')
    model_type        = config_dict['model_type']
    feature_dim       = int(config_dict['feature_dim'])
    num_classes       = config_dict['num_classes'] 
    num_classes       = int(num_classes) if num_classes.isnumeric() else num_classes
    num_units         = int(config_dict['num_units'])
    num_samples       = int(config_dict['num_samples'])
    num_epochs        = int(config_dict['num_epochs'])
    pretrained        = config.getboolean(meta_args.name, 'pretrained')
    device            = config_dict['device']
    solver_method     = config_dict['solver_method']
    batch_size        = int(config_dict['batch_size'])
    avg_deg           = float(config_dict['avg_deg'])
    num_hid_units     = int(config_dict['num_hid_units'])
    gd_steps          = int(config_dict['gd_steps'])
    alpha             = float(config_dict['alpha'])
    initial_phase     = config_dict['initial_phase']
    burn_in_steps     = int(config_dict['burn_in_steps'])
    loss_type         = config_dict['loss_type']
    lr                = float(config_dict['lr'])
    optimizer         = config_dict['optimizer']
    momentum          = float(config_dict['momentum'])
    max_grad_norm     = float(config_dict['max_grad_norm'])
    set_gain          = config.getboolean(meta_args.name, 'set_gain')
    gain              = float(config_dict['gain'])
    show_every        = int(config_dict['show_every'])
    num_eval_batches  = int(config_dict['num_eval_batches'])
    verbose           = int(config_dict['verbose'])
    normalize         = config_dict['normalize']
    symmetric         = config.getboolean(meta_args.name, 'symmetric')
    num_batches       = int(float(num_samples) / num_units)
    rand_inds         = config.getboolean(meta_args.name, 'rand_inds')
    adjoint           = config.getboolean(meta_args.name, 'adjoint')
    measure_cx        = config.getboolean(meta_args.name, 'measure_cx')

    # Create directory where model and intermitten results will be saved.  
    save_path = os.path.join(save_dir, exp_name)
    if not os.path.exists(os.path.join(save_dir, exp_name)):
        os.makedirs(os.path.join(save_dir, exp_name))

    # Load training and testing data.
    train_dls  = {}
    test_dls   = {}

    for dl, regime in zip([train_dls, test_dls], ['train', 'test']):
        for dist_name, data_name in zip(dist_names, data_names):
            if dist_name != 'degenerate':
                dt = np.load(os.path.join(data_base_dir, data_name, dist_name, regime, 'features.npz'))
                ds = TensorDataset(torch.FloatTensor(dt['x']), torch.LongTensor(dt['y'].astype(np.int32)))
            else:
                ds = TensorDataset(torch.zeros(num_samples,1).float(), torch.zeros(num_samples).long())
            dl[data_name] = DataLoader(ds, batch_size=num_units, shuffle=True, drop_last=True)

    if num_classes == 'lookup':
        num_classes= len(set(dt['y']).union(set(dt['y'])))

    # Train model
    all_train_losses = []
    all_test_losses = []
    all_cx = []
    current_best_model = ([],-1,[],[],np.inf) # model, seed, cx, test loss
    for seed in range(meta_args.num_seeds):
        np.random.seed(seed)
        torch.manual_seed(seed)

        loss_train, loss_test, kn, cx = optimize_connectivity_net(num_units,feature_dim, train_dls, test_dls, model_type=model_type,avg_deg=avg_deg,num_classes=num_classes,
							      pretrained=pretrained, save_path=save_path,  num_epochs=num_epochs, batch_size=batch_size,
                                                              burn_in_steps=burn_in_steps, gd_steps=gd_steps,alpha=alpha, initial_phase=initial_phase,
                                                              symmetric=symmetric, normalize=normalize,solver_method=solver_method, adjoint=adjoint,
                                                              loss_type=loss_type, optimizer=optimizer, lr=lr, momentum=momentum, max_grad_norm=max_grad_norm,\
                                                              set_gain=set_gain, gain=gain,
                                                              num_hid_units=num_hid_units, verbose=verbose, show_every=show_every, measure_cx=measure_cx,
                                                              num_eval_batches=num_eval_batches, rand_inds=rand_inds, device=device)
        # Save each train loss curve and the final evaluation loss. 
        if loss_test[-1] < current_best_model[-1]: current_best_model = (kn.cpu(),seed,cx,loss_train,loss_test[-1])

    # Save model and experimental information
    torch.save(current_best_model[0].state_dict(), os.path.join(save_path, 'model.pt')) 
   
    dict = config_dict 
    dict['train_loss'] = np.array(current_best_model[3])
    dict['test_loss'] = np.array(current_best_model[4])
    dict['best_seed'] = current_best_model[1]
    if measure_cx:
        dict['c_x'] = cx

    name = os.path.join(save_path, 'results')
    save_object(dict, name)
