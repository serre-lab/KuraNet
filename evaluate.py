import torch
from models import KuraNet_full, KuraNet_xy
from torch.utils.data.dataset import TensorDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import circular_variance, c_x, save_object
import numpy as np
import argparse
from tqdm import tqdm
from configparser import ConfigParser
import os, csv

# This script is used to run evaluation. Supply one or more experiments (config headings) as a comma separated list using the shell argument --experiments to return the three evaluation metrics described in the companion manuscript. 

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--experiments', nargs='+', required=True)
args = parser.parse_args()

results = {}

for experiment in args.experiments:
    print('Evaluating {}.'.format(experiment))

    results[experiment] = {}

    # Load experiment parameters
    config = ConfigParser()
    config.read('experiments.cfg')
    config_dict = {}
    for (key, val) in config.items(experiment):
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
    rand_inds         = config.getboolean(experiment, 'rand_inds')
    pretrained        = config.getboolean(experiment, 'pretrained')
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
    normalize         = config.getboolean(experiment, 'normalize')
    symmetric         = config.getboolean(experiment, 'symmetric')
    num_batches       = int(float(num_samples) / num_units)
    rand_inds         = config.getboolean(experiment, 'rand_inds')
    adjoint           = config.getboolean(experiment, 'adjoint')
    measure_cx        = config.getboolean(experiment, 'measure_cx')

    save_path = os.path.join(save_dir, exp_name)

    train_dls    = {}
    test_dls     = {}
    train_dts    = {}
    test_dts     = {}

    for dl, dt, regime in zip([train_dls, test_dls], [train_dts, test_dts],['train', 'test']):
        for dist_name, data_name in zip(dist_names, data_names):
            dt[data_name] = {}
            if dist_name != 'degenerate':
                dt[data_name][regime] = np.load(os.path.join(data_base_dir, data_name, dist_name, regime, 'features.npz'))
                ds = TensorDataset(torch.FloatTensor(dt[data_name][regime]['x']), torch.LongTensor(dt[data_name][regime]['y'].astype(np.int32)))
            else:
                dt[data_name][regime] = {'x': torch.zeros(num_samples).float(), 'y' : torch.zeros(num_samples).long()}
                ds = TensorDataset(torch.zeros(num_samples).float(), torch.zeros(num_samples).long())              
            dl[data_name] = DataLoader(ds, batch_size=num_units, shuffle=True, drop_last=True)

    if num_classes == 'lookup':
        num_classes= len(set(dt[data_name][regime]['y']).union(set(dt[data_name][regime]['y'])))

 # Initialize models
    KN_model = KuraNet_full if model_type == 'full' else KuraNet_xy
    kn = KN_model(feature_dim, avg_deg=avg_deg, num_hid_units=num_hid_units,
                          rand_inds=rand_inds,normalize=normalize,
                          adjoint=adjoint, solver_method=solver_method,
                          alpha=alpha, initial_phase=initial_phase,
                          gd_steps=gd_steps, burn_in_steps=burn_in_steps,
                          set_gain=set_gain, gain=gain).to(device) 
    kn_control = KN_object(feature_dim, avg_deg=avg_deg, num_hid_units=num_hid_units,
                          rand_inds=rand_inds,normalize=normalize,
                          adjoint=adjoint, solver_method=solver_method,
                          alpha=alpha, initial_phase=initial_phase,gd_steps=gd_steps,
                          burn_in_steps=burn_in_steps,
                          set_gain=set_gain, gain=gain).to(device) 
 
    kn.load_state_dict(torch.load(os.path.join(save_path, 'model.pt')))

    kn.set_batch_size(batch_size)
    kn_control.set_batch_size(batch_size)

    # Set loss function
    if loss_type == 'circular_variance':
        loss_func = circular_variance
    elif loss_type == 'circular_moments':
        loss_func = circular_moments
    elif loss_type == 'cohn_loss':
        loss_func = cohn_loss
    else:
        raise Exception('Loss type not recognized.')

    data_keys = [key for key in train_dls.keys()]
=
    with torch.no_grad():
        # Test Data
        print('Data generalization')
        kn.eval()
        kn_control.eval()
        loss_test = []
        loss_test_control = []

        kn.set_grids(alpha,1000,gd_steps)
        kn_control.set_grids(alpha,1000,gd_steps)
        for i, batch in tqdm(enumerate(zip(*[test_dls[key] for key in data_keys]))):
            break
            X = {key : x.float().to(device) for (key, (x,_)) in zip(data_keys, batch)}
            Y = {key : y for (key, (_,y)) in zip(data_keys, batch)} 

            # This is only used for cluster synchrony experiments
            if num_classes > 0:
                masks = make_masks(Y,num_classes,device)
            else:
                masks = None

	    # Fix max delay for memory problems
            if 'tau' in X.keys():
                X['tau']  = torch.where(X['tau'] > 40.0, 40.0 * torch.ones_like(X['tau']),X['tau'])
            # Run model, get trajectory
            trajectory         = kn.run(X)
            trajectory_control = kn_control.run(X)
           
            # Calculate and record loss. Update
            ll = loss_func(trajectory[-gd_steps:].data)
            ll_control = loss_func(trajectory_control[-gd_steps:].data)

            loss_test.append(ll.detach().cpu().numpy())
            loss_test_control.append(ll_control.detach().cpu().numpy())

        test_mean = np.mean(loss_test)
        test_mean_control = np.mean(loss_test_control)
        print('Test data loss: {}. Test data loss control: {}'.format(test_mean, test_mean_control))
        results[experiment]['test_data'] = test_mean
        results[experiment]['test_data_control'] = test_mean_control

        # Test Size / Size + Data
        kn.set_batch_size(500)
        kn_control.set_batch_size(500)
        for d, (regime, dt) in enumerate(zip(['train', 'test'], [train_dts, test_dts])):
            X = {key : torch.tensor(dt[key][regime]['x']).float().to(device) for key in data_keys}
            Y = {key : torch.tensor(dt[key][regime]['y']) for key in data_keys} 
            # This is only used for cluster synchrony experiments
            if num_classes > 0:
                masks = make_masks(Y,num_classes,device)
            else:
                masks = None

	    # Fix max delay for memory problems
            if 'tau' in X.keys():
                X['tau']  = torch.where(X['tau'] > 40.0, 40.0 * torch.ones_like(X['tau']),X['tau'])
            # Run model, get trajectory
            

            kn.set_grids(alpha,5000,1)
            kn_control.set_grids(alpha,5000,1)

            kn.rand_inds = True
            kn_control.rand_inds = True
            if d == 0:
                print('Size generalization')
            else:
                print('Size + data generalization')

            trajectory         = kn.run(X, full_trajectory=False).data
            trajectory_control = kn_control.run(X, full_trajectory=False).data

            ll = loss_func(trajectory.data).cpu().numpy()
            ll_control = loss_func(trajectory_control.data).cpu().numpy()
 
            nm = 'test_size' if d == 0 else 'test_size_data'
            if d == 0: 
                print('Test size loss: {}. Test size loss control: {}'.format(ll, ll_control))
            else:
                print('Test size + data loss: {}. Test size + data loss control {}'.format(ll,ll_control))
            results[experiment][nm] = ll
            results[experiment][nm + '_control'] = ll_control

name = os.path.join(save_path, 'eval')
save_object(results,name)
