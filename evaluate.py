import torch
from models import KuraNet
from torchvision.datasets import DatasetFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import circular_variance, c_x, save_object, yield_zero_column
import numpy as np
import argparse
from tqdm import tqdm
from configparser import ConfigParser
import os, csv
import ipdb

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

experiment_names = ['DEFAULT', 'Eef', 'Etau']
results = {}

for experiment in experiment_names:
    print('Evaluating {}.'.format(experiment))

    results[experiment] = {}

    # Load experiment parameters
    config = ConfigParser()
    config.read('experiments.cfg')
    config_dict = {}
    for (key, val) in config.items(experiment):
        config_dict[key] = val

    exp_name = config_dict['exp_name']
    save_dir = config_dict['save_dir']
    data_base_dir = config_dict['data_base_dir']
    num_units = int(config_dict['num_units'])
    num_samples = int(config_dict['num_samples'])
    num_epochs = int(config_dict['num_epochs'])
    device=config_dict['device']
    batch_size= int(config_dict['batch_size'])
    omega_name=config_dict['omega_name']
    h_name=config_dict['h_name']
    tau_name=config_dict['tau_name']
    avg_deg = float(config_dict['avg_deg'])
    num_hid_units= int(config_dict['num_hid_units'])
    gd_steps = int(config_dict['gd_steps'])
    burn_in_steps = int(config_dict['burn_in_steps'])
    alpha = float(config_dict['alpha'])
    lr=float(config_dict['lr'])
    optimizer=config_dict['optimizer']
    momentum= float(config_dict['momentum'])
    show_every = int(config_dict['show_every'])
    verbose = int(config_dict['verbose'])
    symmetric = config.getboolean(experiment, 'symmetric')
    grad_thresh = float(config_dict['grad_thresh'])
    num_batches = int(float(num_samples) / num_units)
    solver_method=config_dict['solver_method']
    adjoint = config.getboolean(experiment, 'adjoint')

    save_path = os.path.join(save_dir, exp_name)

    # Load data
    train_dirs = {}
    test_dirs  = {}
    train_dss  = {}
    test_dss   = {}
    train_dls  = {}
    test_dls   = {}

    if omega_name != 'degenerate':
        train_dirs['omega'] = os.path.join(config_dict['data_base_dir'], omega_name, 'train')
        test_dirs['omega'] = os.path.join(config_dict['data_base_dir'], omega_name, 'test')
        train_dss['omega'] = DatasetFolder(train_dirs['omega'], np.load, 'npy')
        test_dss['omega'] = DatasetFolder(test_dirs['omega'], np.load, 'npy')
        train_dls['omega'] = DataLoader(train_dss['omega'], batch_size=num_units, shuffle=False, drop_last=True)
        test_dls['omega'] = DataLoader(test_dss['omega'], batch_size=num_units, shuffle=False, drop_last=True)
    else:
        train_dls['omega'] = yield_zero_column(num_batches,num_units)
        test_dls['omega'] = yield_zero_column(num_batches,num_units)
    if h_name != 'degenerate':
        train_dirs['h'] = os.path.join(config_dict['data_base_dir'], h_name, 'train')
        test_dirs['h'] = os.path.join(config_dict['data_base_dir'], h_name, 'test')
        train_dss['h'] = DatasetFolder(train_dirs['h'], np.load, 'npy')
        test_dss['h'] = DatasetFolder(test_dirs['h'], np.load, 'npy')
        train_dls['h'] = DataLoader(train_dss['h'], batch_size=num_units, shuffle=False, drop_last=True)
        test_dls['h'] = DataLoader(test_dss['h'], batch_size=num_units, shuffle=False, drop_last=True)
    else:
        train_dls['h'] = yield_zero_column(num_batches,num_units)
        test_dls['h'] = yield_zero_column(num_batches,num_units)
    if tau_name != 'degenerate':
        train_dirs['tau'] = os.path.join(config_dict['data_base_dir'], tau_name, 'train')
        test_dirs['tau'] = os.path.join(config_dict['data_base_dir'], tau_name, 'test')
        train_dss['tau'] = DatasetFolder(train_dirs['tau'], np.load, 'npy')
        test_dss['tau'] = DatasetFolder(test_dirs['tau'], np.load, 'npy')
        train_dls['tau'] = DataLoader(train_dss['tau'], batch_size=num_units, shuffle=False, drop_last=True)
        test_dls['tau'] = DataLoader(test_dss['tau'], batch_size=num_units, shuffle=False, drop_last=True)
    else:
        train_dls['tau'] = yield_zero_column(num_batches,num_units)
        test_dls['tau'] = yield_zero_column(num_batches,num_units)

    # Load network and control
    cn = connectivity_net(3, avg_deg=avg_deg, num_hid_units=num_hid_units,
                          alpha=alpha, gd_steps=gd_steps, burn_in_steps=burn_in_steps,
                          solver_method=solver_method).to(device)
    cn_control = connectivity_net(3, avg_deg=avg_deg, num_hid_units=num_hid_units,
                          alpha=alpha, gd_steps=gd_steps, burn_in_steps=burn_in_steps,
                          solver_method=solver_method).to(device)
    #cn_control = connectivity_net(3, avg_deg=avg_deg, num_hid_units=num_hid_units).to(device)
    cn.load_state_dict(torch.load(os.path.join(save_path, 'model.pt')))

    cn.set_batch_size(batch_size)
    cn_control.set_batch_size(batch_size)

    with torch.no_grad():
        # Test Data
        print('Data generalization')
        cn.eval()
        cn_control.eval()
        cvh_test = []
        cvh_test_control = []

        cn.set_grids(alpha,1000,gd_steps)
        cn_control.set_grids(alpha,1000,gd_steps)

        for j, ((omega, _),(h, _), (tau, _)) in tqdm(enumerate(zip(test_dls['omega'], test_dls['h'], test_dls['tau']))):
            break
            tau = torch.where(tau > 40.0, 40.0 * torch.ones_like(tau),tau)
            x = torch.cat([omega,h, tau],dim=-1).to(device)

            trajectory         = cn.run(x).data
            trajectory_control = cn_control.run(x).data

            cv = circular_variance(trajectory[-gd_steps:].data)
            cv_control = circular_variance(trajectory_control[-gd_steps:].data)
            cvh_test.append(cv.detach().cpu().numpy())
            cvh_test_control.append(cv_control.detach().cpu().numpy())

        test_mean = np.mean(cvh_test)
        test_mean_control = np.mean(cvh_test_control)
        print('Test data CV: {}. Test data CV control: {}'.format(test_mean, test_mean_control))
        results[experiment]['test_data'] = test_mean
        results[experiment]['test_data_control'] = test_mean_control

        # Test Size / Size + Data
        cn.set_batch_size(500)
        cn_control.set_batch_size(500)
        for d, dls in enumerate(([train_dls['omega'], train_dls['h'], train_dls['tau']],[test_dls['omega'], test_dls['h'], test_dls['tau']])):

            cn.set_grids(alpha,10000,gd_steps)
            cn_control.set_grids(alpha,10000,gd_steps)

            cn.rand_inds = True
            cn_control.rand_inds = True
            if d == 0:
                print('Size generalization')
            else:
                print('Size + data generalization')
            omega_dl = dls[0]
            h_dl     = dls[1]
            tau_dl   = dls[2]
            big_omega = []
            big_h     = []
            big_tau   = []
            print('Collating features...')
            for j, ((omega, _),(h, _), (tau, _)) in tqdm(enumerate(zip(omega_dl, h_dl, tau_dl))):
                big_omega.append(omega)
                big_h.append(h)
                big_tau.append(tau)
            print('Running dynamics...') 
            big_omega = torch.stack(big_omega).reshape(num_samples,1)
            big_h     = torch.stack(big_h).reshape(num_samples,1)
            big_tau   = torch.stack(big_tau).reshape(num_samples,1)
 
            big_tau = torch.where(big_tau > 40.0, 40.0 * torch.ones_like(big_tau),big_tau)
            x = torch.cat([big_omega,big_h, big_tau],dim=-1).to(device)

            trajectory         = cn.run(x).data
            trajectory_control = cn_control.run(x).data

            cv = circular_variance(trajectory[-gd_steps:].data)
            cv_control = circular_variance(trajectory_control[-gd_steps:].data)
 
            nm = 'test_size' if d == 0 else 'test_size_data'
            if d == 0: 
                print('Test size CV: {}. Test size CV control: {}'.format(cv, cv_control))
            else:
                print('Test size + data CV: {}. Test size + data CV control {}'.format(cv,cv_control))
            results[experiment][nm] = cv
            results[experiment][nm + '_control'] = cv_control

name = os.path.join(save_path, 'eval')
save_object(results,name)
