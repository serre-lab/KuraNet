import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
plt.style.use('ggplot')
from matplotlib import cm
from matplotlib.lines import Line2D
import torch
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader
import numpy as np
import argparse
from utils import load_object, circular_variance, c_x, save_object, yield_zero_column
from models import KuraNet
from configparser import ConfigParser
import os, csv

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

what_to_visualize=['loss', 'c_x']#, 'degree_generalize']#, 'degree_generalize', 'spectrum']

base_fig_dir = '/media/data_cifs/projects/prj_synchrony/results/matt_results/brede/figures/'

experiment_names = ['DEFAULT', 'Eef', 'Etau']

for experiment in experiment_names:

    print('Visualizing {}.'.format(experiment))

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
    alpha = float(config_dict['alpha'])
    lr=float(config_dict['lr'])
    optimizer=config_dict['optimizer']
    momentum= float(config_dict['momentum'])
    show_every = int(config_dict['show_every'])
    verbose = int(config_dict['verbose'])
    symmetric = config.getboolean(experiment, 'symmetric')
    grad_thresh = float(config_dict['grad_thresh'])
    num_batches = int(float(num_samples) / num_units)
    delay_mask_precomputed = config.getboolean(experiment, 'delay_mask_precomputed')

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
    cn = connectivity_net(3, avg_deg=avg_deg, num_hid_units=num_hid_units).to(device)
    cn_control = connectivity_net(3, avg_deg=avg_deg, num_hid_units=num_hid_units).to(device)
    cn.load_state_dict(torch.load(os.path.join(save_path, 'model.pt')))

    # Load results
    fn ='/media/data_cifs/projects/prj_synchrony/results/models/brede/{}/results'
    results_dict = load_object(fn.format(exp_name))

    fig_dir = os.path.join(base_fig_dir, exp_name)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    if 'loss' in what_to_visualize:
        # Training loss
        print('Saving training loss...')
        loss = results_dict['train_loss']
        fig, ax = plt.subplots(figsize=(10,10))
        ax.plot(loss)
        ax.set_xlabel('Iterations', fontsize=42)
        ax.set_ylabel(r'$V$', fontsize=42)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        fig.savefig(os.path.join(fig_dir,'train_loss.png'))
   
    if 'c_x' in what_to_visualize: 
        # C_X 
        # TODO: Add tau!
        print('Saving c_x plots...')
        all_c_x = [np.array(load_object(fn.format(en))['c_x']) for en in ['base_brede', 'external_field', 'tau']]

        all_cvh = [load_object(fn.format(en))['train_loss'] for en in ['base_brede', 'external_field', 'tau']]
        fig, ax = plt.subplots(figsize=(10,10))

        colors = ['green','orange','purple']
        coeffs = []
        r2s = []
        text_positions = ([-.76, .025], 
                          [-.825, .11],
                          [-.85, .55])
        for c, (color, c_x, cvh) in enumerate(zip(colors, all_c_x, all_cvh)):
            coeff = np.polyfit(c_x, cvh, 1)
            ax.scatter(c_x, cvh, color=color, alpha=.5)

            # r-squared
            p = np.poly1d(coeff)

            # fit values, and mean
            yhat = p(c_x)                         # or [p(z) for z in x]
            ybar = np.sum(cvh)/len(cvh)          # or sum(y)/len(y)
            ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
            sstot = np.sum((cvh - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
            r2 =  ssreg / sstot
            coeffs.append(coeff)
            r2s.append(r2)
            ax.text(text_positions[c][0], text_positions[c][1], r'$r^2={}$'.format(str(r2)[:4]),color=color, fontsize=24)
            ax.plot(c_x, yhat, color=color, linewidth=4)

        ax.set_xlabel(r'$c_x$', fontsize=42)
        ax.set_ylabel(r'$V$', fontsize=42)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        legend_elements = [Line2D([0], [0], marker='o',alpha=.5, linestyle='None', color='green',label=r'Case I: $\omega$', markersize=15),
                        Line2D([0], [0], marker='o',alpha=.5, linestyle='None', color='orange',label=r'Case II: $(\omega, b$)', markersize=15),
                        Line2D([0], [0], marker='o',alpha=.5, linestyle='None',color='purple',label=r'Case III: $(\omega, \tau$)', markersize=15)]
 
        ax.legend(handles=legend_elements, loc='lower right', fontsize=24, facecolor='white')
        fig.savefig(os.path.join(fig_dir, 'c_x.png'))
        plt.xlim([np.array(all_c_x).min() - .05, np.array(all_c_x).max() + .05])

    if 'degree_generalize' in what_to_visualize:
        num_samples = 10
        num_avg_deg = 25
        max_avg_deg = 5
        degs = np.linspace(0, max_avg_deg, num_avg_deg)
        
        all_cv = []
        all_cv_control = []
        for ad in degs:
        
            print('Running average degree %.4f.' % ad)
            cn.avg_deg = ad
            cn_control.avg_deg = ad
        
            sample_cv = []
            sample_cv_control = []
            init_phase = 2*np.pi*torch.rand((num_units,)).to(device)
        
            for i, ((omega,_),(h,_),(tau,_)) in enumerate(zip(train_dls['omega'], train_dls['h'], train_dls['tau'])):
                if i > num_samples - 1:
                    break
                # Fix max delay for memory problems
                tau = torch.where(tau > 40.0, 40.0 * torch.ones_like(tau),tau)
                x = torch.cat([omega,h, tau],dim=-1).to(device)

                if delay_mask_precomputed:
                    T_neg =  tau.max().long() + 1
                    delays = tau.squeeze().repeat(batch_size,1).long() # Every unit has the same transmission delay to its neighbors
                    delays = delays - torch.diag_embed(torch.diag(delays)) # Remove self-delays
                    delay_mask = torch.zeros((T_neg,batch_size,batch_size)).to(x.device)
                    for j in range(batch_size):
                        for k in range(batch_size):
                            delay_mask[-1*(delays[j,k] + 1),j,k] = 1
                else:
                    delay_mask = None

                _, all_phase, _, _ = cn.forward(x, batch_size,
                                                   burn_in_steps=500,
                                                   gd_steps=1, alpha=alpha,
                                                   init_phase=init_phase,
                                                   return_connectivity=True,
                                                   rand_inds=False,delay_mask=delay_mask)
                _, all_phase_control, _, _ = cn_control.forward(x, batch_size,
                                                   burn_in_steps=500,
                                                   gd_steps=1, alpha=alpha,
                                                   init_phase=init_phase,
                                                   return_connectivity=True,
                                                   rand_inds=False,delay_mask=delay_mask)
                

                sample_cv.append(circular_variance(all_phase[-100:]))
                sample_cv_control.append(circular_variance(all_phase_control[-100:]))
            all_cv.append(np.array(sample_cv))
            all_cv_control.append(np.array(sample_cv_control))

        synchrony = 1 - np.array(all_cv).mean(1)
        synchrony_control = 1 - np.array(all_cv_control).mean(1)
        synchrony_std = (1 - np.array(all_cv)).std(1)
        synchrony_control_std = (1 - np.array(all_cv_control)).std(1)
        
        fig, ax = plt.subplots(figsize=(10,10))
        plt.plot(degs, synchrony, color='r')
        plt.fill_between(degs, synchrony-synchrony_std, synchrony+synchrony_std,
            alpha=0.2, edgecolor='r', facecolor='r',antialiased=True)
        plt.plot(degs, synchrony_control, color='b')
        plt.fill_between(degs, synchrony_control-synchrony_control_std, synchrony_control+synchrony_control_std,
            alpha=0.2, edgecolor='b', facecolor='b',antialiased=True)
        plt.legend(('Optimized', r'Randomized'), loc='lower right',fontsize=24.)
        plt.xlabel(r'$\langle d \rangle$', fontsize=42)
        plt.ylabel(r'$r$', fontsize=42)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        fig.savefig(os.path.join(fig_dir, 'degree_generalize.png'))

    if 'flow' in what_to_visualize: 
        # Example flow
        print('Saving example flows...')
        big_omega = []
        big_h = []
        big_tau = []
        for ((omega,_),(h,_),(tau,_)) in zip(train_dls['omega'], train_dls['h'], train_dls['tau']):
            big_omega.append(omega)
            big_h.append(h)
            big_tau.append(tau)
        big_omega = torch.stack(big_omega).reshape(num_samples,1)
        big_h     = torch.stack(big_h).reshape(num_samples,1)
        big_tau   = torch.stack(big_tau).reshape(num_samples,1)

        # Fix max delay for memory problems
        big_tau = torch.where(big_tau > 40.0, 40.0 * torch.ones_like(big_tau),big_tau)
        x = torch.cat([big_omega,big_h, big_tau],dim=-1).to(device)

        print('Optimized flow...')
        _, flow, _, _ = cn.forward(x, batch_size,
                                   burn_in_steps=10000,
                                   gd_steps=1, alpha=alpha,
                                   return_connectivity=True,
                                   rand_inds=True, delay_mask=None)

        print('Control flow...')
        _, flow_control, _, _ = cn_control.forward(x, batch_size,
                                           burn_in_steps=10000,
                                           gd_steps=1, alpha=alpha,
                                           return_connectivity=True,
                                           rand_inds=True, delay_mask=None)
        fig, ax = plt.subplots(figsize=(10,10)) 
        for (flw, color) in zip([flow_control, flow], ['blue', 'red']):
            ax.plot(flw.detach().cpu().numpy(), color=color, alpha=.01) 
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='red', lw=4),
                Line2D([0], [0], color='blue', lw=4)]
        ax.legend(custom_lines, ('Optimized', 'Control'), fontsize=24)
        ax.set_xlabel(r'$t$',fontsize=42)
        ax.set_ylabel(r'$\theta$',fontsize=42)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, 'flow.png'))

    
    # Generalize degree
    # Spectrum

#if __name__=='__main__':

#    argparser = argparse.ArgumentParser()
#    argparser.add_argument('--name',type=str, default='base_brede')
#    argparser.add_argument('--seed',type=int, default=0)
#    args = argparser.parse_args()

#    load_dir = '/media/data_cifs/projects/prj_synchrony/results/models/brede'
#    fig_dir = '/media/data_cifs/projects/prj_synchrony/results/matt_results/brede/figures/{}'.format(args.name)
#    if not os.path.exists(fig_dir):
#        os.makedirs(fig_dir)

#    dict = load_object('/media/data_cifs/projects/prj_synchrony/results/models/brede/{}/results'.format(args.name))

#    fig, ax = plt.subplots(figsize=(10,10))
#    all_c_x = []
#    legend_elements = []
#    colors = []
#     
#    if 'c_omegas' in dict.keys():
#        all_c_x.append(dict['c_omegas'])
#        colors.append('orange')
#        legend_elements.append(Line2D([0], [0], color='orange', linestyle='None', alpha=.5, marker='o', label=r'$x=\omega$', markerfacecolor='orange', markersize=15))
#    
#    if 'c_inertias' in dict.keys():
#        all_c_x.append(dict['c_inertias'])
#        colors.append('green')
#        legend_elements.append(Line2D([0], [0], color='green', linestyle='None', alpha=.5, marker='o', label=r'$x=m$', markerfacecolor='green', markersize=15))
#
#    if 'c_ext_fields' in dict.keys():
#        all_c_x.append(dict['c_ext_fields'])
#        colors.append('purple')
#        legend_elements.append(Line2D([0], [0], color='purple', linestyle='None', alpha=.5, marker='o', label=r'$x=b$', markerfacecolor='purple', markersize=15))
#
#    bins = np.linspace(-1,1,200)
#    R = 1 - np.array(dict['losses']).reshape(-1)
#    bin_centers = [(bins[b] + bins[b + 1]) / 2. for b in range(len(bins) - 1)]
#    num, _ = np.histogram(R, bins=bins)
#
#    for c, c_x in enumerate(all_c_x):
#        c_x = np.array(c_x).reshape(-1)
#        R_means = []
#        R_stds = []
#        for b in range(len(bins) - 1):
#            if b == len(bins) - 1:
#                bin_mask = (bins[b] <= c_x) * (c_x <= bins[b + 1])
#            else:
#                bin_mask = (bins[b] <= c_x) * (c_x < bins[b + 1])
#
#            bin_R = R[bin_mask]
#            R_means.append(bin_R.mean())
#            R_stds.append(bin_R.std())
#    
#        R_means = np.array(R_means)
#        R_stds = np.array(R_stds)    
#
#        im = plt.plot(bin_centers, R_means, color=colors[c], marker='o', linestyle='None')
#        plt.fill_between(bin_centers, R_means - R_stds, R_means+R_stds,
#        alpha=0.2, edgecolor=colors[c], facecolor=colors[c],antialiased=True)
#
#    plt.xticks(fontsize=24)
#    plt.yticks(fontsize=24)
#    plt.legend(handles=legend_elements, loc='best', fontsize=24)
#    plt.ylim([0,1])
#    plt.xlim([-1,1])
#
#    ax.set_xlabel(r'$c_x$', fontsize=42)
#    plt.ylabel(r'$\langle R \rangle$', fontsize=42)
#    plt.savefig(os.path.join(fig_dir, 'c_x.png'))
#    plt.close()
#
#    fig, ax = plt.subplots(figsize=(10,10))
#    for c, c_x in enumerate(all_c_x):
#        c_x = np.array(c_x)
#        c_x_mean = c_x.mean(0)
#        c_x_std  = c_x.std(0) 
#        plt.plot(c_x_mean, color=colors[c])
#        plt.fill_between(np.arange(len(c_x_mean)), c_x_mean - c_x_std, c_x_mean+c_x_std,
#            alpha=0.2, edgecolor=colors[c], facecolor=colors[c],antialiased=True)
#    plt.xlabel('Iterations')
#    plt.xticks(fontsize=24)
#    plt.yticks(fontsize=24)
#    
#    plt.ylim([-1,1])
#    plt.xlabel('Iterations', fontsize=42)
#    plt.ylabel(r'$c_x$', fontsize=42)
#    plt.legend(handles=legend_elements, loc='best', fontsize=24)
#    plt.savefig(os.path.join(fig_dir,'c_x_opt.png'),bbox_inches='tight')
