import subprocess
import argparse
from distutils.util import strtobool
import os 
from data import make_data
from configparser import ConfigParser
import torch
import ipdb

# Use this script to train multiple experiments at once. 

argparser = argparse.ArgumentParser()
argparser.add_argument('--experiments', nargs='+', required=True)
argparser.add_argument('--seed_search', type=lambda x:bool(strtobool(x)), default=True)
argparser.add_argument('--data_dir', type=str, default='~')
argparser.add_argument('--num_samples', type=int,default=10000)
argparser.add_argument('--seed', type=int,default=0)
argparser.add_argument('--num_seeds', type=int,default=1)
argparser.add_argument('--device', type=str,default='cpu')
argparser.add_argument('--generate_data', type=lambda x:bool(strtobool(x)), default=False)
args = argparser.parse_args()

if args.device == 'cuda':
    num_devices=torch.cuda.device_count()
    device_prefix = 'CUDA_VISIBLE_DEVICES={}'.format(device)
else:
    num_devices = 1
    device_prefix = ''

config = ConfigParser()
config.read('experiments.cfg')

config_dict = {}

if args.generate_data:
    print('Generating data.')
    for exp in args.experiments:
        for (key, val) in config.items(exp):
            config_dict[key] = val
        data_names, dist_names = config_dict['data_names'].split(','), config_dict['dist_names'].split(',')
        for dan, din in zip(data_names, dist_names):
            if din == 'degenerate': continue
            make_data(dan,din, **config_dict)

for e, exp in enumerate(args.experiments):
    device = e % num_devices
    if not args.seed_search:
        best_seed = args.seed
        num_seeds = 0
    else:
        best_seed = -1
        num_seeds = args.num_seeds
    subprocess.call(device_prefix + ' python train.py --name {} --num_seeds {} --best_seed {}&'.format(exp, num_seeds, best_seed),shell=True)
