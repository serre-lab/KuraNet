import subprocess
import argparse
from distutils.util import strtobool
import os 
from data import make_all_data
from configparser import ConfigParser
import torch

argparser = argparse.ArgumentParser()
argparser.add_argument('--experiments', nargs='+', required=True)
argparser.add_argument('--seed_search', type=lambda x:bool(strtobool(x)), default=True)
argparser.add_argument('--data_dir', type=str, default='/media/data_cifs/projects/prj_synchrony/data')
argparser.add_argument('--num_samples', type=int,default=10000)
argparser.add_argument('--seed', type=int,default=0)
argparser.add_argument('--num_seeds', type=int,default=1)
argparser.add_argument('--device', type=str,default='cpu')
argparser.add_argument('--generate_data', type=lambda x:bool(strtobool(x)), default=False)
args = argparser.parse_args()

if args.generate_data:
    print('Generating data.')
    make_all_data(num_samples=args.num_samples,data_dir=args.data_dir)

if args.device == 'cuda':
    num_devices=torch.cuda.device_count()
    device_prefix = 'CUDA_VISIBLE_DEVICES={}'.format(device)
else:
    num_devices = 1
    device_prefix = ''
#experiments = ['Etau']
config = ConfigParser()
config.read('experiments.cfg')

for e, exp in enumerate(args.experiments):
    device = e % num_devices + 1
    if not args.seed_search:
        best_seed = args.seed
        num_seeds = 0
    else:
        best_seed = -1
        num_seeds = args.num_seeds
    subprocess.call(device_prefix + ' python train.py --name {} --num_seeds {} --best_seed {}&'.format(exp, num_seeds, best_seed),shell=True)
