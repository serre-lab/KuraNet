import os
import torch
import numpy as np
from tqdm import tqdm
import subprocess
from torch.distributions import uniform, cauchy, normal, relaxed_bernoulli, negative_binomial
import ipdb

# Fetch distribution objects for intrinsic frequencies
def get_dist(dist_name,**kwargs):
    if dist_name == 'cauchy':
        loc = kwargs['loc']
        scale = kwargs['scale']
        dist = cauchy.Cauchy(loc, scale)
        return dist
    elif dist_name == 'uniform':
        high = kwargs['high']
        low = kwargs['low']
        dist = uniform.Uniform(low, high)
        return dist
    elif dist_name == 'normal':
        loc = kwargs['loc']
        scale = kwargs['scale']
        dist = normal.Normal(loc, scale)
        return dist
    elif dist_name == 'bernoulli':
        loc = kwargs['loc']
        dist = custom_bernoulli(loc)
        return dist
    elif dist_name == 'exponential':
        loc = kwargs['loc']
        dist = exponential.Exponential(loc)
        return dist
    elif dist_name == 'geometric':
        loc = kwargs['loc']
        dist = geometric.Geometric(loc)
        return dist
    elif dist_name == 'discrete_uniform':
        low = kwargs['low']
        high = kwargs['high']
        probs = torch.linspace(low,high, high - low + 1)
        dist = categorical.Categorical(probs)
        return dist
    elif dist_name == 'negative_binomial':
        r = kwargs['scale']
        probs = kwargs['loc']
        dist = negative_binomial.NegativeBinomial(r,probs)
        return dist

def make_all_data(num_samples=10000, data_dir='/media/data_cifs/projects/prj_synchrony/data'):
    data_names = ['o_uniform1', 'h_uniform1', 'o_uniform2', 'negative_binomial']
    data_dict = {'data_base_dir':data_dir, 'num_samples':num_samples}
    for dn in data_names:
        make_data(dn,**data_dict)

def make_data(data_name,**kwargs):
    if data_name == 'o_uniform1':
        seed = 0
        generator = get_dist('uniform',low=-1.0,high=1.0)
    elif data_name == 'h_uniform1':
        seed = 1
        generator = get_dist('uniform',low=-1.0,high=1.0)
    elif data_name == 'o_uniform2':
        seed = 2
        generator = get_dist('uniform', low=2.0, high=4.0)
    elif data_name == 'negative_binomial':
        seed = 3
        generator = get_dist('negative_binomial',loc=.5,scale=15)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data_base_dir = kwargs['data_base_dir']
    num_samples = int(kwargs['num_samples'])
    data_dir    = os.path.join(data_base_dir, data_name)
    if os.path.exists(data_dir):
        subprocess.call('rm -rf {}'.format(data_dir), shell=True)

    for regime in ['train', 'test']:
        os.makedirs(os.path.join(data_dir, regime, '0'))
        print('Generating {} {} data'.format(data_name, regime))
        for n in tqdm(range(num_samples)):
            x = generator.sample(sample_shape=torch.Size([1,])).numpy()
            full_path = os.path.join(data_dir,regime, '0', 'sample_%05d.npy' % n) 
            np.save(full_path,x)
