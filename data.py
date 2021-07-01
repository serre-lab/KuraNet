import os
import torch
import numpy as np
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdmn
import subprocess
from torch.distributions import uniform, cauchy, normal, relaxed_bernoulli, negative_binomial
from scipy.io import loadmat
from skimage.io import imread
import cv2 as cv
import os
import urllib, tarfile
import ipdb

data_info_dict_BSDS500 = {
    "90076":{"mode":"train", "seg_ind":4, "groups":[[11], [9], [7]]},
    "118020":{"mode":"train", "seg_ind":3, "groups":[[12, 10, 8], [1], [2,3], [4]]},
    "124084":{"mode":"train", "seg_ind":0, "groups":[[2], [3], [4], [5]]},
    "189011":{"mode":"train", "seg_ind":0, "groups":[[11], [7], [5]]},
    "229036":{"mode":"val", "seg_ind":2, "groups":[[3], [4], [5]]},
    "247003":{"mode":"test", "seg_ind":0, "groups":[[2], [3], [4]]},
    "296028":{"mode":"test", "seg_ind":0, "groups":[[1], [6], [8], [10]]},
    "317080":{"mode":"train", "seg_ind":2, "groups":[[1,3], [4,5,6], [7]]},
    "361010":{"mode":"val", "seg_ind":3, "groups":[[5,14,17,27], [13,15,18,20], [12]]}
}

def get_BSDS_info(image_tag):
    return data_info_dict_BSDS500[image_tag]

# Fetch distribution objects for intrinsic frequencies

def get_dist(dist_name,**kwargs):
    if dist_name == 'cauchy':
        loc = kwargs['loc']
        scale = kwargs['scale']
        return cauchy.Cauchy(loc, scale)
    elif dist_name == 'uniform':
        high = kwargs['high']
        low = kwargs['low']
        return uniform.Uniform(low, high)
    elif dist_name == 'normal':
        loc = kwargs['loc']
        scale = kwargs['scale']
        return normal.Normal(loc, scale)
    elif dist_name == 'bernoulli':
        loc = kwargs['loc']
        return custom_bernoulli(loc)
    elif dist_name == 'exponential':
        loc = kwargs['loc']
        return exponential.Exponential(loc)
    elif dist_name == 'geometric':
        loc = kwargs['loc']
        return geometric.Geometric(loc)
    elif dist_name == 'discrete_uniform':
        low = kwargs['low']
        high = kwargs['high']
        probs = torch.linspace(low,high, high - low + 1)
        return categorical.Categorical(probs)
    elif dist_name == 'negative_binomial':
        r = kwargs['scale']
        probs = kwargs['loc']
        return negative_binomial.NegativeBinomial(r,probs)
    elif dist_name == 'GMM':
        num_classes = int(kwargs['num_classes'])
        centroids = [10*np.array([np.cos(2*np.pi * i / num_classes), np.sin(2*np.pi * i / num_classes)]) for i in range(num_classes)]
        cov_matrices = 1.0*np.array([np.eye(2) for _ in range(num_classes)])
        return GMM(n_components=num_classes,
                             centroids=centroids,
                             cov_matrices=cov_matrices)
    elif dist_name == 'Moons':
        return Moons(noise=float(kwargs['noise']), random_state=None)
    elif dist_name == 'Spirals':
        return Spirals(noise = float(kwargs['noise']))
    elif data_name == 'Circles':
        return Circles(noise=float(kwargs['noise']),random_state=None)

class Moons(object):
    def __init__(self, noise = 0.05, random_state=1):
        self.noise = noise
        self.random_state = random_state
        self.n_components = 2

    def sample(self, n_samples=400):
        if n_samples > 1:
            x, y = datasets.make_moons(n_samples,
                            noise = self.noise,
                            random_state=self.random_state)
        else:
             x, y = datasets.make_moons(512,
                            noise = self.noise,
                             random_state=self.random_state)
             ind = np.random.randint(512)
             x = x[ind,...]
             y = y[ind]
      
        return x, y

    def train_test(selfn_samples_trainn_samples_test):
        return self.sample(n_samples_train), self.sample(n_samples_test)
   
class Circles(object):
    def __init__(self, noise=.1, factor = 0.5, random_state=None):
        self.factor = factor
        self.random_state = random_state
        self.n_components = 2
        self.noise = noise

    def sample(self, n_samples=400):
        if n_samples > 1:
            x, y = datasets.make_circles(n_samples,
                            noise  = self.noise,
                            factor = self.factor,
                             random_state=self.random_state)
        else:
             x, y = datasets.make_circles(512,
                            noise = self.noise,
                            factor = self.factor,
                            random_state=self.random_state)
             ind = np.random.randint(512)
             x = x[ind,...]
             y = y[ind]
        return x, y 

class GMM(object):
    def __init__(self, n_components, centroids, cov_matrices):
        self.n_components = n_components
        self.centroids = centroids
        self.cov_matrices = cov_matrices

    def one_sampling(self,statistics):
        return np.random.multivariate_normal(statistics[0], statistics[1])

    def sample(self, n_samples):
        idx = np.random.randint(0, self.n_components,n_samples)
        list_stat = [[self.centroids[i],self.cov_matrices[i]] for i in idx]
        return np.array((np.vstack(list(map(self.one_sampling,list_stat))))), np.array(idx)

    def train_test(selfn_samples_trainn_samples_test):
        return self.sample(n_samples_train), self.sample(n_samples_test)

class Spirals(object):
    def __init__(self, noise):
        self.noise = noise
        self.n_components = 2

    def sample(self, n_samples):
        #n_samples=max(1,int(n_samples/2))
        branch = 1*(np.random.rand((n_samples)) < .5)

        sgn   = np.cos(np.pi*branch)
        theta = np.sqrt(np.random.rand(n_samples))*2*np.pi # np.linspace(0,2*pi,100)
        r_a = sgn*(2*theta + np.pi)
        data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
        x_a = data_a + np.random.randn(n_samples,2)*self.noise

        return x_a, branch

def load_img_BSDS500(file_id, mode, seg_ind,
                     data_dir):
    """
    Parameters:
    file_id (str):id of image
    mode (str): train|val|test
    seg_ind (int): 0~4
    """
    img = imread(os.path.join(data_dir, 'images/{}/{}.jpg'.format(mode, file_id)))
    mat = np.squeeze(loadmat(os.path.join(data_dir, 'groundTruth/{}/{}.mat'.format(mode, file_id)))['groundTruth'])
    mat = mat[seg_ind][0][0][0]
    return img, mat

def generate_data_BSDS500(data_dir,image_dir=None,train_prop=.5,download=False):
    """
    Parameters:
    repeat_num (int): number of different train-test splits
    train_prop (float): proportion for training (0~1)
    data_dir (str): path for saving created datasets
    """
    from os.path import expanduser
    home = expanduser("~")

    image_dir = home if image_dir is None else image_dir
    if download:
        url='http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz'
        save_path =  os.path.join(home, 'BSR_bsds500.tgz')
        urllib.request.urlretrieve(url, save_path)
        file = tarfile.open(save_path)
        file.extractall(os.path.join(image_dir, 'BSDS'))
        file.close()
        subprocess.call('rm -rf save_path&', shell=True)

    np.random.seed(0)
    # loop through each selected images
    full_BSDS_path = os.path.join(image_dir, 'BSDS/BSR/BSDS500/data/')
    for d in data_info_dict_BSDS500:
        img, mat = load_img_BSDS500(
            file_id=d, 
            mode=data_info_dict_BSDS500[d]['mode'], 
            seg_ind=data_info_dict_BSDS500[d]['seg_ind'], 
            data_dir=full_BSDS_path)
        group_matrix = np.zeros_like(mat)
        # for each group, find the pixels. The rest pixels will be background
        for i, g in enumerate(data_info_dict_BSDS500[d]['groups']):
            for e in g:
                group_matrix[np.where(mat == e)] = i + 1
        # create features as the concatenation of location and intensity
        h, w, _ = img.shape
        xx = np.repeat(np.linspace(0,w,w)[None,:], h, axis=0)[:,:,None] / w
        yy = np.repeat(np.linspace(0,h,h)[:,None], w, axis=1)[:,:,None] / h
        X = np.concatenate([img / 255, xx, yy], axis=-1).reshape(-1, 5)
        y = group_matrix.reshape(-1)

        # save data
        train_num = int(train_prop*y.shape[0])
        inds = np.random.permutation(y.shape[0])
        for regime in ['train', 'test']:
            full_dir = os.path.join(data_dir, d,regime)
            if not os.path.exists(full_dir):
                 os.makedirs(full_dir)
            if regime is 'train':
                np.savez(os.path.join(full_dir, 'features.npz'),
                     x=X[inds[:train_num]],
                     y=y[inds[:train_num]])
            else:
                np.savez(os.path.join(full_dir, 'features.npz'),
                     x=X[inds[train_num:]],
                     y=y[inds[train_num:]])

def make_data(data_name, dist_name,**kwargs):

    '''make data : generates and saves data.
	
	Positional arguments are
	
	* data_name (str) : A tag for the "type" of data you are saving. For non-image data, this is used as a directory name and as a way for KuraNet to know where to put dynamically
	                    relevant parameters, like omega, h and tau. For BSDS, it is additionally used to save and generate images.
	* dist_name (str) : A tag for the distribution by which the data tagged by `data_name` is distributed. For instance, the data with `data_name` "omega" could be distributed as `uniform1` or `uniform2`. 
	
	Keyword arguments are
	
	* data_base_dir (str) : super directory in which data will be stored. 
	* num_samples (int)   : how many samples to generate. For image data, this argument is not needed. 
	* num_classes (int)   : for multi-class data, how many classes are associated with the data. 
	* download (bool)     : whether or not to download BSDS data.'''
	
	# Set random seeds and return generator object except for images where data is saved using other code above. 
    if dist_name == 'uniform1':
        seed = 0
        generator = get_dist('uniform',low=-1.0,high=1.0)
        is_torch = True
    elif dist_name == 'uniform2':
        seed = 1
        generator = get_dist('uniform', low=2.0, high=4.0)
        is_torch = True
    elif dist_name == 'negative_binomial':
        seed = 2
        generator = get_dist('negative_binomial',loc=.5,scale=15)
        is_torch = True
    elif dist_name == 'GMM':
        seed = 3
        generator = get_dist(dist_name, num_classes = int(kwargs['num_classes']))
        is_torch = False
    elif dist_name == 'Moons':
        seed = 4
        generator = get_dist(dist_name)
        is_torch = False
    elif dist_name == 'Circles':
        seed = 5
        generator = get_dist(dist_name,noise=.05)
        is_torch = False
    elif dist_name == 'Spirals':
        seed = 6
        generator = get_dist(dist_name,noise=.5)
        is_torch = False
    elif data_name == 'BSDS':
	    # Note this is data_name and not dist_name!
        data_dir    = os.path.join(kwargs['data_base_dir'], data_name)
        generate_data_BSDS500(data_dir=data_dir,download=kwargs['download'])
        return True
    else:
        raise Exception('Distribution name not recognized!')
		
	# Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
	
    # Unpack kwargs
    data_base_dir = kwargs['data_base_dir']
    num_samples = int(kwargs['num_samples'])
	
    # Where training and testing data will be saved. 
    data_dir    = os.path.join(data_base_dir, data_name, dist_name)

    # Delete it if it already exists. 
    if os.path.exists(data_dir):
        subprocess.call('rm -rf {}'.format(data_dir), shell=True)

    # Using the generator object, save npz file containing relevant node features. 
    for regime in ['train', 'test']:
        full_dir = os.path.join(data_dir, regime)
        if not os.path.exists(full_dir):
            os.makedirs(full_dir)
        if is_torch:
            x = generator.sample(sample_shape=torch.Size([num_samples,]))
            if len(x.shape) < 2 : x = x.unsqueeze(-1)
            x = x.numpy()
            y = np.zeros_like(x)
        else:
            x,y = generator.sample(n_samples=num_samples)
            if len(x.shape) < 2: x = x.reshape(-1,1)
        full_path = os.path.join(full_dir, 'features.npz') 
        np.savez(full_path,x=x,y=y)
