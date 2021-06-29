## KuraNet: A fully differentiable Kuramoto model for understanding disordered oscillator networks

This repository contains the code for a fully-differentiable Kuramoto model in the form of "KuraNet". KuraNet, based on reference [1], is used for understanding the relationship between disordered node features and dynamical behavior by modeling this relationship as a differentiable, parametrized function. This function is a deep neural network trained by gradient descent using [torchdiffeq](https://www.github.com/rtqichen/torchdiffeq "torchdiffeq") [2].

## Requirements and installation

All code is written in Pytorch 1.4.0. using Python 3.5.2. After cloning this repository, run

`pip install -r requirements.txt`

to install all necessary requirements (besides Python and CUDA backend).

## Examples

Demos for three experiments are contained in the notebooks `global.ipynb`, `cluster.ipynb` and `images.ipynb`. For example, here is (top) KuraNet learning to synchronize oscillators on a sparse graph (optimized left, control right) and (bottom) an image of an elk that KuraNet has learned to synchronize:

![kuramoto](./kuramoto.gif)

![elk](./elk.gif)

## Basic Usage

We consider a general Kuramoto model of the form
$$
\frac{d\theta_j}{dt} = f(\theta, I, K),
$$
where `K` is a coupling matrix and`I` is a random sample of intrinsic oscillator features comprising natural frequencies, external field strengths and transmission delays. Suppose a particular dynamical state (e.g. global synchrony; `theta_i ~= theta_j` for all `ij`) minimizes the loss function `L`. The goal of KuraNet is to model the differentiable function `I --> K` as a neural network so that `L` is minimized on average over realizations of `I`. KuraNet is built to model the relationship between disordered node features and couplings which gives rise to collective oscillator behavior. 

To train a single model, first edit the config file `experiments.cfg` according to the experiment you'd like to run (for explanation of config fields, see below). This file also contains several default experiments. Next, call

`python train.py --name <EXP_NAME> --num_seeds <n>` 

in terminal, where `<EXP_NAME>` is the config file heading corresponding to the experiment you'd like to run and `<n>` is the number of random seeds you'd like to test. For instance, to train KuraNet on the default experiment in the config file using 1 seed, call `python train.py --name DEFAULT --num_seeds 1`.  

Make sure that KuraNet has access to the data you'd like inside the folder designated by the field `data_base_dir` in the config file. You can also generate all of the data in reference [1] by calling the function `make_all_data` in `data.py`. 

You can also run multiple experiments at once by calling 

`python run.py --experiments <EXP_NAME1> <EXP_NAME2> ... <EXP_NAMEn> --device <device>`

where `<EXP_NAMEi> `is the heading corresponding to experiment i in the config file and `<device>`is either `cpu`or `cuda`. (CPU usage is slow, so GPU is recommended.). 

### Experiment configuration

Details for each field in `experiments.cfg ` can be found in the [extended README](C:\Users\Matt Ricci\Desktop\extended_README.md). 

### References

[1]  Ricci, M., Jung, M., Zhang, Y., Chalvidal, M., Soni, A., & Serre, T. (2021). KuraNet: Systems of Coupled Oscillators that Learn to Synchronize, 1–9. Retrieved from http://arxiv.org/abs/2105.02838

[2]  Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural Ordinary Differential Equations. In 32nd Conference on Neural Information Processing Systems (NeurIPS 2018). Montréal, Canada: Curran Associates. https://doi.org/10.1007/978-3-662-55774-7_3
