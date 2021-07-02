### Extended README

You can change the following values in `experiments.cfg` to configure an experiment: 

#### Saving/Loading/Data

* `exp_name` A name for your experiment.
* `save_dir`  Full path for where you model will be saved. 
* `data_base_dir` Super directory where all data is stored; e.g., the eventual full path to data will be`'data_base_dir/data_name/dist_name/train_or_test/features.npz'`.
* `num_samples`  (int) How many total node features to generate (or have been generated).  This value is not used for image experiments. 
* `pretrained`  (bool) Whether or not to load pretrained model.
* `data_names` (comma-separated list) Names of each of the types of data associated to a node; e.g.,  `omega,h,tau` (see `data.py`).
* `dist_names` (comma-separated list) How each of these data types is distributed (see `data.py`). 
* `num_classes`(int) How many classes associated to data. For non-classification experiments (e.g. global synchrony), must be set to 0. 
* `download` (bool) Whether or not to download BSDS image data. 

#### Dynamics

* `model_type` one of `full` or `xy`. `full` is the fully disordered model, while `xy` has only the interaction terms in the dynamics. Use the latter for images, mixture distributions.
* `num_units ` (int) Size of the networks used in training. 
* `burn_in_steps` (int) Number of timesteps to discard before calculating loss. 
* `gd_steps` (int) Number of timesteps over which to integrate the loss
* `alpha` (float) Width of timestep.
* `batch_size` (int) Number of nodes to sample at a time during ODE solving (set to `num_units` for small networks).  Must be <= `num_units`. 
* `initial_phase` Either 'zero' or 'normal'. If 'zero', then all phases are initialized at 0 for each dynamics. If 'normal', then initialized from standard gaussian bump on circle. Need the latter to break symmetry in the `xy` model.
* `adjoint` (bool) Set to True to use adjoint method for calculating gradients. 
* `solver_method` Which solver to use; e.g. 'euler'. See [2]. 

#### Neural Architecture

* `avg_deg` (float) If the couplings are node-normalized, the value of the average weighted node degree. 
* `num_hid_units` (int) Size of hidden layer in coupling neural network. 
* `symmetric` (bool) Set to True for undirected couplings. 
* `normalize` Either 'node', 'graph' or 'none'. If 'node', then forces graphs to have constant average node degree given by `avg_deg`. If 'graph', then graph is normalized to have weight matrix of mass 1. If 'none', then weight matrix is not normalized. This can result in `NaNs` in the gradient if either `alpha` or `lr` is too large. 
* `set_gain` (bool) If true, then use manual Xavier weight initialization with gain `gain`.
* `gain` (float) The gain to use if doing manual weight initializations.

#### Optimization

* `device` Device on which to run the model: cpu or cuda. 
* `num_epochs` (int) How many full passes through the data to take. 
* `loss_type` Which loss to use (e.g. `circular_variance`, `circular_moments`, or `cohn_loss`). Latter is experimental. 
* `lr ` (float) Learning rate for optimizer. 
* `optimizer` Which optimizer to use (e.g. `'SGD'`)
* `momentum` (float) Momentum value for SGD. 
* `max_grad_norm` (float) Clipping value for gradient norm. 
* `show_every` (int) Period at which to display training results. 
* `num_eval_batches` (int) How many testing batches to evaluate each epoch. 
* `verbose` (bool) Boolean for whether any intermittent results should be displayed. 
* `rand_inds` (bool) Set to True for random solver updating. Must be true when `batch_size` <= `num_units`. Mostly useful for evaluation and visualization to avoid memory problems.
