import numpy as np
import torch
from utils import BatchSamplingMode

# Ensure deterministic behavior
# torch.backends.cudnn.deterministic = True
# random.seed(hash("setting random seeds") % 2**32 - 1)
# np.random.seed(hash("improves reproducibility") % 2**32 - 1)
# torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
# torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_ROOT = "../out"
INPUT_PATH = "../data/IMG_0201_DxO.jpg"


def init_P(P, image):
    P["image_shape"] = image.shape
    P["input_layer_size"] = P["hidden_size"] * 2

    if P["batch_sampling_mode"] == BatchSamplingMode.whole.name:
        P["batch_size"] = np.prod(P["image_shape"][:2])
        P["n_batches"] = 1
    else:
        P["n_batches"] = np.prod(P["image_shape"][:2]) // (P["batch_size"])

    if P["batch_sampling_mode"] == BatchSamplingMode.sequence.name:
        P["n_batches"] += 1


hparams_grownet = {
    'type': 'grownet',
    'B_scale': 0.07,
    'acc_gradients': False,
    'batch_sampling_mode': BatchSamplingMode.nth_element.name,
    'shuffle_batches': True,
    'batch_size': 10000,
    'boost_rate': 1.0,
    'epochs_per_correction': 0,
    'epochs_per_stage': 25,
    'hidden_size': 128,
    'hidden_layers': 1,
    'lr_ensemble': 0.001,
    'lr_model': 0.001,
    'model': 'siren',
    'num_nets': 30,
    'optimizer': 'adamW',
    'scale': 0.1,
}

hparams_base = {
    'type': 'base',
    'B_scale': 0.07,
    'acc_gradients': False,
    'batch_sampling_mode': BatchSamplingMode.nth_element.name,
    'shuffle_batches': True,
    'batch_size': 5000,
    'epochs': 2000,
    'hidden_size': 256,
    'hidden_layers': 3,
    'lr': 0.000728,  # [0.01, 0.0001],
    'model': 'siren',
    'optimizer': 'adamW',
    'scale': 0.1,
}

hparams_xgboost = {
    'type': 'xgboost',
    'B_scale': 0.03,
    'eval_metric': ['rmse'],  # 'mae'
    'input_layer_size': 16 * 2,
    'lambda': 1,
    'learning_rate': 1.0,
    'max_depth': 7,
    # 'n_estimators': 10000,
    'num_boost_round': 10000,
    # 'num_parallel_tree': 4, # for CPU?
    'tree_method': 'gpu_hist',
    'objective': 'reg:squarederror',
    'reg_lambda': 0.01,
    'scale': 0.1,
    'subsample': 1,
    'desired_psnr': 35
}
# TODO:
# check if loss is correct - OK
# move scale into P - OK
# implement acc_gradients - OK
# implement an universal object to wrap wandb logging and metrics - OK
# add tunable n_layers - OK
# check support for gon model - OK
# add more metrics and organize them - OK
# log network model memory size - OK?
# try without input mapping
# compare imageRegression with fourier-feature-networks repo
# implement checkpoints saving and loading
# save final results and models of best runs
# TODO: clean code to also work with zero correction epochs
# log n_parameters in xgboost
