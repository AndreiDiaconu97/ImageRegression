import numpy as np
import torch
from utils import BatchSamplingMode

# Ensure deterministic behavior
# torch.backends.cudnn.deterministic = True
# random.seed(hash("setting random seeds") % 2**32 - 1)
# np.random.seed(hash("improves reproducibility") % 2**32 - 1)
# torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
# torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

OUT_ROOT = "out"
INPUT_PATH = "data/mountains_final.jpg"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_batches = device
device_image = device
device_pred_img = device

USE_LESS_VRAM = False
if USE_LESS_VRAM:
    # device_batches = 'cpu'  # enable only as last resort
    device_image = 'cpu'
    device_pred_img = 'cpu'


def init_P(P, image):
    P["image_shape"] = image.shape
    if not P["B_scale"]:
        P["input_layer_size"] = 2
    else:
        P["input_layer_size"] = P["hidden_size"]
    P["use_less_vram"] = USE_LESS_VRAM

    if P["batch_sampling_mode"] == BatchSamplingMode.whole.name:
        P["batch_size"] = np.prod(P["image_shape"][:2])
        P["n_batches"] = 1
    else:
        P["n_batches"] = np.prod(P["image_shape"][:2]) // (P["batch_size"])

    if P["batch_sampling_mode"] == BatchSamplingMode.sequence.name:
        P["n_batches"] += 1


#### EXPERIMENTS CONFIGS ################################################################

hparams_base = {
    'type': 'base',
    'B_scale': 30,  # set False for no posEnc # NOTE: consider w0=30 in Siren, with 4 hidden set w0=250 if not using positional encoding
    'w0': 30,  # only used if model=siren
    'acc_gradients': False,
    'batch_sampling_mode': BatchSamplingMode.nth_element.name,
    'shuffle_batches': True,
    'batch_size': 5000,
    'epochs': 10000,
    'hidden_size': 256,
    'hidden_layers': 5,
    'lr': 1e-4,  # [0.01, 0.0001],
    'lr_patience': 1000,
    'model': 'siren',  # relu
    'optimizer': 'adam',  # RMSprop, adam, adamW, SGD
    'scale': 1.0,
}

hparams_grownet = {
    'type': 'grownet',
    'B_scale': 30,  # set False for no posEnc # NOTE: consider w0=30 in Siren, with 4 hidden set w0=250 if not using positional encoding
    'w0': 30,  # only used if model=siren
    'acc_gradients': False,
    'batch_sampling_mode': BatchSamplingMode.nth_element.name,
    'shuffle_batches': True,
    'batch_size': 5000,
    'boost_rate': 1.0,
    'epochs_per_correction': 20,
    'epochs_per_stage': 20,
    'hidden_size': 256,
    'hidden_layers': 1,
    'lr_ensemble': 1e-4,
    'lr_model': 1e-4,
    'lr_patience_model': 2,
    'lr_patience_ensemble': 2,
    'model': 'siren',
    'num_nets': 17,
    'optimizer': 'adam',
    'scale': 0.2,
}

hparams_xgboost = {
    'type': 'xgboost',
    'B_scale': 30,
    'eval_metric': ['rmse'],  # 'mae'
    'posEnc_size': 16 * 2,
    'lambda': 1,
    'learning_rate': 1.0,
    'max_depth': 7,
    # 'n_estimators': 10000,
    'num_boost_round': 10000,
    # 'num_parallel_tree': 4, # for CPU?
    'tree_method': 'gpu_hist',
    'objective': 'reg:squarederror',
    'reg_lambda': 0.01,
    'scale': 0.2,
    'subsample': 1,
    'desired_psnr': 40
}

#########################################################################################

# TODO:
# check if loss is correct - OK
# move scale into P - OK
# implement acc_gradients - OK
# implement an universal object to wrap wandb logging and metrics - OK
# add tunable n_layers - OK
# check support for gon model - OK
# add more metrics and organize them - OK
# log network model memory size - OK
# implement checkpoints saving and loading - OK
# log n_parameters in xgboost - OK
# try without input mapping - OK
# compare imageRegression with fourier-feature-networks repo - OK?
# clean code to also work with zero correction epochs - OK
# try forward_grad() without boostrate - OK
# zero_grad order in fully corrective step may be wrong - OK
# TODO: model seems to saturate at the end, maybe batch size and B is not that important
# solve the grownet corrective step mystery
