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
    "type": "grownet",
    "B_scale": 0.03,
    "acc_gradients": False,
    "batch_sampling_mode": BatchSamplingMode.whole.name,
    "shuffle_batches": True,
    "batch_size": 20000,
    "boost_rate": 1.0,
    "epochs_per_correction": 1,
    "epochs_per_stage": 100,
    "hidden_size": 32,
    "hidden_layers": 1,
    "image_shape": None,
    "lr_ensemble": 0.001,
    "lr_model": 0.001,
    "model": 'siren',
    "num_nets": 50,
    "optimizer": 'adamW',
    "scale": 0.1
}

hparams_base = {
    "type": "base",
    "B_scale": 0.0676,
    "acc_gradients": False,
    "batch_sampling_mode": BatchSamplingMode.nth_element.name,
    "shuffle_batches": True,
    "batch_size": 2272,
    "epochs": 5000,
    "hidden_size": 128,
    "hidden_layers": 3,
    "lr": 0.000728,  # [0.01, 0.0001],
    "model": 'siren',
    "optimizer": 'adamW',
    "scale": 0.1
}

hparams_xgboost = {
    "type": "xgboost",
    'eval_metric': ['mae', 'rmse'],
    'learning_rate': 0.6,
    'max_depth': 12,
    'n_estimators': 800,
    'num_parallel_tree': 1,
    'objective': 'reg:squarederror',
    'reg_lambda': 0.01,
    'scale': 0.1,
    'seed': 0,
    'subsample': 1.0,
    'tree_method': 'gpu_hist',
    'verbosity': 2
}

# TODO: prepare best default version for each script. use .py or .yaml?
