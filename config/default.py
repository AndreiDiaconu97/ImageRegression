import torch

OUT_ROOT = "../out"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hparams = {
    "epochs_per_stage": 100,
    "epochs_per_correction": 1,
    "boost_rate": 1.0,
    "lr_model": 0.001,
    "lr_ensemble": 0.001,
    "batch_size": 20000,
    "B_scale": 0.03,
    "hidden_size": 32,
    "model": 'gon',
    "optimizer": 'adamW',
    "batch_sampling_mode": 'whole',
    "num_nets": 50,
    "acc_gradients": False,
    # "input_size": image.shape
}
