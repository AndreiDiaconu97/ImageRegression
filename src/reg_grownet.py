from adabound import adabound
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchmetrics import PSNR
from torchmetrics.functional import ssim
from torchmetrics.image import psnr
from torchvision import transforms
from torchviz import make_dot
from tqdm import tqdm
import cv2
import enum
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import shutil
import time
import torch
import torch.nn.functional as F
import torchvision
import wandb

# Ensure deterministic behavior
# torch.backends.cudnn.deterministic = True
# random.seed(hash("setting random seeds") % 2**32 - 1)
# np.random.seed(hash("improves reproducibility") % 2**32 - 1)
# torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
# torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)
from utils import input_mapping, BatchSamplingMode, get_optimizer, batch_generator_in_memory
from utils.grownet import DynamicNet
from config.default import OUT_ROOT, device


# os.environ["WANDB_DIR"] = "../runs"


class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()

    def init_weights(self):
        b = 1 / \
            self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)


def gon_model(num_layers, input_dim, hidden_dim):
    layers = [SirenLayer(input_dim, hidden_dim, is_first=True)]
    for i in range(1, num_layers - 1):
        layers.append(SirenLayer(hidden_dim, hidden_dim))
    layers.append(SirenLayer(hidden_dim, 3, is_last=True))

    return nn.Sequential(*layers)


class GON(nn.Module):
    def __init__(self, dim_in, hidden_size, num_layers):
        super(GON, self).__init__()
        layers = [SirenLayer(dim_in, hidden_size, is_first=True)]
        for i in range(1, num_layers - 1):
            layers.append(SirenLayer(hidden_size, hidden_size))

        self.model_part1 = nn.Sequential(*layers)
        self.model_part2 = SirenLayer(hidden_size, 3, is_last=True)

    def forward(self, x, prev_penultimate=None):
        if prev_penultimate is not None:
            x = torch.cat([x, prev_penultimate], dim=1)
        penultimate = self.model_part1(x)
        out = self.model_part2(penultimate)
        return penultimate, out

    @classmethod
    def get_model(cls, stage, input_size, penultimate_size):
        if stage == 0:
            dim_in = input_size
        else:
            dim_in = input_size + penultimate_size
        model = GON(dim_in, penultimate_size, num_layers=3)
        return model


class NN(nn.Module):  # [x,y]->[RGB]
    def __init__(self, dim_in, hidden_size):
        super(NN, self).__init__()
        self.model_part1 = nn.Sequential(
            nn.Linear(dim_in, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.model_part2 = nn.Sequential(
            nn.Linear(hidden_size, 3), nn.Sigmoid()
        )

    def forward(self, x, prev_penultimate=None):
        if prev_penultimate is not None:
            x = torch.cat([x, prev_penultimate], dim=1)
        penultimate = self.model_part1(x)
        out = self.model_part2(penultimate)
        return penultimate, out

    @classmethod
    def get_model(cls, stage, input_size, penultimate_size):
        if stage == 0:
            dim_in = input_size
        else:
            dim_in = input_size + penultimate_size
        model = NN(dim_in, penultimate_size)
        return model


# noinspection PyPep8Naming
def get_psnr(pred, target):
    return 10 * torch.log10((torch.max(pred) ** 2) / (torch.sum(pred - target) ** 2 / target.numel()))


def train_weak(stage, P, B, net_ensemble, batches, image, pred_img_weak):
    model = GON.get_model(stage, P["hidden_size"] * 2, P["hidden_size"]).to(device) if P["model"] == 'gon' \
        else NN.get_model(stage, P["hidden_size"] * 2, P["hidden_size"]).to(device)

    optimizer = get_optimizer(model.parameters(), P["lr_model"], P["optimizer"])
    # scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.1, verbose=False)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=1, epochs=200, anneal_strategy='linear')
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 0.95 ** e)
    # scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.00001)
    criterion = nn.MSELoss()

    stage_model_loss = 0
    loss_tmp = 0
    pbar = tqdm(range(P["epochs_per_stage"]), desc=f"Stage {stage + 1}/{P['num_nets']}: weak model training\t", unit=" epoch")
    for epoch in pbar:
        for pos_batch in batches:
            h_idx = pos_batch[:, 0].long()
            w_idx = pos_batch[:, 1].long()
            x = input_mapping(pos_batch.to(device), B)
            y = image[h_idx, w_idx]

            middle_feat, out = net_ensemble.forward(x)
            grad_direction = -(out - y)
            # grad_direction = y / (1.0 + torch.exp(y * out))

            _, out = model(x, middle_feat)
            loss = criterion(net_ensemble.boost_rate * out, grad_direction)
            pred_img_weak[h_idx, w_idx] = out.detach()

            # make_dot(loss, params=dict(model.named_parameters()), show_attrs=True, show_saved=True).render(f"_model_{stage + 1}", format="svg")
            model.zero_grad()
            loss.backward()
            optimizer.step()
            loss_tmp = loss.item()
            stage_model_loss += loss_tmp * len(y)

            # cv2.imwrite(OUT_ROOT + f'/_weak_stage{stage + 1}.png', pred_img_weak.cpu().numpy() * 255)
        # scheduler_plateau.step(loss)

        with torch.no_grad():
            pbar.set_postfix({
                'lr': optimizer.param_groups[0]['lr'],
                'loss_weak': loss_tmp,
                'grad0': grad_direction[0].detach().cpu().numpy()
            })
            wandb.log({
                f"loss/model/{stage + 1}": loss_tmp,
                f"lr/model/{stage + 1}": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1
            })

    with torch.no_grad():
        model_mse = np.sqrt(stage_model_loss / (P["n_batches"] * P["epochs_per_stage"]))
        wandb.log({
            "Summary/Model_final_loss": model_mse,
            "Summary/Model_final_lr": optimizer.param_groups[0]['lr'],
            # "Summary/Model_psnr": get_psnr(pred_img_weak, image),
            "stage": stage + 1
        })
    return model


def fully_corrective_step(stage, P, B, net_ensemble, batches, image, pred_img_ensemble, lr_ensemble):
    h, w, channels = P["input_size"]
    lr_scaler = 1
    optimizer = get_optimizer(net_ensemble.parameters(), lr_ensemble / lr_scaler, P["optimizer"])
    scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.1, verbose=False)
    criterion = nn.MSELoss()

    if stage % 15 == 0:
        # lr_scaler *= 2
        lr_ensemble /= 2
        optimizer.param_groups[0]["lr"] = lr_ensemble

    stage_loss = 0
    pbar = tqdm(range(P["epochs_per_correction"]), desc=f"Stage {stage + 1}/{P['num_nets']}: fully corrective step", unit=" epoch")
    for epoch in pbar:
        ensemble_loss = 0
        for pos_batch in batches:
            h_idx = pos_batch[:, 0].long()
            w_idx = pos_batch[:, 1].long()
            x = input_mapping(pos_batch.to(device), B)
            y = image[h_idx, w_idx]

            _, out = net_ensemble.forward(x)  # TODO: check forward_grad()
            pred_img_ensemble[h_idx, w_idx] = out.detach()
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ensemble_loss = loss.item()
            stage_loss += ensemble_loss * len(y)
        # scheduler_plateau.step(loss)

        with torch.no_grad():
            pbar.set_postfix({
                'lr': optimizer.param_groups[0]['lr'],
                'loss_ensemble': ensemble_loss,
                # 'boost': net_ensemble.boost_rate.data.item()
            })
            wandb.log({
                f"loss/ensemble/{stage + 1}": ensemble_loss,
                f"lr/ensemble/{stage + 1}": optimizer.param_groups[0]["lr"],
                "epoch": epoch + 1
            })

    with torch.no_grad():
        ensemble_mse = np.sqrt(stage_loss / (len(batches) * P["epochs_per_correction"]))
        ensemble_psnr = get_psnr(pred_img_ensemble, image)
        ensemble_ssim = ssim(pred_img_ensemble.view(1, channels, h, w), image.view(1, channels, h, w))  # WARNING: takes a lot of memory
        wandb.log({
            "Summary/Ensemble_lr": optimizer.param_groups[0]['lr'],
            "Summary/Ensemble_psnr": ensemble_psnr,
            "Summary/Ensemble_ssim": ensemble_ssim,
            "Summary/Ensemble_loss": ensemble_mse,
            "Summary/boost_rate": net_ensemble.boost_rate.data,
            "stage": stage + 1
        })
        print(f'Ensemble_MSE: {ensemble_mse: .5f}, Ensemble_PSNR: {ensemble_psnr: .5f}, Ensemble_SSIM: {ensemble_ssim: .5f}, Boost rate: {net_ensemble.boost_rate:.4f}\n')
    return lr_ensemble


def train(net_ensemble, P, image):
    h, w, channels = P["input_size"]
    batches = batch_generator_in_memory(P, shuffle=False)

    pred_img_weak = torch.Tensor(np.empty(shape=(h, w, channels))).to(device)
    pred_img_ensemble = torch.Tensor(np.empty(shape=(h, w, channels))).to(device)

    criterion = nn.MSELoss()
    B = P["B_scale"] * torch.randn((P["hidden_size"], 2)).to(device)
    lr_ensemble = P["lr_ensemble"]

    # train new model and append to ensemble
    for stage in range(P["num_nets"]):
        net_ensemble.to_train()
        weak_model = train_weak(stage, P, B, net_ensemble, batches, image, pred_img_weak)
        net_ensemble.add(weak_model)

        if stage > 0 and P["epochs_per_correction"] != 0:
            lr_ensemble = fully_corrective_step(stage, P, B, net_ensemble, batches, image, pred_img_ensemble, lr_ensemble)

        max_w = torch.max(pred_img_weak).cpu().numpy()
        min_w = torch.min(pred_img_weak).cpu().numpy()
        # cv2.imwrite(ROOT + f'/_ensembleP1_stage{stage + 1}.png', (pred_img_ensemble.cpu().numpy() * 255))
        cv2.imwrite(OUT_ROOT + f'/_ensembleP2_stage{stage + 1}.png', (pred_img_ensemble.cpu().numpy() * 255))
        cv2.imwrite(OUT_ROOT + f'/_weak_stage{stage + 1}.png', pred_img_weak.cpu().numpy() * 255)
        cv2.imwrite(OUT_ROOT + f'/_weakN_stage{stage + 1}.png', ((pred_img_weak.cpu().numpy() - min_w) / max_w * 255))


def main():
    image = cv2.imread("../data/IMG_0201_DxO.jpg")
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, channels = image.shape
    scale = 0.5
    h *= scale
    w *= scale
    image = cv2.resize(image, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(OUT_ROOT + '/sample.png', image)
    image = torch.Tensor(image).div(255).to(device)  # values:[0,1]

    hparams = {
        "epochs_per_stage": 5,
        "epochs_per_correction": 1,
        "boost_rate": 1.0,
        "lr_model": 0.001,
        "lr_ensemble": 0.001,
        "batch_size": 20000,
        "B_scale": 0.03,
        "hidden_size": 32,
        "model": 'gon',
        "optimizer": 'adamW',
        "batch_sampling_mode": BatchSamplingMode.nth_element.name,
        "num_nets": 50,
        "acc_gradients": False,
        "input_size": image.shape
    }
    if hparams["batch_sampling_mode"] == BatchSamplingMode.whole:
        hparams["batch_size"] = h * w
        hparams["n_batches"] = 1
    else:
        hparams["n_batches"] = np.prod(image.shape[:-1]) // (hparams["batch_size"]) + 1  # FIXME: wrong, remove -1 and change code accordingly
    # batch_size = h * w // n_batches + 1

    wandb.init(project="image_compression_grownet", entity="a-di", dir="../runs", config=hparams, tags=["debug"], mode="disabled")  # mode="disabled", id="2"
    print(wandb.config)

    # get_checkpoint = False
    # save_checkpoints = True

    c0 = torch.Tensor([0.0, 0.0, 0.0]).to(device)  # torch.mean(image, dim=[0, 1])
    net_ensemble = DynamicNet(c0, hparams["boost_rate"])
    # wandb.watch(net_ensemble, criterion, log="all")  # log="all", log_freq=10, log_graph=True
    # summary(model, input_size=(P["batch_size"], P["hidden_size"] * 2))

    train(net_ensemble, hparams, image)


if __name__ == '__main__':
    start = time.time()
    main()
    print("seconds: ", time.time() - start)

# TODO:
# check support for gon model
# recycle optimizer?
# check if loss is correct
# add tunable n_layers
# try without input mapping
# compare grownet vs DNN vs my imageRegression without GrowNet
# compare imageRegression with fourier-feature-networks repo
# Finally make W&B report
# separate image regression project from this
