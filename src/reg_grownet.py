from datetime import timedelta

from config.default import OUT_ROOT, device, hparams_grownet, init_P, INPUT_PATH
from torch import nn
from torchmetrics.functional import ssim
from tqdm import tqdm
from utils import input_mapping, get_optimizer, batch_generator_in_memory, get_psnr, image_preprocess, get_model
from utils.grownet import DynamicNet
import cv2
import numpy as np
import time
import torch
import wandb


class MetricsManager:
    def __init__(self, wandb_run):
        self.wandb = wandb_run
        self.ensemble_boost_rate = None
        self.ensemble_mse = None
        self.ensemble_ssim = None
        self.ensemble_psnr = None
        self.ensemble_lr = None
        self.ensemble_stage = None

    def log_weak_epoch(self, P, epoch, stage, loss_epoch, optimizer, image, pred_image, pbar=None):
        h, w, channels = P["image_shape"]
        lr = optimizer.param_groups[0]['lr']
        _psnr = get_psnr(pred_image, image).item()
        _ssim = ssim(pred_image.view(1, channels, h, w), image.view(1, channels, h, w)).item()

        if pbar:
            pbar.set_postfix({
                'lr': lr,
                'loss': loss_epoch,
                'psnr': _psnr,
                'ssim': _ssim
                # 'grad0': grad_direction[0].detach().cpu().numpy()
            })

        self.wandb.log({
            "epoch": epoch,
            "stage": stage,
            "Weak/lr": lr,
            "Weak/loss": loss_epoch,
            "Weak/psnr": _psnr,
            "Weak/ssim": _ssim
        }, commit=True)

    def log_ensemble_epoch(self, P, epoch, stage, boost_rate, loss_epoch, optimizer, image, pred_image, pbar=None):
        h, w, channels = P["image_shape"]
        self.ensemble_mse = loss_epoch  # np.sqrt(loss_stage / (P["n_batches"] * P["epochs_per_correction"]))
        self.ensemble_lr = optimizer.param_groups[0]["lr"]
        self.ensemble_psnr = get_psnr(pred_image, image).item()
        self.ensemble_ssim = ssim(pred_image.view(1, channels, h, w), image.view(1, channels, h, w)).item()  # WARNING: takes a lot of memory
        self.ensemble_boost_rate = boost_rate.item()

        if pbar:
            pbar.set_postfix({
                'lr': self.ensemble_lr,
                'loss': self.ensemble_mse,
                'psnr': self.ensemble_psnr,
                'ssim': self.ensemble_ssim,
                'boost_rate': self.ensemble_boost_rate
            })

        self.wandb.log({
            "epoch": epoch,
            "stage": self.ensemble_stage,
            "Ensemble/lr": self.ensemble_lr,
            "Ensemble/psnr": self.ensemble_psnr,
            "Ensemble/ssim": self.ensemble_ssim,
            "Ensemble/loss": self.ensemble_mse,
            "Ensemble/boost_rate": self.ensemble_boost_rate
        }, commit=True)

    def log_ensemble_summary(self):
        self.wandb.log({
            "Final/stage": self.ensemble_stage,
            "Final/lr": self.ensemble_lr,
            "Final/psnr": self.ensemble_psnr,
            "Final/ssim": self.ensemble_ssim,
            "Final/loss": self.ensemble_mse,
            "Final/boost_rate": self.ensemble_boost_rate,
        }, commit=True)
        print("FINISH:")
        print(f'\tEnsemble_MSE: {self.ensemble_mse: .5f}, Ensemble_PSNR: {self.ensemble_psnr: .5f}, Ensemble_SSIM: {self.ensemble_ssim: .5f}, Boost rate: {self.ensemble_boost_rate:.4f}\n')


def train_weak(stage, P, B, net_ensemble, batches, image, pred_img_weak, criterion, metrics_manager):
    model = get_model(P, stage).to(device)
    optimizer = get_optimizer(model.parameters(), P["lr_model"], P["optimizer"])
    # scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.1, verbose=False)

    h, w, channels = P["image_shape"]
    img_grad_weak = torch.Tensor(np.empty(shape=(h, w, channels))).to(device)

    pbar = tqdm(range(P["epochs_per_stage"]), desc=f"[{str(timedelta(seconds=time.time() - start))[:-5]}] Stage {stage + 1}/{P['num_nets']}: weak model training", unit=" epoch")
    for epoch in pbar:
        loss_batch_cum = 0
        for pos_batch in batches:
            h_idx = pos_batch[:, 0].long()
            w_idx = pos_batch[:, 1].long()
            x = input_mapping(pos_batch.to(device), B)
            y = image[h_idx, w_idx]

            middle_feat, out = net_ensemble.forward(x)
            grad_direction = -(out - y)  # grad_direction = y / (1.0 + torch.exp(y * out))
            img_grad_weak[h_idx, w_idx] = grad_direction.detach()

            _, out = model(x, middle_feat)
            pred_img_weak[h_idx, w_idx] = out.detach()
            # make_dot(loss, params=dict(model.named_parameters()), show_attrs=True, show_saved=True).render(f"_model_{stage + 1}", format="svg")

            loss = criterion(net_ensemble.boost_rate * out, grad_direction)
            loss.backward()
            loss_batch_cum += loss.item()

            if not P["acc_gradients"]:
                optimizer.step()
                model.zero_grad()
        if P["acc_gradients"]:
            optimizer.step()
            model.zero_grad()
        # scheduler_plateau.step(loss)

        loss_epoch = loss_batch_cum / len(batches)
        metrics_manager.log_weak_epoch(P, epoch + 1, stage + 1, loss_epoch, optimizer, img_grad_weak, pred_img_weak, pbar)
    return model


def fully_corrective_step(stage, P, B, net_ensemble, batches, image, pred_img_ensemble, criterion, lr_ensemble, metrics_manager):
    lr_scaler = 1
    optimizer = get_optimizer(net_ensemble.parameters(), lr_ensemble / lr_scaler, P["optimizer"])
    # scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.1, verbose=False)

    # if stage % 15 == 0:
    #     # lr_scaler *= 2
    #     lr_ensemble /= 2
    #     optimizer.param_groups[0]["lr"] = lr_ensemble

    pbar = tqdm(range(P["epochs_per_correction"]), desc=f"[{str(timedelta(seconds=time.time() - start))[:-5]}] Stage {stage + 1}/{P['num_nets']}: fully corrective step", unit=" epoch")
    for epoch in pbar:
        loss_batch_cum = 0
        for pos_batch in batches:
            h_idx = pos_batch[:, 0].long()
            w_idx = pos_batch[:, 1].long()
            x = input_mapping(pos_batch.to(device), B)
            y = image[h_idx, w_idx]

            _, out = net_ensemble.forward(x)  # TODO: check forward_grad()
            pred_img_ensemble[h_idx, w_idx] = out.detach()

            loss = criterion(out, y)
            loss.backward()
            loss_batch_cum += loss.item()

            if not P["acc_gradients"]:
                optimizer.zero_grad()
                optimizer.step()
        if P["acc_gradients"]:
            optimizer.zero_grad()
            optimizer.step()
        # scheduler_plateau.step(loss)

        loss_epoch = loss_batch_cum / len(batches)
        metrics_manager.log_ensemble_epoch(P, epoch + 1, stage + 1, net_ensemble.boost_rate, loss_epoch, optimizer, image, pred_img_ensemble, pbar)
    return lr_ensemble


def train(net_ensemble, P, image, batches, criterion):
    h, w, channels = P["image_shape"]
    wandb_run = wandb.init(project="image_regression_final", entity=WANDB_USER, dir="../out", config=P, tags=["grownet"], mode=WANDB_MODE)  # mode="disabled", id="2"
    print(wandb_run.config)
    # wandb.watch(net_ensemble, criterion, log="all")  # log="all", log_freq=10, log_graph=True

    metrics_manager = MetricsManager(wandb_run)

    pred_img_weak = torch.Tensor(np.empty(shape=(h, w, channels))).to(device)
    pred_img_ensemble = torch.Tensor(np.empty(shape=(h, w, channels))).to(device)

    B = P["B_scale"] * torch.randn((P["input_layer_size"] // 2, 2)).to(device)
    lr_ensemble = P["lr_ensemble"]

    # train new model and append to ensemble
    for stage in range(P["num_nets"]):
        net_ensemble.to_train()
        weak_model = train_weak(stage, P, B, net_ensemble, batches, image, pred_img_weak, criterion, metrics_manager)
        net_ensemble.add(weak_model)

        if stage > 0 and P["epochs_per_correction"] != 0:
            lr_ensemble = fully_corrective_step(stage, P, B, net_ensemble, batches, image, pred_img_ensemble, criterion, lr_ensemble, metrics_manager)

        max_w = torch.max(pred_img_weak).cpu().numpy()
        min_w = torch.min(pred_img_weak).cpu().numpy()
        # cv2.imwrite(ROOT + f'/grownet/_ensembleP1_stage{stage + 1}.png', (pred_img_ensemble.cpu().numpy() * 255))
        cv2.imwrite(OUT_ROOT + f'/grownet/_ensembleP2_stage{stage + 1}.png', (pred_img_ensemble.cpu().numpy() * 255))
        cv2.imwrite(OUT_ROOT + f'/grownet/_weak_stage{stage + 1}.png', pred_img_weak.cpu().numpy() * 255)
        cv2.imwrite(OUT_ROOT + f'/grownet/_weakN_stage{stage + 1}.png', ((pred_img_weak.cpu().numpy() - min_w) / max_w * 255))
    metrics_manager.log_ensemble_summary()


def main():
    P = hparams_grownet

    image = cv2.imread(INPUT_PATH)
    image = image_preprocess(image, P).to(device)
    cv2.imwrite(OUT_ROOT + '/grownet/sample.png', image.cpu().numpy() * 255)

    init_P(P, image)

    c0 = torch.Tensor([0.0, 0.0, 0.0]).to(device)  # torch.mean(image, dim=[0, 1])
    net_ensemble = DynamicNet(c0, P["boost_rate"])
    criterion = nn.MSELoss()

    batches = batch_generator_in_memory(P, device, shuffle=P["shuffle_batches"])
    train(net_ensemble, P, image, batches, criterion)


if __name__ == '__main__':
    start = time.time()
    # os.environ["WANDB_DIR"] = "../runs"
    WANDB_USER = "a-di"
    WANDB_MODE = "online"  # ["online", "offline", "disabled"]
    main()
    print("seconds: ", time.time() - start)
