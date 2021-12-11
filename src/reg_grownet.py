import glob
import os
from datetime import timedelta
from config.default import OUT_ROOT, device, hparams_grownet, init_P, INPUT_PATH
from torch import nn
from torchmetrics.functional import ssim
from tqdm import tqdm
from utils import input_mapping, get_optimizer, batch_generator_in_memory, get_psnr, image_preprocess, get_model, get_params_num, save_checkpoint, load_checkpoint
from utils.grownet import DynamicNet
import cv2
import numpy as np
import time
import torch
import wandb


def save_ensemble_final(net_ensemble, batches, B):
    for i, model in enumerate(net_ensemble.models):
        f_path = f"{PATH_ONNX.removesuffix('net_ensemble.onnx')}weak_{i}.onnx"
        x = input_mapping(batches[0], B)
        middle_feat, out = net_ensemble.forward(x)
        input_shape = x if i == 0 else (x, middle_feat)
        torch.onnx.export(model, input_shape, f_path, input_names=["2D"], output_names=["RGB"], dynamic_axes={"2D": {0: "batch_size"}})
    np.save(PATH_B, B.cpu().numpy())


class MetricsManager:
    def __init__(self):
        self.ensemble_mse = None
        self.ensemble_ssim = None
        self.ensemble_psnr = None
        self.ensemble_lr = None
        self.ensemble_stage = None
        self.ensemble_boost_rate = None
        self.n_params = None

    @staticmethod
    def log_weak_epoch(P, epoch, stage, model, loss_epoch, optimizer, pbar=None):
        lr = optimizer.param_groups[0]['lr']
        _n_params = get_params_num(model)

        if pbar:
            pbar.set_postfix({
                'lr': lr,
                'loss': loss_epoch,
                'n_params': _n_params
                # 'grad0': grad_direction[0].detach().cpu().numpy()
            })

        wandb.log({
            "epoch": epoch,
            "stage": stage,
            "Weak/lr": lr,
            "Weak/loss": loss_epoch,
            "Weak/n_params": _n_params
        }, commit=True)

    @staticmethod
    def log_weak_stage(P, stage, image, pred_image):
        h, w, channels = P["image_shape"]
        _psnr = get_psnr(pred_image, image).item()
        _ssim = ssim(pred_image.view(1, channels, h, w), image.view(1, channels, h, w)).item()

        wandb.log({
            "stage": stage,
            "Weak/psnr": _psnr,
            "Weak/ssim": _ssim,
        }, commit=True)

        print(f'Weak_PSNR: {_psnr: .5f}, Weak_SSIM: {_ssim: .5f}')

    def log_ensemble_epoch(self, epoch, stage, net_ensemble, loss_epoch, optimizer, pbar=None):
        self.ensemble_stage = stage
        self.n_params = get_params_num(net_ensemble)
        self.ensemble_boost_rate = net_ensemble.boost_rate.item()
        self.ensemble_lr = optimizer.param_groups[0]["lr"]
        self.ensemble_mse = loss_epoch  # np.sqrt(loss_stage / (P["n_batches"] * P["epochs_per_correction"]))

        if pbar:
            pbar.set_postfix({
                'n_params': self.n_params,
                'boost_rate': self.ensemble_boost_rate,
                'lr': self.ensemble_lr,
                'loss': self.ensemble_mse,
            })

        wandb.log({
            "epoch": epoch,
            "stage": self.ensemble_stage,
            "Ensemble/n_params": self.n_params,
            "Ensemble/boost_rate": self.ensemble_boost_rate,
            "Ensemble/lr": self.ensemble_lr,
            "Ensemble/loss": self.ensemble_mse,
        }, commit=True)

    def log_ensemble_stage(self, P, stage, net_ensemble, loss_epoch, optimizer, image, pred_image):
        h, w, channels = P["image_shape"]
        self.ensemble_stage = stage
        self.n_params = get_params_num(net_ensemble)
        self.ensemble_boost_rate = net_ensemble.boost_rate.item()
        self.ensemble_lr = optimizer.param_groups[0]["lr"]
        self.ensemble_mse = loss_epoch  # np.sqrt(loss_stage / (P["n_batches"] * P["epochs_per_correction"]))
        self.ensemble_psnr = get_psnr(pred_image, image).item()
        self.ensemble_ssim = ssim(pred_image.view(1, channels, h, w), image.view(1, channels, h, w)).item()  # WARNING: takes a lot of memory

        wandb.log({
            "stage": self.ensemble_stage,
            "Ensemble/n_params": self.n_params,
            "Ensemble/boost_rate": self.ensemble_boost_rate,
            "Ensemble/lr": self.ensemble_lr,
            "Ensemble/loss": self.ensemble_mse,
            "Ensemble/psnr": self.ensemble_psnr,
            "Ensemble/ssim": self.ensemble_ssim,
        }, commit=True)

        print(f'Ensemble_MSE: {self.ensemble_mse: .5f}, Ensemble_PSNR: {self.ensemble_psnr: .5f}, Ensemble_SSIM: {self.ensemble_ssim: .5f}, Boost rate: {self.ensemble_boost_rate:.4f}\n')

    def log_ensemble_summary(self):
        onnx_weak_s = 0
        onnx_weak_files = glob.glob(f'{os.path.join(OUT_ROOT, FOLDER)}/*weak*.onnx')
        for f in onnx_weak_files:
            onnx_weak_s += os.path.getsize(f)

        wandb.log({
            "Final/stage": self.ensemble_stage,
            "Final/lr": self.ensemble_lr,
            "Final/psnr": self.ensemble_psnr,
            "Final/ssim": self.ensemble_ssim,
            "Final/loss": self.ensemble_mse,
            "Final/boost_rate": self.ensemble_boost_rate,
            "Final/n_params": self.n_params,
            "Final/model(KB)": os.path.getsize(PATH_CHECKPOINT) / 1000,
            "Final/onnx(KB)": onnx_weak_s / 1000,
            "image": wandb.Image(f'{OUT_ROOT}/{FOLDER}/pred/_ensemble_{self.ensemble_stage}.png'),
            "image_weak": wandb.Image(f'{OUT_ROOT}/{FOLDER}/pred_weak/_weak_{self.ensemble_stage}.png'),
            "image_error": wandb.Image(f'{OUT_ROOT}/{FOLDER}/error/_grad_{self.ensemble_stage}.png')
        }, commit=True)


def train_weak(stage, P, B, net_ensemble, batches, image, pred_img_weak, img_grad_weak, criterion, metrics_manager):
    model = get_model(P, stage).to(device)
    optimizer = get_optimizer(model.parameters(), P["lr_model"], P["optimizer"])
    scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.1, verbose=False)

    pbar = tqdm(range(P["epochs_per_stage"]), desc=f"[{str(timedelta(seconds=time.time() - start))[:-5]}] Stage {stage + 1}/{P['num_nets']}: weak model training", unit=" epoch")
    middle_features = []
    for epoch in pbar:
        loss_batch_cum = 0
        middle_features.clear()
        for pos_batch in batches:
            h_idx = pos_batch[:, 0].long()
            w_idx = pos_batch[:, 1].long()
            x = input_mapping(pos_batch, B)  # pos_batch.to(device)
            y = image[h_idx, w_idx]

            middle_feat, out = net_ensemble.forward(x)
            middle_features.append(middle_feat)
            grad_direction = -(out - y)  # grad_direction = y / (1.0 + torch.exp(y * out))
            img_grad_weak[h_idx, w_idx] = grad_direction.detach()

            _, out = model(x, middle_feat)

            loss = criterion(net_ensemble.boost_rate * out, grad_direction)
            loss.backward()
            loss_batch_cum += loss.item()

            if not P["acc_gradients"]:
                optimizer.step()
                model.zero_grad()
        if P["acc_gradients"]:
            optimizer.step()
            model.zero_grad()
        scheduler_plateau.step(loss)

        loss_epoch = loss_batch_cum / len(batches)
        metrics_manager.log_weak_epoch(P, epoch + 1, stage + 1, model, loss_epoch, optimizer, pbar)

    # DEBUG + PSNR + SSIM #
    for i, pos_batch in enumerate(batches):
        h_idx = pos_batch[:, 0].long()
        w_idx = pos_batch[:, 1].long()
        x = input_mapping(pos_batch, B)
        _, out = model(x, middle_features[i])
        pred_img_weak[h_idx, w_idx] = out.detach()
        # make_dot(loss, params=dict(model.named_parameters()), show_attrs=True, show_saved=True).render(f"_model_{stage + 1}", format="svg")
    metrics_manager.log_weak_stage(P, stage + 1, img_grad_weak, pred_img_weak)

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
            x = input_mapping(pos_batch, B)
            y = image[h_idx, w_idx]

            _, out = net_ensemble.forward(x)  # TODO: check forward_grad()
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
        metrics_manager.log_ensemble_epoch(epoch + 1, stage + 1, net_ensemble, loss_epoch, optimizer, pbar)

    # DEBUG + PSNR + SSIM #
    loss_batch_cum = 0
    for i, pos_batch in enumerate(batches):
        h_idx = pos_batch[:, 0].long()
        w_idx = pos_batch[:, 1].long()
        x = input_mapping(pos_batch, B)
        y = image[h_idx, w_idx]

        _, out = net_ensemble.forward(x)
        pred_img_ensemble[h_idx, w_idx] = out.detach()

        loss = criterion(out, y)
        loss.backward()
        loss_batch_cum += loss.item()
    loss_last_epoch = loss_batch_cum / len(batches)
    metrics_manager.log_ensemble_stage(P, stage + 1, net_ensemble, loss_last_epoch, optimizer, image, pred_img_ensemble)

    return lr_ensemble


def train(net_ensemble, P, B, image, criterion, start_stage, batches):
    net_ensemble.to_train()
    h, w, channels = P["image_shape"]
    metrics_manager = MetricsManager()

    img_grad_weak = torch.Tensor(np.empty(shape=(h, w, channels))).to(device)
    pred_img_weak = torch.Tensor(np.empty(shape=(h, w, channels))).to(device)
    pred_img_ensemble = torch.Tensor(np.empty(shape=(h, w, channels))).to(device)

    # train new model and append to ensemble
    lr_ensemble = P["lr_ensemble"]
    for stage in range(start_stage, P["num_nets"]):
        weak_model = train_weak(stage, P, B, net_ensemble, batches, image, pred_img_weak, img_grad_weak, criterion, metrics_manager)
        net_ensemble.add(weak_model)

        if stage > 0:
            lr_ensemble = fully_corrective_step(stage, P, B, net_ensemble, batches, image, pred_img_ensemble, criterion, lr_ensemble, metrics_manager)
        if save_checkpoints:
            save_checkpoint(PATH_CHECKPOINT, net_ensemble, None, None, stage + 1, B, P)
            if os.path.isfile(PATH_CHECKPOINT):
                wandb.log({"checkpoint(KB)": os.path.getsize(PATH_CHECKPOINT) / 1000})
            # wandb.save(PATH_CHECKPOINT)

        img_grad_weak = (img_grad_weak - img_grad_weak.min()) * 1 / (img_grad_weak.max() - img_grad_weak.min())
        pred_img_weak = (pred_img_weak - pred_img_weak.min()) * 1 / (pred_img_weak.max() - pred_img_weak.min())
        cv2.imwrite(f'{OUT_ROOT}/{FOLDER}/error/_grad_{stage + 1}.png', (img_grad_weak.cpu().numpy() * 255))
        cv2.imwrite(f'{OUT_ROOT}/{FOLDER}/pred_weak/_weak_{stage + 1}.png', (pred_img_weak.cpu().numpy() * 255))
        cv2.imwrite(f'{OUT_ROOT}/{FOLDER}/pred/_ensemble_{stage + 1}.png', (pred_img_ensemble.cpu().numpy() * 255))
    save_ensemble_final(net_ensemble, batches, B)
    metrics_manager.log_ensemble_summary()


def main():
    P = hparams_grownet

    image = cv2.imread(INPUT_PATH)
    image = image_preprocess(image, P).to(device)
    cv2.imwrite(f'{OUT_ROOT}/{FOLDER}/sample.png', image.cpu().numpy() * 255)

    init_P(P, image)

    c0 = torch.mean(image, dim=[0, 1])  # torch.Tensor([0.0, 0.0, 0.0]).to(device)
    net_ensemble = DynamicNet(c0, P["boost_rate"])
    criterion = nn.MSELoss()

    start_stage = 0
    B = P["B_scale"] * torch.randn((P["input_layer_size"] // 2, 2)).to(device)
    if get_checkpoint and os.path.isfile(PATH_CHECKPOINT):
        start_stage, _, B, P = load_checkpoint(PATH_CHECKPOINT, net_ensemble, None, is_grownet=True)
        net_ensemble.to(device)
        print("Checkpoint loaded")

    batches = batch_generator_in_memory(P, device, shuffle=P["shuffle_batches"])
    with wandb.init(project="image_regression_final", **WANDB_CFG, dir=OUT_ROOT, config=P):
        print(wandb.config)
        # wandb.watch(net_ensemble, criterion, log="all")  # log="all", log_freq=10, log_graph=True
        train(net_ensemble, P, B, image, criterion, start_stage, batches)


if __name__ == '__main__':
    start = time.time()
    get_checkpoint = False
    save_checkpoints = True
    FOLDER = "grownet"
    PATH_CHECKPOINT = os.path.join(OUT_ROOT, FOLDER, "checkpoint.pth")
    PATH_ONNX = os.path.join(OUT_ROOT, FOLDER, "net_ensemble.onnx")
    PATH_B = os.path.join(OUT_ROOT, FOLDER, "B")
    if not os.path.isdir(os.path.join(OUT_ROOT, FOLDER, "error")):
        os.mkdir(os.path.join(OUT_ROOT, FOLDER, "error"))
    if not os.path.isdir(os.path.join(OUT_ROOT, FOLDER, "pred")):
        os.mkdir(os.path.join(OUT_ROOT, FOLDER, "pred"))
    if not os.path.isdir(os.path.join(OUT_ROOT, FOLDER, "pred_weak")):
        os.mkdir(os.path.join(OUT_ROOT, FOLDER, "pred_weak"))

    WANDB_CFG = {
        "entity": "a-di",
        "mode": "online",  # ["online", "offline", "disabled"]
        "tags": ["grownet"],
        "group": None,  # "exp_1",
        "job_type": None,
        "id": None
    }

    main()
    print("seconds: ", time.time() - start)
