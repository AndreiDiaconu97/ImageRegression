import argparse
import random
import sys

# sys.path.append('C:/Users/USER/Documents/Programming/ImageRegression')
import glob
import os
from datetime import timedelta
from torchviz import make_dot
from torch import nn
from torchmetrics.functional import ssim
from tqdm import tqdm
import config.default
from config.default import OUT_ROOT, device, hparams_grownet, init_P, INPUT_PATH, device_batches, device_image, device_pred_img
from utils import input_mapping, get_optimizer, batch_generator, get_psnr, image_preprocess, get_model, get_params_num, save_checkpoint, load_checkpoint, get_scaled_image, str_to_bool
from utils.grownet import DynamicNet
import cv2
import numpy as np
import time
import torch
import wandb


def save_ensemble_final(net_ensemble, batches, P, B):
    h, w, channels = P["image_shape"]
    h_idx = batches[0][:, 0]
    w_idx = batches[0][:, 1]
    batch = torch.stack((h_idx / h, w_idx / w), dim=1).to(device)
    x = input_mapping(batch, B) if P["B_scale"] else batch
    for i, model in enumerate(net_ensemble.models):
        f_path = f"{PATH_ONNX.removesuffix('net_ensemble.onnx')}weak_{i}.onnx"
        middle_feat, out = net_ensemble.forward(x)
        if not P["propagate_context"]:
            middle_feat = None
        input_shape = x if i == 0 else (x, middle_feat)
        torch.onnx.export(model, input_shape, f_path, input_names=["2D"], output_names=["RGB"], dynamic_axes={"2D": {0: "batch_size"}})
    if B is not None:
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
                'loss': loss_epoch,
                'lr': lr,
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
    def log_weak_stage(P, stage, loss_epoch, image, pred_image):
        h, w, channels = P["image_shape"]
        _psnr = get_psnr(pred_image, image).item()
        _ssim = ssim(pred_image.view(1, channels, h, w), image.view(1, channels, h, w)).item()

        wandb.log({
            "stage": stage,
            "Weak/loss": loss_epoch,
            "Weak/psnr": _psnr,
            "Weak/ssim": _ssim,
        }, commit=True)

        print(f'Weak_MSE: {loss_epoch:.5f}, Weak_PSNR: {_psnr:.5f}, Weak_SSIM: {_ssim:.5f}')

    def log_ensemble_epoch(self, epoch, stage, net_ensemble, loss_epoch, optimizer, pbar=None):
        self.ensemble_stage = stage
        self.n_params = get_params_num(net_ensemble)
        self.ensemble_boost_rate = net_ensemble.boost_rate.item()
        self.ensemble_lr = optimizer.param_groups[0]["lr"]
        self.ensemble_mse = loss_epoch  # np.sqrt(loss_stage / (P["n_batches"] * P["epochs_per_correction"]))

        if pbar:
            pbar.set_postfix({
                'loss': self.ensemble_mse,
                'lr': self.ensemble_lr,
                'n_params': self.n_params,
                'boost_rate': self.ensemble_boost_rate,
            })

        wandb.log({
            "epoch": epoch,
            "stage": self.ensemble_stage,
            "Ensemble/n_params": self.n_params,
            "Ensemble/boost_rate": self.ensemble_boost_rate,
            "Ensemble/lr": self.ensemble_lr,
            "Ensemble/loss": self.ensemble_mse,
        }, commit=True)

    def log_ensemble_stage(self, P, stage, loss_epoch, image, pred_image, net_ensemble):
        h, w, channels = P["image_shape"]
        self.ensemble_stage = stage
        self.ensemble_mse = loss_epoch  # np.sqrt(loss_stage / (P["n_batches"] * P["epochs_per_correction"]))
        self.ensemble_psnr = get_psnr(pred_image, image).item()
        self.ensemble_ssim = ssim(pred_image.view(1, channels, h, w), image.view(1, channels, h, w)).item()  # WARNING: takes a lot of memory
        self.n_params = get_params_num(net_ensemble)
        self.ensemble_boost_rate = net_ensemble.boost_rate.item()

        wandb.log({
            "stage": self.ensemble_stage,
            "Ensemble/n_params": self.n_params,
            "Ensemble/boost_rate": self.ensemble_boost_rate,
            "Ensemble/loss": self.ensemble_mse,
            "Ensemble/psnr": self.ensemble_psnr,
            "Ensemble/ssim": self.ensemble_ssim,
        }, commit=True)

        print(f'Ensemble_MSE: {self.ensemble_mse:.5f}, Ensemble_PSNR: {self.ensemble_psnr:.5f}, Ensemble_SSIM: {self.ensemble_ssim:.5f}, Boost rate: {self.ensemble_boost_rate:.4f}\n')

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
    h, w, channels = P["image_shape"]
    model = get_model(P, stage if P["propagate_context"] else 0).to(device)
    optimizer = get_optimizer(model.parameters(), P["lr_model"], P["optimizer"])
    scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=P["lr_patience_model"], factor=.5, verbose=False, cooldown=P["lr_cooldown_weak"])

    pbar = tqdm(range(P["epochs_per_stage"]), desc=f"[{str(timedelta(seconds=time.time() - start))[:-5]}] Stage {stage + 1}/{P['num_nets']}: weak model training", unit=" epoch")
    for epoch in pbar:
        loss_batch_cum = 0
        for pos_batch in batches:
            h_idx = pos_batch[:, 0]
            w_idx = pos_batch[:, 1]
            batch = torch.stack((h_idx / h, w_idx / w), dim=1).to(device)
            x = input_mapping(batch, B) if P["B_scale"] else batch
            y = image[h_idx, w_idx]

            middle_feat, out_ensemble = net_ensemble.forward(x)
            grad_direction = -(out_ensemble - y.to(device))  # grad_direction = y / (1.0 + torch.exp(y * out))

            if not P["propagate_context"]:
                middle_feat = None
            _, out = model(x, middle_feat)

            loss = criterion(net_ensemble.boost_rate * out, grad_direction)
            loss.backward()
            loss_batch_cum += loss.item()

            if not P["acc_gradients"]:
                optimizer.step()
                optimizer.zero_grad()
            scheduler_plateau.step(loss)
        if P["acc_gradients"]:
            optimizer.step()
            optimizer.zero_grad()
        # P["lr_model"] = optimizer.param_groups[0]['lr']

        loss_epoch = loss_batch_cum / len(batches)
        metrics_manager.log_weak_epoch(P, epoch + 1, stage + 1, model, loss_epoch, optimizer, pbar)

    # LAST LOGGING PASS + PSNR + SSIM #
    with torch.no_grad():
        loss_batch_cum = 0
        for i, pos_batch in enumerate(batches):
            h_idx = pos_batch[:, 0]
            w_idx = pos_batch[:, 1]
            batch = torch.stack((h_idx / h, w_idx / w), dim=1).to(device)
            x = input_mapping(batch, B) if P["B_scale"] else batch
            y = image[h_idx, w_idx]

            middle_feat, out_ensemble = net_ensemble.forward(x)
            grad_direction = -(out_ensemble.to(device) - y.to(device))
            img_grad_weak[h_idx, w_idx] = grad_direction.to(device_pred_img)

            if not P["propagate_context"]:
                middle_feat = None
            _, out = model(x, middle_feat)
            pred_img_weak[h_idx, w_idx] = out.to(device_pred_img)

            loss = criterion(net_ensemble.boost_rate * out, grad_direction)
            loss_batch_cum += loss.item()

            # make_dot(loss, params=dict(model.named_parameters()), show_attrs=True, show_saved=True).render(f"_model_{stage + 1}", format="svg")
        loss_epoch = loss_batch_cum / len(batches)
        metrics_manager.log_weak_stage(P, stage + 1, loss_epoch, img_grad_weak, pred_img_weak)

    return model


def fully_corrective_step(stage, P, B, net_ensemble, batches, image, pred_img_ensemble, criterion, optimizer, scheduler_plateau, metrics_manager):
    h, w, channels = P["image_shape"]

    optimizer.param_groups[0]["params"] = net_ensemble.parameters()  # reinsert updated ensemble parameters
    # optimizer_boost_rate = get_optimizer([net_ensemble.boost_rate], 1e-2, P["optimizer"])

    # if stage % 1 == 0:
    #     # lr_scaler *= 2
    #     lr_ensemble *= 0.90
    #     optimizer.param_groups[0]["lr"] = lr_ensemble

    if P["epochs_per_correction"] > 0:
        pbar = tqdm(range(P["epochs_per_correction"]), desc=f"[{str(timedelta(seconds=time.time() - start))[:-5]}] Stage {stage + 1}/{P['num_nets']}: fully corrective step", unit=" epoch")
        for epoch in pbar:
            loss_batch_cum = 0
            for pos_batch in batches:
                h_idx = pos_batch[:, 0]
                w_idx = pos_batch[:, 1]
                batch = torch.stack((h_idx / h, w_idx / w), dim=1).to(device)
                x = input_mapping(batch, B) if P["B_scale"] else batch
                y = image[h_idx, w_idx]

                _, out = net_ensemble.forward_grad(x)
                loss = criterion(out, y.to(device))
                loss.backward()
                loss_batch_cum += loss.item()

                if not P["acc_gradients"]:
                    optimizer.step()
                    optimizer.zero_grad()
                    # optimizer_boost_rate.step()
                    # optimizer_boost_rate.zero_grad()
                scheduler_plateau.step(loss)
            if P["acc_gradients"]:
                optimizer.step()
                optimizer.zero_grad()

            loss_epoch = loss_batch_cum / len(batches)
            metrics_manager.log_ensemble_epoch(epoch + 1, stage + 1, net_ensemble, loss_epoch, optimizer, pbar)

    # LAST LOGGING PASS + PSNR + SSIM #
    with torch.no_grad():
        loss_batch_cum = 0
        for pos_batch in batches:
            h_idx = pos_batch[:, 0]
            w_idx = pos_batch[:, 1]
            batch = torch.stack((h_idx / h, w_idx / w), dim=1).to(device)
            x = input_mapping(batch, B) if P["B_scale"] else batch
            y = image[h_idx, w_idx]

            _, out = net_ensemble.forward(x)
            pred_img_ensemble[h_idx, w_idx] = out.to(device_pred_img)

            loss = criterion(out, y.to(device))
            loss_batch_cum += loss.item()
        loss_last_epoch = loss_batch_cum / len(batches)
        metrics_manager.log_ensemble_stage(P, stage + 1, loss_last_epoch, image, pred_img_ensemble, net_ensemble)


def train(net_ensemble, P, B, image, criterion, start_stage, batches):
    net_ensemble.to_train()
    h, w, channels = P["image_shape"]
    metrics_manager = MetricsManager()

    img_grad_weak = torch.Tensor(np.empty(shape=(h, w, channels))).to(device_pred_img)
    pred_img_weak = torch.Tensor(np.empty(shape=(h, w, channels))).to(device_pred_img)
    pred_img_ensemble = torch.Tensor(np.empty(shape=(h, w, channels))).to(device_pred_img)

    lr_scaler = 1
    ens_optimizer = get_optimizer([
        {'params': net_ensemble.parameters()},
        # {'params': [net_ensemble.boost_rate], 'lr': 2}
    ], P["lr_ensemble"] / lr_scaler, P["optimizer"])
    ens_scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(ens_optimizer, patience=P["lr_patience_ensemble"], factor=.5, verbose=False, cooldown=P["lr_cooldown_ensemble"])

    # train new model and append to ensemble
    for stage in range(start_stage, P["num_nets"]):
        weak_model = train_weak(stage, P, B, net_ensemble, batches, image, pred_img_weak, img_grad_weak, criterion, metrics_manager)
        net_ensemble.add(weak_model)

        if stage > 0:
            fully_corrective_step(stage, P, B, net_ensemble, batches, image, pred_img_ensemble, criterion, ens_optimizer, ens_scheduler_plateau, metrics_manager)
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

        if MAX_MINUTES:
            if time.time() - start > MAX_MINUTES * 60:
                print("Time is up! Closing experiment...")
                break

    save_ensemble_final(net_ensemble, batches, P, B)
    metrics_manager.log_ensemble_summary()


def main(P):
    image = cv2.imread(INPUT_PATH)
    image = image_preprocess(image, P).to(device_image)
    cv2.imwrite(f'{OUT_ROOT}/{FOLDER}/sample.png', image.cpu().numpy() * 255)

    init_P(P, image)

    with wandb.init(project="image_regression_final", **WANDB_CFG, dir=OUT_ROOT, config=P):
        # wandb.config.update({"input_layer_size": wandb.config["hidden_size"] * 2}, allow_val_change=True)
        P.update(wandb.config)

        if P["model"] == 'siren':
            c0 = torch.mean(image, dim=[0, 1]).to(device)
        else:
            c0 = torch.Tensor([0.0, 0.0, 0.0]).to(device)
        net_ensemble = DynamicNet(c0, P["boost_rate"], P["propagate_context"], P["enable_boost_rate"])
        criterion = nn.MSELoss()

        start_stage = 0
        if P["B_scale"]:
            B = P["B_scale"] * torch.randn((P["input_layer_size"] // 2, 2)).to(device)
        else:
            B = None
        if get_checkpoint and os.path.isfile(PATH_CHECKPOINT):
            start_stage, _, B, P, lr = load_checkpoint(PATH_CHECKPOINT, net_ensemble, is_grownet=True)
            net_ensemble.to(device)
            print("Checkpoint loaded")

        batches = batch_generator(P, device_batches, shuffle=P["shuffle_batches"])
        print(P)
        # print(wandb.config)
        # wandb.watch(net_ensemble, criterion, log="all")  # log="all", log_freq=10, log_graph=True
        train(net_ensemble, P, B, image, criterion, start_stage, batches)

    # pred_img = get_scaled_image(P, B, net_ensemble, scale=5.0)
    # cv2.imwrite(f'{OUT_ROOT}/{FOLDER}/sample_resized.png', (pred_img.cpu().numpy() * 255))


if __name__ == '__main__':
    start = time.time()
    get_checkpoint = False
    save_checkpoints = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-name", type=str, required=False,
                        help="name of the configuration (see in default.py)")
    parser.add_argument("--run_name", type=str, default="",
                        help="Name of the run (for wandb), leave empty for random name.")
    parser.add_argument("--input_img", type=str, default="",
                        help="path for the input image")
    parser.add_argument("--max_mins", type=int,
                        help="maximum duration of the training")

    # Config overrides #
    parser.add_argument("--B_scale", type=int,
                        help="Random Fourier features scaler, set to 0 to disable fourier features")
    parser.add_argument("--w0", type=int,
                        help="Siren scaler factor, ignored if using ReLU")
    parser.add_argument("--acc_gradients", type=str_to_bool,
                        help="accumulate gradient for all batches before gradient descent")
    parser.add_argument("--batch_sampling_mode", type=str,
                        help="options: [sequence, nth_element, whole]")
    parser.add_argument("--shuffle_batches", type=str_to_bool,
                        help="better leaving True")
    parser.add_argument("--batch_size", type=int,
                        help="best around 5000-20000")
    parser.add_argument("--boost_rate", type=float,
                        help="GrowNet boosting rate")
    parser.add_argument("--epochs_per_correction", type=int,
                        help="epochs of fully corrective steps for each weak learner")
    parser.add_argument("--epochs_per_stage", type=int,
                        help="epoch of training for each weak learner")
    parser.add_argument("--hidden_size", type=int,
                        help="weak model width")
    parser.add_argument("--hidden_layers", type=int,
                        help="weak model depth")
    parser.add_argument("--lr_ensemble", type=float,
                        help="learning rate during fully-corrective phase")
    parser.add_argument("--lr_model", type=float,
                        help="learning rate during weak model training")
    parser.add_argument("--lr_patience_model", type=int,
                        help="Number of batches to wait for the weak learner before dropping the learning rate.")
    parser.add_argument("--lr_patience_ensemble", type=int,
                        help="Number of batches to wait for the corrective phase before dropping the learning rate.")
    parser.add_argument("--lr_cooldown_weak", type=int,
                        help="Cooldown before re-enabling plateau scheduler for the weak learner")
    parser.add_argument("--lr_cooldown_ensemble", type=int,
                        help="Cooldown before re-enabling plateau scheduler for the corrective phase")
    parser.add_argument("--model", type=str,
                        help="options: [relu, siren]")
    parser.add_argument("--num_nets", type=int,
                        help="final number of weak learners")
    parser.add_argument("--optimizer", type=str,
                        help="options: [adam, adamW, RMSprop, SGD]")
    parser.add_argument("--scale", type=float,
                        help="online image resizing factor")
    parser.add_argument("--propagate_context", type=str_to_bool,
                        help="Choose if appending the penultimate layer to the next learner.")
    parser.add_argument("--enable_boost_rate", type=str_to_bool,
                        help="Choose if learning the boost rate, otherwise is left to 1.")

    configargs = parser.parse_args()

    P = hparams_grownet
    if configargs.cfg_name:
        try:
            P = getattr(config.default, configargs.cfg_name)
        except:
            print("ERROR: Wrong configuration name argument")
    tmp_args = {k: v for k, v in configargs.__dict__.items() if k not in ["cfg_name", "run_name", "input_img", "max_mins"]}
    for k, v in tmp_args.items():
        if v is not None:
            P[k] = v

    if configargs.input_img:
        INPUT_PATH = configargs.input_img
    MAX_MINUTES = configargs.max_mins if configargs.max_mins else None

    FOLDER = os.path.join("grownet", configargs.run_name) if configargs.run_name else os.path.join("grownet", "unnamed")
    PATH_CHECKPOINT = os.path.join(OUT_ROOT, FOLDER, "checkpoint.pth")
    PATH_ONNX = os.path.join(OUT_ROOT, FOLDER, "net_ensemble.onnx")
    PATH_B = os.path.join(OUT_ROOT, FOLDER, "B")

    if not os.path.exists(os.path.join(OUT_ROOT, FOLDER)):
        os.makedirs(os.path.join(OUT_ROOT, FOLDER))
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
    if configargs.run_name:
        WANDB_CFG["name"] = configargs.run_name
        WANDB_CFG["tags"].append("thesis")

    # Seed experiment for repeatability
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    main(P)
    print("seconds: ", time.time() - start)
