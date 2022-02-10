import argparse
import random
import sys

from torch.optim.lr_scheduler import MultiStepLR

sys.path.append('C:/Users/USER/Documents/Programming/ImageRegression')

from adabound import adabound
import config.default
from config.default import OUT_ROOT, device, INPUT_PATH, hparams_base, init_P, device_batches, device_pred_img, device_image
from torch import nn
from torchinfo import summary
from torchmetrics.functional import ssim
from tqdm import tqdm
from utils import input_mapping, get_psnr, batch_generator, image_preprocess, get_model, get_optimizer, get_params_num, save_checkpoint, load_checkpoint, get_scaled_image
import cv2
import numpy as np
import os
import time
import torch
import wandb


@torch.no_grad()
def validate_model(P, batches, image, pred_img, B, model, criterion):
    h, w, channels = P["image_shape"]
    loss_batch_cum = 0
    for pos_batch in batches:  # save prediction
        h_idx = pos_batch[:, 0]
        w_idx = pos_batch[:, 1]
        # batch = torch.stack((h_idx / h * 2 - 1, w_idx / w * 2 - 1), dim=1).to(device)  # [-1,1] better for Siren
        batch = torch.stack((h_idx / h, w_idx / w), dim=1).to(device)  # [0,1] better for Fourier
        x = input_mapping(batch, B) if P["B_scale"] else batch
        y = image[h_idx, w_idx]

        y_pred = model(x)
        loss_batch_cum += criterion(y.to(device), y_pred).item()
        pred_img[h_idx, w_idx] = y_pred.to(device_pred_img)
    loss_epoch = loss_batch_cum / len(batches)
    return loss_epoch


def save_image_out(image, pred_img, epoch):
    img_error = image - pred_img
    img_error = (img_error - img_error.min()) * 1 / (img_error.max() - img_error.min())
    cv2.imwrite(f'{OUT_ROOT}/{FOLDER}/error_{epoch}.png', (img_error.cpu().numpy() * 255))
    cv2.imwrite(f'{OUT_ROOT}/{FOLDER}/sample_{epoch}.png', (pred_img.cpu().numpy() * 255))


class MetricsManager:
    def __init__(self):
        self.n_params = None
        self.epoch = None
        self.lr = None
        self.loss = None
        self.psnr = None
        self.ssim = None

    def log_loss(self, epoch, loss_epoch):
        self.loss = loss_epoch
        self.epoch = epoch

        wandb.log({
            "epoch": self.epoch,
            "loss": self.loss,
        }, commit=True)

    def log_metrics(self, P, model, optimizer, image, pred_image, epoch=None, pbar=None):
        h, w, channels = P["image_shape"]
        self.n_params = get_params_num(model)
        if epoch:
            self.epoch = epoch
        self.lr = optimizer.param_groups[0]['lr']
        self.psnr = get_psnr(pred_image, image).item()
        self.ssim = ssim(pred_image.view(1, channels, h, w), image.view(1, channels, h, w)).item()

        if pbar:
            pbar.set_postfix({
                "lr": optimizer.param_groups[0]['lr'],
                "psnr": self.psnr,
                "ssim": self.ssim,
                "n_params": self.n_params
            })

        wandb.log({
            "epoch": self.epoch,
            "lr": self.lr,
            "psnr": self.psnr,
            "ssim": self.ssim,
            "n_params": self.n_params
        }, commit=True)

    def log_final_metrics(self):
        wandb.log({
            "Final/epoch": self.epoch,
            "Final/lr": self.lr,
            "Final/loss": self.loss,
            "Final/psnr": self.psnr,
            "Final/ssim": self.ssim,
            "Final/n_params": self.n_params,
            "Final/model(KB)": os.path.getsize(PATH_CHECKPOINT) / 1000,
            "Final/onnx(KB)": os.path.getsize(PATH_ONNX) / 1000,
            "image": wandb.Image(f'{OUT_ROOT}/{FOLDER}/sample_{self.epoch}.png'),
            "image_error": wandb.Image(f'{OUT_ROOT}/{FOLDER}/error_{self.epoch}.png')
        })


def train(model, P, B, image, optimizer, criterion, loss, start_epoch, batches, scheduler=None):
    model.train()
    h, w, channels = P["image_shape"]
    metrics_manager = MetricsManager()
    # scheduler_step = MultiStepLR(optimizer, milestones=[1000], gamma=0.1)

    pred_img = torch.Tensor(np.empty(shape=(h, w, channels))).to(device_pred_img)

    last_t, last_check_t = 0, 0
    pbar = tqdm(range(start_epoch, P["epochs"]), unit=" epoch", desc=f"Training")
    for epoch in pbar:
        loss_batch_cum = 0
        for i, pos_batch in enumerate(batches):
            # mask_1 = get_hole_mask(pos_batch, h_from=40, w_from=530, size=80)
            # mask_2 = get_hole_mask(pos_batch, h_from=180, w_from=530, size=10)
            # mask_3 = get_hole_mask(pos_batch, h_from=530, w_from=530, size=20)
            # mask = mask_1 & mask_2 & mask_3
            #
            # h_idx = pos_batch[mask, 0]
            # w_idx = pos_batch[mask, 1]

            h_idx = pos_batch[:, 0]
            w_idx = pos_batch[:, 1]
            # batch = torch.stack((h_idx / h * 2 - 1, w_idx / w * 2 - 1), dim=1).to(device)  # [-1,1] better for Siren
            batch = torch.stack((h_idx / h, w_idx / w), dim=1).to(device)  # [0,1] better for Fourier
            x = input_mapping(batch, B) if P["B_scale"] else batch
            y = image[h_idx, w_idx]

            y_pred = model(x)
            loss = criterion(y.to(device), y_pred)
            loss_batch_cum += loss.item()
            loss.backward()

            if not P["acc_gradients"]:
                optimizer.step()
                optimizer.zero_grad()
            # scheduler_step.step()
            if scheduler:
                scheduler.step(loss)
        if P["acc_gradients"]:
            optimizer.step()
            optimizer.zero_grad()

        # optimizer.param_groups[0]['lr'] *= 0.90
        loss_epoch = loss_batch_cum / len(batches)
        metrics_manager.log_loss(epoch + 1, loss_epoch)

        with torch.no_grad():
            if (epoch + 1) % 1 == 0 and (time.time() - last_t > 1):
                loss_epoch = validate_model(P, batches, image, pred_img, B, model, criterion)
                metrics_manager.log_metrics(P, model, optimizer, image, pred_img, epoch + 1, pbar)
                save_image_out(image, pred_img, epoch + 1)
                last_t = time.time()

            if save_checkpoints and (time.time() - last_check_t > 10):
                last_check_t = time.time()
                save_checkpoint(PATH_CHECKPOINT, model, optimizer.param_groups[0]['lr'], loss, epoch, B)
                wandb.log({"checkpoint(KB)": os.path.getsize(PATH_CHECKPOINT) / 1000})
                # wandb.save(PATH_CHECKPOINT)
                print("Checkpoint saved")

            if MAX_MINUTES:
                if time.time() - start > MAX_MINUTES * 60:
                    print("Time is up! Closing experiment...")
                    break

    loss_epoch = validate_model(P, batches, image, pred_img, B, model, criterion)
    metrics_manager.log_metrics(P, model, optimizer, image, pred_img)

    save_image_out(image, pred_img, P["epochs"])
    torch.onnx.export(model, input_mapping(batches[0].to(device), B).float(), PATH_ONNX, input_names=["2D"], output_names=["RGB"], dynamic_axes={"2D": {0: "batch_size"}})
    if B is not None:
        np.save(PATH_B, B.cpu().numpy())
    metrics_manager.log_final_metrics()
    # wandb.save(PATH_ONNX)
    # wandb.save(PATH_B + ".npy")

    cv2.imwrite(OUT_ROOT + "/" + "_".join([
        "base",
        "EP", str(P["epochs"]),
        "M", P["model"],
        "lr", str(P["lr"]),
        "batchS", str(P["batch_size"]),
        "Bscale", str(P["B_scale"]),
        "OP", P["optimizer"]
    ]) + '.png', (pred_img.cpu().numpy() * 255))


def main(P):
    image = cv2.imread(INPUT_PATH)
    image = image_preprocess(image, P).to(device_image)
    cv2.imwrite(f'{OUT_ROOT}/{FOLDER}/sample.png', image.cpu().numpy() * 255)

    init_P(P, image)

    model = get_model(P).to(device)
    optimizer = get_optimizer(model.parameters(), P["lr"], P["optimizer"])
    criterion = nn.MSELoss()

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=1, epochs=200, anneal_strategy='linear')
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 0.95 ** e)
    # scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.00001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=P["lr_patience"], factor=.5, verbose=True, cooldown=1000)

    start_epoch = 0
    loss = None
    if P["B_scale"]:
        B = P["B_scale"] * torch.randn((P["input_layer_size"] // 2, 2)).to(device)
    else:
        B = None
    if get_checkpoint and os.path.isfile(PATH_CHECKPOINT):
        start_epoch, loss, B, _, lr = load_checkpoint(PATH_CHECKPOINT, model)
        optimizer.param_groups[0]['lr'] = lr
        print("Checkpoint loaded")

    batches = batch_generator(P, device=device_batches, shuffle=P["shuffle_batches"])
    with wandb.init(project="image_regression_final", dir=OUT_ROOT, config=P, **WANDB_CFG):
        print(wandb.config)
        summary(model, input_size=(P["batch_size"], P["input_layer_size"]))
        wandb.watch(model, criterion, log="all")  # log="all", log_freq=10, log_graph=True
        train(model, P, B, image, optimizer, criterion, loss, start_epoch, batches, scheduler)

    # pred_img = get_scaled_image(P, B, model, scale=5.0)
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
    parser.add_argument("--normalized_coordinates", type=bool,
                        help="map pixel coordinates to [0,1], leave True.")
    parser.add_argument("--acc_gradients", type=bool,
                        help="accumulate gradient for all batches before gradient descent")
    parser.add_argument("--batch_sampling_mode", type=str,
                        help="options: [sequence, nth_element, whole]")
    parser.add_argument("--shuffle_batches", type=bool,
                        help="better leaving True")
    parser.add_argument("--batch_size", type=int,
                        help="best around 5000-20000")
    parser.add_argument("--epochs", type=int,
                        help="number of training epochs")
    parser.add_argument("--hidden_size", type=int,
                        help="weak model width")
    parser.add_argument("--hidden_layers", type=int,
                        help="weak model depth")
    parser.add_argument("--lr", type=float,
                        help="training learning rate")
    parser.add_argument("--model", type=str,
                        help="options: [relu, siren]")
    parser.add_argument("--optimizer", type=str,
                        help="options: [adam, adamW, RMSprop, SGD]")
    parser.add_argument("--scale", type=float,
                        help="online image resizing factor")

    configargs = parser.parse_args()

    P = hparams_base
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

    FOLDER = os.path.join("base", configargs.run_name) if configargs.run_name else os.path.join("base", "unnamed")
    PATH_CHECKPOINT = os.path.join(OUT_ROOT, FOLDER, "checkpoint.pth")
    PATH_ONNX = os.path.join(OUT_ROOT, FOLDER, "_model_base.onnx")
    PATH_B = os.path.join(OUT_ROOT, FOLDER, "B")

    if not os.path.exists(os.path.join(OUT_ROOT, FOLDER)):
        os.makedirs(os.path.join(OUT_ROOT, FOLDER))

    WANDB_CFG = {
        "entity": "a-di",
        "mode": "online",  # ["online", "offline", "disabled"]
        "tags": ["base"],
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
