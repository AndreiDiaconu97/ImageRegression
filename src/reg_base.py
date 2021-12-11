from adabound import adabound
from torchmetrics.functional import ssim

from config.default import OUT_ROOT, device, INPUT_PATH, hparams_base, init_P
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchmetrics import SSIM
from torchvision import transforms
from tqdm import tqdm
from utils import input_mapping, get_psnr, batch_generator_in_memory, image_preprocess, get_model, get_optimizer, get_paramas_num, save_checkpoint, load_checkpoint
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


# os.environ["WANDB_DIR"] = "../runs"
# writer = SummaryWriter(ROOT + '/test')


# def load_checkpoint(model, optimizer, schedulers):
#     print("Load checkpoint")
#     checkpoint = torch.load(OUT_ROOT + '/model.pth')
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     schedulers.load_state_dict(checkpoint['schedulers'])
#     B = checkpoint['B']
#     saved_epoch = checkpoint['epoch']
#     return B, saved_epoch

class MetricsManager:
    def __init__(self):
        self.n_params = None
        self.epoch = None
        self.lr = None
        self.loss = None
        self.psnr = None
        self.ssim = None

    def log_metrics(self, P, epoch, model, loss_epoch, optimizer, image, pred_image, pbar=None):
        h, w, channels = P["image_shape"]
        self.n_params = get_paramas_num(model)
        self.epoch = epoch
        self.lr = optimizer.param_groups[0]['lr']
        self.loss = loss_epoch
        self.psnr = get_psnr(pred_image, image).item()
        self.ssim = ssim(pred_image.view(1, channels, h, w), image.view(1, channels, h, w)).item()

        if pbar:
            pbar.set_postfix({
                "loss": loss_epoch,
                "lr": optimizer.param_groups[0]['lr'],
                "psnr": self.psnr,
                "ssim": self.ssim,
                "n_params": self.n_params
            })

        wandb.log({
            "epoch": self.epoch,
            "lr": self.lr,
            "loss": self.loss,
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
            "image": wandb.Image(OUT_ROOT + f'/base/sample_{self.epoch}.png'),
            "image_error": wandb.Image(OUT_ROOT + f'/base/error_{self.epoch}.png')
        })


def train(model, P, B, image, optimizer, criterion, loss, start_epoch, batches, schedulers):
    model.train()
    h, w, channels = P["image_shape"]
    metrics_manager = MetricsManager()

    pred_img = torch.Tensor(np.empty(shape=(h, w, channels))).to(device)

    last_t, last_check_t = 0, 0
    pbar = tqdm(range(start_epoch, P["epochs"]), unit=" epoch", desc=f"Training")
    for epoch in pbar:
        loss_batch_cum = 0
        for i, pos_batch in enumerate(batches):
            h_idx = pos_batch[:, 0].long()
            w_idx = pos_batch[:, 1].long()
            x = input_mapping(pos_batch, B)

            y_pred = model(x)
            loss = criterion(image[h_idx, w_idx], y_pred)
            loss.backward()
            loss_batch_cum += loss.item()

            if not P["acc_gradients"]:
                optimizer.step()
                optimizer.zero_grad()
        if P["acc_gradients"]:
            optimizer.step()
            optimizer.zero_grad()
        if schedulers["plateau"]:
            schedulers["plateau"].step(loss)
        loss_epoch = loss_batch_cum / len(batches)

        for i, pos_batch in enumerate(batches):  # save prediction
            h_idx = pos_batch[:, 0].long()
            w_idx = pos_batch[:, 1].long()
            x = input_mapping(pos_batch, B)

            y_pred = model(x)
            pred_img[h_idx, w_idx] = y_pred.detach()

        if (epoch + 1) % 1 == 0 and (time.time() - last_t > 1):
            last_t = time.time()
            # print(f'epoch {epoch + 1}/{P["epochs"]}, loss={actual_loss:.8f}, psnr={model_psnr:.2f}, ssim={model_ssim:.2f} lr={optimizer.param_groups[0]["lr"]:.7f}')
            metrics_manager.log_metrics(P, epoch + 1, model, loss_epoch, optimizer, image, pred_img, pbar)
            img_error = image - pred_img
            img_error = (img_error - img_error.min()) * 1 / (img_error.max() - img_error.min())
            cv2.imwrite(OUT_ROOT + f'/base/error_{epoch + 1}.png', (img_error.cpu().numpy() * 255))  # .reshape((h, w, channels)))
            cv2.imwrite(OUT_ROOT + f'/base/sample_{epoch + 1}.png', (pred_img.cpu().numpy() * 255))  # .reshape((h, w, channels)))

        if save_checkpoints and (time.time() - last_check_t > 10):
            last_check_t = time.time()
            save_checkpoint(PATH_CHECKPOINT, model, optimizer, loss, epoch, B)
            wandb.log({"checkpoint(KB)": os.path.getsize(PATH_CHECKPOINT) / 1000})
            # wandb.save(PATH_CHECKPOINT)
            print("Checkpoint saved")

    torch.onnx.export(model, input_mapping(batches[0], B), PATH_ONNX, input_names=["2D"], output_names=["RGB"], dynamic_axes={"2D": {0: "batch_size"}})
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


def main():
    P = hparams_base

    image = cv2.imread(INPUT_PATH)
    image = image_preprocess(image, P).to(device)
    cv2.imwrite(OUT_ROOT + '/base/sample.png', image.cpu().numpy() * 255)

    init_P(P, image)

    model = get_model(P).to(device)
    optimizer = get_optimizer(model.parameters(), P["lr"], P["optimizer"])
    criterion = nn.MSELoss()

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=1, epochs=200, anneal_strategy='linear')
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 0.95 ** e)
    # scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.00001)
    schedulers = {
        "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.1, verbose=False)
    }

    start_epoch = 0
    loss = None
    B = P["B_scale"] * torch.randn((P["input_layer_size"] // 2, 2)).to(device)
    if get_checkpoint and os.path.isfile(PATH_CHECKPOINT):
        start_epoch, loss, B = load_checkpoint(PATH_CHECKPOINT, model, optimizer)
        print("Checkpoint laded")

    batches = batch_generator_in_memory(P, device, shuffle=P["shuffle_batches"])
    with wandb.init(project="image_regression_final", dir=OUT_ROOT, config=P, **WANDB_CFG):
        print(wandb.config)
        # summary(model, input_size=(P["batch_size"], P["hidden_size"] * 2))
        wandb.watch(model, criterion, log="all")  # log="all", log_freq=10, log_graph=True
        train(model, P, B, image, optimizer, criterion, loss, start_epoch, batches, schedulers)


if __name__ == '__main__':
    start = time.time()
    get_checkpoint = False
    save_checkpoints = True
    FOLDER = "base"
    PATH_CHECKPOINT = os.path.join(OUT_ROOT, FOLDER, "checkpoint.pth")
    PATH_ONNX = os.path.join(OUT_ROOT, FOLDER, "model_base.onnx")
    PATH_B = os.path.join(OUT_ROOT, FOLDER, "B")

    WANDB_CFG = {
        "entity": "a-di",
        "mode": "online",  # ["online", "offline", "disabled"]
        "tags": ["base_nn"],
        "group": None,  # "exp_1",
        "job_type": None,
        "id": None
    }
    main()
    print("seconds: ", time.time() - start)

# TODO: try printing same model with different B
