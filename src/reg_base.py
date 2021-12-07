import random

from adabound import adabound
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchmetrics import SSIM
from torch import nn
from torchvision import transforms
from tqdm import tqdm
import cv2
import enum
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import time
import torch
import torch.nn.functional as F
import torchvision
import wandb
from utils import input_mapping
from config.default import OUT_ROOT, device


# Ensure deterministic behavior
# torch.backends.cudnn.deterministic = True
# random.seed(hash("setting random seeds") % 2**32 - 1)
# np.random.seed(hash("improves reproducibility") % 2**32 - 1)
# torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
# torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# os.environ["WANDB_DIR"] = "../runs"


# writer = SummaryWriter(ROOT + '/test')
from utils.models import gon_model, NN


class BatchSamplingMode(enum.Enum):
    sequential = 1
    nth_element = 2
    whole = 3


def slice_per(source, step):
    return [source[i::step] for i in range(step)]


def get_batches():
    positions = []
    for y in range(h):
        for x in range(w):
            positions.append((y, x))
    positions = torch.Tensor(positions).to(device)

    if P["batch_sampling_mode"] == BatchSamplingMode.nth_element.name:
        return slice_per(positions, n_slices)
    elif P["batch_sampling_mode"] == BatchSamplingMode.whole.name:
        return [positions]
    elif P["batch_sampling_mode"] == BatchSamplingMode.sequential.name:
        return torch.split(positions, P["batch_size"])
    else:
        raise ValueError('BatchSamplingMode: unknown value')


def load_checkpoint(model, optimizer, scheduler):
    print("Load checkpoint")
    checkpoint = torch.load(OUT_ROOT + '/model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    B = checkpoint['B']
    saved_epoch = checkpoint['epoch']
    return B, saved_epoch


# noinspection PyPep8Naming
def get_psnr(pred, target):
    return 10 * torch.log10((torch.max(pred) ** 2) / (torch.sum(pred - target) ** 2 / target.numel()))


def train():
    ssim = SSIM()
    loss = torch.tensor(0)
    pred_img = torch.Tensor(np.empty(shape=(h, w, channels))).to(device)
    pbar = tqdm(range(saved_epoch, P["epochs"]), unit=" epoch", desc=f"Training")
    for epoch in pbar:
        for i, pos_batch in enumerate(batches):
            h_idx = pos_batch[:, 0].long()
            w_idx = pos_batch[:, 1].long()

            y_pred = model(input_mapping(pos_batch, B))
            pred_img[h_idx, w_idx] = y_pred.detach()
            loss = criterion(image[h_idx, w_idx], y_pred)

            if P["acc_gradients"]:
                loss /= len(batches)
            loss.backward()

            if not P["acc_gradients"]:
                optimizer.step()
                optimizer.zero_grad()
        if P["acc_gradients"]:
            optimizer.step()
            optimizer.zero_grad()
        # scheduler.step()
        scheduler_plateau.step(loss)

        model_psnr = get_psnr(pred_img, image)
        model_ssim = ssim(pred_img.view(1, channels, h, w), image.view(1, channels, h, w))

        if (epoch + 1) % 1 == 0:
            actual_loss = loss.item() * len(batches) if P["acc_gradients"] else loss.item()
            # print(f'epoch {epoch + 1}/{P["epochs"]}, loss={actual_loss:.8f}, psnr={model_psnr:.2f}, ssim={model_ssim:.2f} lr={optimizer.param_groups[0]["lr"]:.7f}')

            pbar.set_postfix({
                "loss": actual_loss,
                "lr": optimizer.param_groups[0]['lr'],
                "psnr": model_psnr.item(),
                "ssim": model_ssim.item()
            })

            wandb.log({
                "loss": actual_loss,
                "lr": optimizer.param_groups[0]['lr'],
                "psnr": model_psnr,
                "ssim": model_ssim
            }, epoch)
            # writer.add_scalar('Loss/train', actual_loss, epoch)
            # writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
            cv2.imwrite(OUT_ROOT + f'/sample_{epoch + 1}.png', (pred_img.cpu().numpy() * 255))  # .reshape((h, w, channels)))

        if (epoch + 1) == P["epochs"]:
            cv2.imwrite(OUT_ROOT + f'/f_{epoch + 1}_' + "_".join([
                "M", P["model"],
                "lr", str(P["lr"]),
                "Bsize", str(P["batch_size"]),
                "Bscale", str(P["B_scale"]),
                "OP", P["optimizer"]
            ]) + '.png', (pred_img.cpu().numpy() * 255))  # .reshape((h, w, channels)))

        # save checkpoint
        if save_checkpoints and (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler_plateau.state_dict(),
                'B': B
            }, OUT_ROOT + '/model.pth')
            # torch.onnx.export(model, input_mapping(batches[0], B), os.path.join(wandb.run.dir, "model.onnx"))
            # wandb.save(os.path.join(wandb.run.dir, "model.onnx"))
            print('Saved checkpoint')


if __name__ == '__main__':
    start = time.time()
    get_checkpoint = False
    save_checkpoints = True

    # TODO: add n_layers as tunable parameter
    hyperparameter_defaults = {
        "lr": 0.000728,  # [0.01, 0.0001],
        "epochs": 5000,
        "batch_size": 2272,
        # "n_slices": n_slices,
        "B_scale": 0.0676,
        "hidden_size": 256,
        "model": 'nn',
        "optimizer": 'adamW',
        "batch_sampling_mode": BatchSamplingMode.nth_element.name,
        "acc_gradients": False
    }
    wandb.init(project="image_compression", entity="a-di", dir="../runs", config=hyperparameter_defaults, tags=["debug"], mode="disabled")  # mode="disabled", id="2"
    P = wandb.config

    B = P["B_scale"] * torch.randn((P["hidden_size"], 2)).to(device)

    image = cv2.imread("../data/IMG_0201_DxO.jpg")
    h, w, channels = image.shape
    h, w = h // 10, w // 10
    n_slices = h * w // (P["batch_size"] - 1)
    # batch_size = h * w // n_slices + 1

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(OUT_ROOT + '/sample.png', image)
    image = torch.Tensor(image).div(255).to(device)

    model = gon_model(4, P['hidden_size'] * 2, P['hidden_size']).to(device) if P["model"] == 'gon' else NN(P["hidden_size"]).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), P["lr"]) if P["optimizer"] == 'adam' \
        else torch.optim.AdamW(model.parameters(), P["lr"]) if P["optimizer"] == 'adamW' \
        else torch.optim.SGD(model.parameters(), P["lr"])
    # optimizer = torch.optim.Adagrad(model.parameters(), lr)
    # optimizer = adabound.AdaBound(model.parameters(), lr, final_lr=1)

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=1, epochs=200, anneal_strategy='linear')
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 0.95 ** e)
    # scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.00001)
    scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.1, verbose=False)

    summary(model, input_size=(P["batch_size"], P["hidden_size"] * 2))
    wandb.config.update({
        "n_slices": n_slices,
        "input_size": image.shape,
    })
    print(wandb.config)
    wandb.watch(model, criterion, log="all")  # log="all", log_freq=10, log_graph=True

    saved_epoch = 0
    if get_checkpoint and os.path.exists(OUT_ROOT + '/model.pth'):
        B, saved_epoch = load_checkpoint(model, optimizer, scheduler_plateau)
    # optimizer.param_groups[-1]['lr'] = 0.00001

    # wandb.agent(sweep_id, train, count=10)

    batches = get_batches()
    random.shuffle(batches)
    model.train()
    train()

    print("seconds: ", time.time() - start)
