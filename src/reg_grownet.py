from config.default import OUT_ROOT, device, hparams_grownet, init_P, INPUT_PATH
from torch import nn
from torchmetrics.functional import ssim
from tqdm import tqdm
from utils import input_mapping, BatchSamplingMode, get_optimizer, batch_generator_in_memory, get_psnr, image_preprocess
from utils.grownet import DynamicNet
from utils.models import GON, NN
import cv2
import numpy as np
import time
import torch
import wandb


class MetricsManager:
    def __init__(self, wandb_run):
        self.wandb = wandb_run

    def log_weak_epoch(self, epoch, stage, loss_epoch, optimizer, pbar=None):
        if pbar:
            pbar.set_postfix({
                'lr': optimizer.param_groups[0]['lr'],
                'loss_weak': loss_epoch
                # 'grad0': grad_direction[0].detach().cpu().numpy()
            })

        self.wandb.log({
            f"loss/model/{stage + 1}": loss_epoch,
            f"lr/model/{stage + 1}": optimizer.param_groups[0]['lr'],
            "Summary / Model_final_loss_epoch": loss_epoch,
            "epoch": epoch + 1
        }, commit=True)

    def log_weak_summary(self, stage, P, loss_stage, optimizer):
        weak_mse = loss_stage  # np.sqrt(loss_stage / (P["n_batches"] * P["epochs_per_stage"]))
        self.wandb.log({
            "Summary/Model_final_mse": weak_mse,
            "Summary/Model_final_lr": optimizer.param_groups[0]['lr'],
            # "Summary/Model_psnr": get_psnr(pred_img_weak, image),
            "stage": stage + 1
        }, commit=True)

    def log_ensemble_epoch(self, epoch, stage, loss_epoch, optimizer, pbar=None):
        if pbar:
            pbar.set_postfix({
                'lr': optimizer.param_groups[0]['lr'],
                'loss_ensemble': loss_epoch,
                # 'boost': net_ensemble.boost_rate.data.item()
            })

        self.wandb.log({
            f"loss/ensemble/{stage + 1}": loss_epoch,
            f"lr/ensemble/{stage + 1}": optimizer.param_groups[0]["lr"],
            "epoch": epoch + 1
        }, commit=True)

    def log_ensemble_summary(self, stage, P, loss_stage, optimizer, image, pred_image, net_ensemble):
        h, w, channels = P["input_shape"]

        ensemble_mse = loss_stage  # np.sqrt(loss_stage / (P["n_batches"] * P["epochs_per_correction"]))
        ensemble_psnr = get_psnr(pred_image, image)
        ensemble_ssim = ssim(pred_image.view(1, channels, h, w), image.view(1, channels, h, w))  # WARNING: takes a lot of memory
        self.wandb.log({
            "Summary/Ensemble_lr": optimizer.param_groups[0]['lr'],
            "Summary/Ensemble_psnr": ensemble_psnr,
            "Summary/Ensemble_ssim": ensemble_ssim,
            "Summary/Ensemble_mse": ensemble_mse,
            "Summary/boost_rate": net_ensemble.boost_rate.data,
            "stage": stage + 1
        }, commit=True)
        print(f'Ensemble_MSE: {ensemble_mse: .5f}, Ensemble_PSNR: {ensemble_psnr: .5f}, Ensemble_SSIM: {ensemble_ssim: .5f}, Boost rate: {net_ensemble.boost_rate:.4f}\n')


def train_weak(stage, P, B, net_ensemble, batches, image, pred_img_weak, metrics_manager):
    model = GON.get_model(stage, P["hidden_size"] * 2, P["hidden_size"]).to(device) if P["model"] == 'gon' \
        else NN.get_model(stage, P["hidden_size"] * 2, P["hidden_size"]).to(device)

    optimizer = get_optimizer(model.parameters(), P["lr_model"], P["optimizer"])
    # scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.1, verbose=False)
    criterion = nn.MSELoss()

    loss_epoch_cum = 0
    pbar = tqdm(range(P["epochs_per_stage"]), desc=f"Stage {stage + 1}/{P['num_nets']}: weak model training\t", unit=" epoch")
    for epoch in pbar:
        loss_batch_cum = 0
        for pos_batch in batches:
            h_idx = pos_batch[:, 0].long()
            w_idx = pos_batch[:, 1].long()
            x = input_mapping(pos_batch.to(device), B)
            y = image[h_idx, w_idx]

            middle_feat, out = net_ensemble.forward(x)
            grad_direction = -(out - y)  # grad_direction = y / (1.0 + torch.exp(y * out))

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
        loss_epoch_cum += loss_epoch
        metrics_manager.log_weak_epoch(epoch, stage, loss_epoch, optimizer, pbar)
    loss_stage = loss_epoch_cum / len(pbar)
    metrics_manager.log_weak_summary(stage, P, loss_stage, optimizer)
    return model


def fully_corrective_step(stage, P, B, net_ensemble, batches, image, pred_img_ensemble, lr_ensemble, metrics_manager):
    h, w, channels = P["input_shape"]
    lr_scaler = 1
    optimizer = get_optimizer(net_ensemble.parameters(), lr_ensemble / lr_scaler, P["optimizer"])
    # scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.1, verbose=False)
    criterion = nn.MSELoss()

    # if stage % 15 == 0:
    #     # lr_scaler *= 2
    #     lr_ensemble /= 2
    #     optimizer.param_groups[0]["lr"] = lr_ensemble

    loss_epoch_cum = 0
    pbar = tqdm(range(P["epochs_per_correction"]), desc=f"Stage {stage + 1}/{P['num_nets']}: fully corrective step", unit=" epoch")
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
        loss_epoch_cum += loss_epoch
        metrics_manager.log_ensemble_epoch(epoch, stage, loss_epoch, optimizer, pbar)
    loss_stage = loss_epoch_cum / len(pbar)
    metrics_manager.log_ensemble_summary(stage, P, loss_stage, optimizer, image, pred_img_ensemble, net_ensemble)
    return lr_ensemble


def train(net_ensemble, P, image):
    h, w, channels = P["input_shape"]
    batches = batch_generator_in_memory(P, device, shuffle=False)

    wandb_run = wandb.init(project="image_regression_final", entity="a-di", dir="../out", config=P, tags=[], mode=WANDB_MODE)  # mode="disabled", id="2"
    print(wandb_run.config)
    # wandb.watch(net_ensemble, criterion, log="all")  # log="all", log_freq=10, log_graph=True

    metrics_manager = MetricsManager(wandb_run)

    pred_img_weak = torch.Tensor(np.empty(shape=(h, w, channels))).to(device)
    pred_img_ensemble = torch.Tensor(np.empty(shape=(h, w, channels))).to(device)

    criterion = nn.MSELoss()
    B = P["B_scale"] * torch.randn((P["hidden_size"], 2)).to(device)
    lr_ensemble = P["lr_ensemble"]

    # train new model and append to ensemble
    for stage in range(P["num_nets"]):
        net_ensemble.to_train()
        weak_model = train_weak(stage, P, B, net_ensemble, batches, image, pred_img_weak, metrics_manager)
        net_ensemble.add(weak_model)

        if stage > 0 and P["epochs_per_correction"] != 0:
            lr_ensemble = fully_corrective_step(stage, P, B, net_ensemble, batches, image, pred_img_ensemble, lr_ensemble, metrics_manager)

        max_w = torch.max(pred_img_weak).cpu().numpy()
        min_w = torch.min(pred_img_weak).cpu().numpy()
        # cv2.imwrite(ROOT + f'/_ensembleP1_stage{stage + 1}.png', (pred_img_ensemble.cpu().numpy() * 255))
        cv2.imwrite(OUT_ROOT + f'/_ensembleP2_stage{stage + 1}.png', (pred_img_ensemble.cpu().numpy() * 255))
        cv2.imwrite(OUT_ROOT + f'/_weak_stage{stage + 1}.png', pred_img_weak.cpu().numpy() * 255)
        cv2.imwrite(OUT_ROOT + f'/_weakN_stage{stage + 1}.png', ((pred_img_weak.cpu().numpy() - min_w) / max_w * 255))


def main():
    P = hparams_grownet

    image = cv2.imread(INPUT_PATH)
    image = image_preprocess(image, P).to(device)
    cv2.imwrite(OUT_ROOT + '/sample.png', image.cpu().numpy() * 255)

    init_P(P, image)

    c0 = torch.Tensor([0.0, 0.0, 0.0]).to(device)  # torch.mean(image, dim=[0, 1])
    net_ensemble = DynamicNet(c0, P["boost_rate"])

    train(net_ensemble, P, image)


if __name__ == '__main__':
    start = time.time()
    # os.environ["WANDB_DIR"] = "../runs"
    WANDB_MODE = "online"  # ["online", "offline", "disabled"]
    main()
    print("seconds: ", time.time() - start)

# TODO:
# check if loss is correct - OK
# move scale into P - OK
# implement acc_gradients - OK
# implement an universal object to wrap wandb logging and metrics - OK
# check support for gon model
# recycle optimizer?
# add tunable n_layers
# try without input mapping
# compare grownet vs DNN vs my imageRegression without GrowNet
# compare imageRegression with fourier-feature-networks repo
# Finally make W&B report
# implement checkpoints saving and loading
# save final results and models of best runs
# add more metrics and organize them better
