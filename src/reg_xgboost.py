import pickle
from xgboost.core import EarlyStopException
from config.default import OUT_ROOT, init_P, INPUT_PATH, hparams_xgboost
from torchmetrics.functional import ssim
from utils import batch_generator_in_memory, input_mapping, image_preprocess, get_psnr
from xgboost import callback
import cv2
import numpy as np
import time
import torch
import wandb
import xgboost as xgb

early_stop_execution = callback.early_stop(10)  # raise EarlyStopException(best_iteration)


def metrics_R_callback(channel):
    xgtrain = xgtrains[channel]

    def callback(env):
        _l = time.time()
        if _l - l["time"] > 1:  # save metrics every second
            l["time"] = _l
            ypred = torch.from_numpy(env.model.predict(xgtrain))
            y = torch.from_numpy(xgtrain.get_label())
            mse = env.evaluation_result_list[0][1] ** 2
            psnr = get_psnr(ypred, y)
            _ssim = ssim(ypred.view(1, 1, h, w), y.view(1, 1, h, w))

            wandb.log({
                "loss": mse,
                "psnr": psnr,
                "ssim": _ssim,
            })

            if psnr >= P["desired_psnr"]:
                raise EarlyStopException(env.iteration)

    return callback


if __name__ == '__main__':
    start = time.time()
    # SETUP #
    l = {'time': time.time()}
    WANDB_USER = "a-di"
    WANDB_MODE = "online"  # ["online", "offline", "disabled"]
    P = hparams_xgboost

    image = cv2.imread(INPUT_PATH)
    image = image_preprocess(image, P)
    cv2.imwrite(OUT_ROOT + '/sample.png', image.numpy() * 255)
    P["image_shape"] = image.shape
    h, w, channels = P["image_shape"]

    batches = batch_generator_in_memory(P, 'cpu', shuffle=False)
    B = P["B_scale"] * torch.randn((P["input_layer_size"] // 2, 2))
    xtrain = input_mapping(batches[0], B)
    xgtrains = {
        "R": xgb.DMatrix(xtrain, label=image.view(-1, channels)[:, 0]),
        "G": xgb.DMatrix(xtrain, label=image.view(-1, channels)[:, 1]),
        "B": xgb.DMatrix(xtrain, label=image.view(-1, channels)[:, 2])
    }

    # TRAIN #
    wandb_run = wandb.init(project="image_regression_final", entity=WANDB_USER, dir="../out", config=P, tags=["xgboost"], mode=WANDB_MODE)  # id="2"
    channels_pred = {"R": None, "G": None, "B": None}
    models = {"R": None, "G": None, "B": None}
    for c in channels_pred:
        model = xgb.train(P, xgtrains[c], num_boost_round=P["num_boost_round"], evals=[(xgtrains[c], 'Train')], verbose_eval=False,
                          callbacks=[metrics_R_callback(c), early_stop_execution])
        channels_pred[c] = model.predict(xgtrains[c])
        models[c] = model
        # pred_img = torch.Tensor(channels_pred[c]).view(h, w, 1).numpy()
        # cv2.imwrite(OUT_ROOT + f'/xgboost_{c}.png', (pred_img * 255))

    ypred_RGB = torch.tensor(np.asarray([channels_pred[c] for c in channels_pred])).permute((1, 0))
    y = image.view(-1, channels)

    # LOG summary metrics #
    mse = torch.mean((ypred_RGB - y) ** 2)
    psnr = get_psnr(ypred_RGB, y)
    _ssim = ssim(ypred_RGB.reshape(1, channels, h, w), y.view(1, channels, h, w))
    wandb.log({
        "Final/loss": mse,
        "Final/psnr": psnr,
        "Final/ssim": _ssim,
    })

    # save output #
    pred_img = ypred_RGB.view(h, w, channels).numpy()
    cv2.imwrite(OUT_ROOT + f'/xgboost_RGB.png', (pred_img * 255))

    for c in models:
        pickle.dump(models[c], open(OUT_ROOT + f"/xgboost_model_{c}.pkl", "wb"))  # saveopts model
    print("seconds: ", time.time() - start)

# TODO: add early termination based on metrics

# OLD MODEL #
# model = MultiOutputRegressor(xgb.XGBRegressor(**P, verbosity=2, seed=0))
# model_R = xgb.XGBRegressor(**P, verbosity=2, seed=0)
# model_R.fit(xtrain, ytrain_R, eval_set=[(xtrain, ytrain_R)], verbose=False, callbacks=[metrics_R_callback(), early_stop_execution])
# ypred_R = model_R.predict(xtrain)
