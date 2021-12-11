import os
import pickle
from datetime import timedelta
import onnxmltools
from onnxconverter_common import FloatTensorType
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


def metrics_callback(channel):
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

            print(f"[{str(timedelta(seconds=time.time() - start))[:-5]}] Channel {channel}, - loss: {mse:.5f}, psnr: {psnr:.5f}/{P['desired_psnr']}, ssim: {_ssim:.5f}")

            if psnr >= P["desired_psnr"]:
                raise EarlyStopException(env.iteration)

    return callback


if __name__ == '__main__':
    start = time.time()
    l = {'time': 0}

    FOLDER = "xgboost"
    PATH_MODEL = os.path.join(OUT_ROOT, FOLDER, "xgboost_RGBmodels.pkl")
    PATH_ONNX = os.path.join(OUT_ROOT, FOLDER, "xgboost.onnx")
    WANDB_CFG = {
        "entity": "a-di",
        "mode": "online",  # ["online", "offline", "disabled"]
        "tags": ["xgboost"],
        "group": None,
        "job_type": None,
        "id": None
    }

    P = hparams_xgboost

    image = cv2.imread(INPUT_PATH)
    image = image_preprocess(image, P)
    cv2.imwrite(os.path.join(OUT_ROOT, FOLDER, 'sample.png'), image.numpy() * 255)
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
    channels_pred = {"R": None, "G": None, "B": None}
    models = {"R": None, "G": None, "B": None}
    with wandb.init(project="image_regression_final", **WANDB_CFG, dir=OUT_ROOT, config=P):
        for c in channels_pred:
            model = xgb.train(P, xgtrains[c], num_boost_round=P["num_boost_round"], evals=[(xgtrains[c], 'Train')], verbose_eval=False,
                              callbacks=[metrics_callback(c), early_stop_execution])
            channels_pred[c] = model.predict(xgtrains[c])
            models[c] = model
            # pred_img = torch.Tensor(channels_pred[c]).view(h, w, 1).numpy()
            # cv2.imwrite(OUT_ROOT + f'/xgboost_{c}.png', (pred_img * 255))

        ypred_RGB = torch.tensor(np.asarray([channels_pred[c] for c in channels_pred])).permute((1, 0))
        y = image.view(-1, channels)

        pickle.dump({"models": models, "B": B}, open(PATH_MODEL, "wb"))
        np.save(OUT_ROOT + "/B", B)
        onnx_s = 0
        for c in models:
            initial_type = [('mapped2D', FloatTensorType([None, P["input_layer_size"]]))]
            onx = onnxmltools.convert.convert_xgboost(models[c], initial_types=initial_type)
            f_path = f"{PATH_ONNX.removesuffix('.onnx')}{c}.onnx"
            with open(f_path, "wb") as f:
                f.write(onx.SerializeToString())
            onnx_s += os.path.getsize(f_path) / 1000

        pred_img = ypred_RGB.view(h, w, channels).numpy()
        img_error = image - pred_img
        img_error = ((img_error - img_error.min()) * 1 / (img_error.max() - img_error.min())).numpy()
        cv2.imwrite(os.path.join(OUT_ROOT, FOLDER, 'xgboost_RGB.png'), pred_img * 255)
        cv2.imwrite(os.path.join(OUT_ROOT, FOLDER, 'xgboost_error.png'), img_error * 255)

        wandb.log({
            "Final/loss": torch.mean((ypred_RGB - y) ** 2),
            "Final/psnr": get_psnr(ypred_RGB, y),
            "Final/ssim": ssim(ypred_RGB.reshape(1, channels, h, w), y.view(1, channels, h, w)),
            "Final/model(KB)": os.path.getsize(PATH_MODEL) / 1000,
            "Final/onnx(KB)": onnx_s,
            "image": wandb.Image(cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB) * 255),
            "image_error": wandb.Image(cv2.cvtColor(img_error, cv2.COLOR_BGR2RGB) * 255)
        })

    print("seconds: ", time.time() - start)

# OLD MODEL #####################################################################
# model = MultiOutputRegressor(xgb.XGBRegressor(**P, verbosity=2, seed=0))
# model_R = xgb.XGBRegressor(**P, verbosity=2, seed=0)
# model_R.fit(xtrain, ytrain_R, eval_set=[(xtrain, ytrain_R)], verbose=False, callbacks=[metrics_R_callback(), early_stop_execution])
# ypred_R = model_R.predict(xtrain)
