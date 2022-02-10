import argparse
import os
import pickle
from datetime import timedelta
import onnxmltools
from onnxconverter_common import FloatTensorType
from xgboost.core import EarlyStopException
import config.default
from config.default import OUT_ROOT, init_P, INPUT_PATH, hparams_xgboost
from torchmetrics.functional import ssim
from utils import batch_generator, input_mapping, image_preprocess, get_psnr
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
    parser.add_argument("--eval_metric", type=str,
                        help="rmse, mae, ...")
    parser.add_argument("--posEnc_size", type=int,
                        help="width of positional encoding input layer")
    parser.add_argument("--lambda", type=float,
                        help="lambda")
    parser.add_argument("--learning_rate", type=float,
                        help="learning rate")
    parser.add_argument("--max_depth", type=int,
                        help="maximum depth of an individual tree")
    parser.add_argument("--num_boost_round", type=int,
                        help="like training epochs?")
    parser.add_argument("--reg_lambda", type=float,
                        help="some regularizer")
    parser.add_argument("--scale", type=float,
                        help="online image resizing factor")
    parser.add_argument("--desired_psnr", type=float,
                        help="target quality for the individual color channel")

    configargs = parser.parse_args()

    P = hparams_xgboost
    if configargs.cfg_name:
        try:
            P = getattr(config.default, configargs.cfg_name)
        except:
            print("ERROR: Wrong configuration name argument")
    tmp_args = {k: v for k, v in configargs.__dict__.items() if k not in ["cfg_name", "run_name", "input_img", "max_mins"]}
    for k, v in tmp_args.items():
        if v is not None:
            P[k] = v

    FOLDER = os.path.join("xgboost", configargs.run_name) if configargs.run_name else os.path.join("xgboost", "unnamed")
    PATH_MODEL = os.path.join(OUT_ROOT, FOLDER, "xgboost_RGBmodels.pkl")
    PATH_ONNX = os.path.join(OUT_ROOT, FOLDER, "xgboost.onnx")

    if not os.path.exists(os.path.join(OUT_ROOT, FOLDER)):
        os.makedirs(os.path.join(OUT_ROOT, FOLDER))

    WANDB_CFG = {
        "entity": "a-di",
        "mode": "online",  # ["online", "offline", "disabled"]
        "tags": ["xgboost"],
        "group": None,
        "job_type": None,
        "id": None
    }
    if configargs.run_name:
        WANDB_CFG["name"] = configargs.run_name
        WANDB_CFG["tags"].append("thesis")

    image = cv2.imread(INPUT_PATH)
    image = image_preprocess(image, P)
    cv2.imwrite(os.path.join(OUT_ROOT, FOLDER, 'sample.png'), image.numpy() * 255)
    P["image_shape"] = image.shape
    h, w, channels = P["image_shape"]

    batch = batch_generator(P, 'cpu', shuffle=False)[0]
    batch = torch.stack((batch[:, 0] / h, batch[:, 1] / w), dim=1)  # normalize coordinates to [0,1]
    if P["B_scale"]:
        B = P["B_scale"] * torch.randn((P["posEnc_size"] // 2, 2))
    else:
        B = None
    xtrain = input_mapping(batch, B)
    xtrain = input_mapping(batch, B) if P["B_scale"] else xtrain
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
            initial_type = [('mapped2D', FloatTensorType([None, P["posEnc_size"]]))]
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
