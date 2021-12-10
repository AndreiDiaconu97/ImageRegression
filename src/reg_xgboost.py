import pickle
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

            if psnr >= P["desired_psnr"]:
                raise EarlyStopException(env.iteration)

    return callback


if __name__ == '__main__':
    start = time.time()
    WANDB_CFG = {
        "entity": "a-di",
        "mode": "online",  # ["online", "offline", "disabled"]
        "tags": ["xgboost"],
        "group": "exp_1",
        "job_type": None,
        "id": None
    }

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
    channels_pred = {"R": None, "G": None, "B": None}
    models = {"R": None, "G": None, "B": None}
    with wandb.init(project="image_regression_final", **WANDB_CFG, dir="../out", config=P):
        for c in channels_pred:
            model = xgb.train(P, xgtrains[c], num_boost_round=P["num_boost_round"], evals=[(xgtrains[c], 'Train')], verbose_eval=False,
                              callbacks=[metrics_callback(c), early_stop_execution])
            channels_pred[c] = model.predict(xgtrains[c])
            models[c] = model
            # pred_img = torch.Tensor(channels_pred[c]).view(h, w, 1).numpy()
            # cv2.imwrite(OUT_ROOT + f'/xgboost_{c}.png', (pred_img * 255))

    ypred_RGB = torch.tensor(np.asarray([channels_pred[c] for c in channels_pred])).permute((1, 0))
    y = image.view(-1, channels)

    # LOG summary metrics #
    wandb.log({
        "Final/loss": torch.mean((ypred_RGB - y) ** 2),
        "Final/psnr": get_psnr(ypred_RGB, y),
        "Final/ssim": ssim(ypred_RGB.reshape(1, channels, h, w), y.view(1, channels, h, w)),
    })

    # save output #
    pred_img = ypred_RGB.view(h, w, channels).numpy()

    # FIXME: save onnx model + B
    # num_features = 32
    # initial_type = [('mapped_coordinates', FloatTensorType([1, num_features]))]
    # onx = onnxmltools.convert.convert_xgboost(models["R"], initial_types=initial_type)
    # with open(OUT_ROOT + "/xgboost.onnx", "wb") as f:
    #     f.write(onx.SerializeToString())
    #
    # pickle.dump(models, open(OUT_ROOT + f"/xgboost_RGBmodels.pkl", "wb"))
    # np.save(OUT_ROOT + "/B", B)

    cv2.imwrite(OUT_ROOT + f'/xgboost_RGB.png', (pred_img * 255))
    print("seconds: ", time.time() - start)

# OLD MODEL #####################################################################
# model = MultiOutputRegressor(xgb.XGBRegressor(**P, verbosity=2, seed=0))
# model_R = xgb.XGBRegressor(**P, verbosity=2, seed=0)
# model_R.fit(xtrain, ytrain_R, eval_set=[(xtrain, ytrain_R)], verbose=False, callbacks=[metrics_R_callback(), early_stop_execution])
# ypred_R = model_R.predict(xtrain)
