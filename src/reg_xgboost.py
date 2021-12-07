from config.default import OUT_ROOT, init_P, INPUT_PATH
from sklearn.multioutput import MultiOutputRegressor
from utils import BatchSamplingMode, batch_generator_in_memory, input_mapping, image_preprocess
import cv2
import enum
import numpy as np
import random
import time
import torch
import xgboost as xgb

device = 'cpu'

if __name__ == '__main__':
    start = time.time()

    P = {
        "B_scale": 0.02,
        "hidden_size": 16,
        "batch_sampling_mode": BatchSamplingMode.whole.name,
        "scale": 0.1
    }

    image = cv2.imread(INPUT_PATH)
    image = image_preprocess(image, P).to(device)
    cv2.imwrite(OUT_ROOT + '/sample.png', image.cpu().numpy() * 255)

    init_P(P, image)
    h, w, channels = P["input_shape"]

    batches = batch_generator_in_memory(P, device, shuffle=True)
    B = P["B_scale"] * torch.randn((P["hidden_size"], 2)).to(device)

    # train = xgb.DMatrix(batches[0], label=image.view(-1, 3))

    params = {
        'max_depth': 3,
        'learning_rate': 0.6,
        'subsample': 1.0,
        'objective': 'reg:squarederror',
        'tree_method': 'gpu_hist',
        'verbosity': 2,
        'num_parallel_tree': 1,
        'eval_metric': ['mae', 'rmse'],
        'seed': 0,
        'n_estimators': 800,
        'reg_lambda': 0.01,
    }
    # booster = xgb.train(param, train, num_boost_round=10)

    xtrain = input_mapping(batches[0].to(device), B)
    ytrain = image.view(-1, 3)
    model = MultiOutputRegressor(xgb.XGBRegressor(**params))
    model.fit(xtrain, ytrain)
    score = model.score(xtrain, ytrain)
    print("Training score: ", score)

    ypred = model.predict(xtrain)
    pred_img = torch.Tensor(ypred).view(h, w, channels).numpy()

    cv2.imwrite(OUT_ROOT + f'/xgboost_test.png', (pred_img * 255))
    print("seconds: ", time.time() - start)

# FIXME: why can't set device 'cuda'?
