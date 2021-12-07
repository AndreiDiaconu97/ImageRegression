import enum
import random
import time

import cv2
import numpy as np
import torch
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor


class BatchSamplingMode(enum.Enum):
    sequential = 1
    nth_element = 2
    whole = 3


def get_coordinate_batches(P, shuffle):
    h, w, channels = P["input_size"]

    positions = []
    for y in range(h):
        for x in range(w):
            positions.append((y, x))
    positions = torch.Tensor(positions).to(device)

    if P["batch_sampling_mode"] == BatchSamplingMode.nth_element.name:
        batches = (lambda source, step: [source[i::step] for i in range(step)])(positions, P["n_slices"])
    elif P["batch_sampling_mode"] == BatchSamplingMode.whole.name:
        batches = [positions]
    elif P["batch_sampling_mode"] == BatchSamplingMode.sequential.name:
        batches = torch.split(positions, P["batch_size"])
    else:
        raise ValueError('BatchSamplingMode: unknown value')
    if shuffle:
        random.shuffle(batches)
    return batches


def input_mapping(x, B):
    if B is None:
        return x
    else:
        x_proj = (2. * np.pi * x) @ B.t()
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


if __name__ == '__main__':
    start = time.time()
    device = 'cpu'
    ROOT = "../runs/image_regression"

    image = cv2.imread("../data/IMG_0201_DxO.jpg")
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, channels = image.shape
    scale = 0.25
    w = int(w * scale)
    h = int(h * scale)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(ROOT + '/sample.png', image)
    image = torch.Tensor(image).div(255).to(device)  # values:[0,1]

    P = {
        "B_scale": 0.02,
        "hidden_size": 16,
        "batch_sampling_mode": BatchSamplingMode.whole.name,
        "input_size": image.shape
    }
    batches = get_coordinate_batches(P, shuffle=True)
    B = P["B_scale"] * torch.randn((P["hidden_size"], 2)).to(device)

    # train = xgb.DMatrix(batches[0], label=image.view(-1, 3))

    params = {
        'max_depth': 12,
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

    xtrain = input_mapping(batches[0], B)
    ytrain = image.view(-1, 3)
    model = MultiOutputRegressor(xgb.XGBRegressor(**params))
    model.fit(xtrain, ytrain)
    score = model.score(xtrain, ytrain)
    print("Training score: ", score)

    ypred = model.predict(xtrain)
    pred_img = torch.Tensor(ypred).view(h, w, -1).numpy()

    cv2.imwrite(ROOT + f'/xgboost_test.png', (pred_img * 255))
    print("seconds: ", time.time() - start)
