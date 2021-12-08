import enum
import random
import cv2
import torch
import numpy as np
from utils.models import SIREN_grownet, NN_grownet, NN, gon_model


class BatchSamplingMode(enum.Enum):
    sequence = 1
    nth_element = 2
    whole = 3


def image_preprocess(image, P):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, channels = image.shape
    h *= P["scale"]
    w *= P["scale"]
    image = cv2.resize(image, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
    image = torch.Tensor(image).div(255)  # values:[0,1]
    return image


def input_mapping(x, B):
    if B is None:
        return x
    else:
        x_proj = (2. * np.pi * x) @ B.t()
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def get_psnr(pred, target):
    return 10 * torch.log10((torch.max(pred) ** 2) / torch.mean((pred - target) ** 2))


def get_model(P, stage=0):
    if P["type"] == "base":
        if P["model"] == 'siren':
            model = gon_model(P["input_layer_size"], P["hidden_size"], P["hidden_layers"])
        elif P["model"] == 'nn':
            model = NN(P["input_layer_size"], P["hidden_size"], P["hidden_layers"])
        else:
            raise ValueError(f'weak model type: unknown value {P["model"]}')
        return model

    elif P["type"] == "grownet":
        if P["model"] == 'siren':
            model = SIREN_grownet.get_model(stage, P["input_layer_size"], P["hidden_size"], P["hidden_layers"])
        elif P["model"] == 'nn':
            model = NN_grownet.get_model(stage, P["input_layer_size"], P["hidden_size"], P["hidden_layers"])
        else:
            raise ValueError(f'weak model type: unknown value {P["model"]}')
        return model

    else:
        raise ValueError(f'unknown network type: {P["type"]}')


def get_optimizer(model_params, lr, optim_name):
    return torch.optim.Adam(model_params, lr) if optim_name == 'adam' \
        else torch.optim.AdamW(model_params, lr) if optim_name == 'adamW' \
        else torch.optim.SGD(model_params, lr)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr)
    # optimizer = adabound.AdaBound(model.parameters(), lr, final_lr=1)


def batch_generator_in_memory(P, device, shuffle=False):
    h, w, channels = P["image_shape"]

    positions = []
    for y in range(h):
        for x in range(w):
            positions.append((y, x))
    positions = torch.Tensor(positions).to(device)

    if not "batch_sampling_mode" in P or P["batch_sampling_mode"] == BatchSamplingMode.whole.name:
        batches = [positions]
    elif P["batch_sampling_mode"] == BatchSamplingMode.nth_element.name:
        batches = [positions[i:: P["n_batches"]] for i in range(P["n_batches"])]
    elif P["batch_sampling_mode"] == BatchSamplingMode.sequence.name:
        batches = list(torch.split(positions, P["batch_size"]))
    else:
        raise ValueError(f'BatchSamplingMode: unknown value {P["batch_sampling_mode"]}')
    if shuffle:
        random.shuffle(batches)
    return batches

# def batch_generator_yield(P, shuffle=False):  # WARNING: don't use this
#     h, w, channels = P["input_size"]
#
#     if P["batch_sampling_mode"] == BatchSamplingMode.whole.name:
#         for y in range(h):
#             for x in range(w):
#                 yield torch.tensor([y, x]).unsqueeze(dim=0)
#
#     elif P["batch_sampling_mode"] == BatchSamplingMode.sequence.name:
#         batch_ids = range(P["n_batches"])  # FIXME: probably wrong
#         if shuffle:
#             batch_ids = list(batch_ids)
#             random.shuffle(batch_ids)
#         for s in batch_ids:
#             yx = np.empty((P["batch_size"], 2), dtype=np.float32)
#             for i in range(P["batch_size"]):
#                 yx[i] = ((P["batch_size"] * s + i) // w,
#                          (P["batch_size"] * s + i) % w)
#             if s == batch_ids[-1]:
#                 # removes extra elements
#                 yx = yx[(yx[:, 0] < h) & (yx[:, 1] < w)]
#             yield torch.from_numpy(yx)
#
#     elif P["batch_sampling_mode"] == BatchSamplingMode.nth_element.name:
#         batch_ids = range(P["batch_size"])
#         if shuffle:
#             batch_ids = list(batch_ids)
#             random.shuffle(batch_ids)
#         for s in batch_ids:
#             yx = np.empty((P["n_batches"], 2), dtype=np.float32)
#             for i in range(P["n_batches"]):
#                 yx[i] = ((P["batch_size"] * i + s) // w,
#                          (P["batch_size"] * i + s) % w)
#             if (yx[-1][0] >= h) or (yx[-1][1] >= w):
#                 # removes extra elements
#                 yx = yx[:-1]
#             yield torch.from_numpy(yx)
#
#     else:
#         raise ValueError('BatchSamplingMode: unknown value')
