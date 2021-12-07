import enum
import random
import torch
import numpy as np


def input_mapping(x, B):
    if B is None:
        return x
    else:
        x_proj = (2. * np.pi * x) @ B.t()
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class BatchSamplingMode(enum.Enum):
    sequence = 1
    nth_element = 2
    whole = 3


def batch_generator_yield(P, shuffle=False):  # WARNING: don't use this
    h, w, channels = P["input_size"]

    if P["batch_sampling_mode"] == BatchSamplingMode.whole.name:
        for y in range(h):
            for x in range(w):
                yield torch.tensor([y, x]).unsqueeze(dim=0)

    elif P["batch_sampling_mode"] == BatchSamplingMode.sequence.name:
        batch_ids = range(P["n_batches"])  # FIXME: probably wrong
        if shuffle:
            batch_ids = list(batch_ids)
            random.shuffle(batch_ids)
        for s in batch_ids:
            yx = np.empty((P["batch_size"], 2), dtype=np.float32)
            for i in range(P["batch_size"]):
                yx[i] = ((P["batch_size"] * s + i) // w,
                         (P["batch_size"] * s + i) % w)
            if s == batch_ids[-1]:
                # removes extra elements
                yx = yx[(yx[:, 0] < h) & (yx[:, 1] < w)]
            yield torch.from_numpy(yx)

    elif P["batch_sampling_mode"] == BatchSamplingMode.nth_element.name:
        batch_ids = range(P["batch_size"])
        if shuffle:
            batch_ids = list(batch_ids)
            random.shuffle(batch_ids)
        for s in batch_ids:
            yx = np.empty((P["n_batches"], 2), dtype=np.float32)
            for i in range(P["n_batches"]):
                yx[i] = ((P["batch_size"] * i + s) // w,
                         (P["batch_size"] * i + s) % w)
            if (yx[-1][0] >= h) or (yx[-1][1] >= w):
                # removes extra elements
                yx = yx[:-1]
            yield torch.from_numpy(yx)

    else:
        raise ValueError('BatchSamplingMode: unknown value')


def batch_generator_in_memory(P, shuffle=False):
    h, w, channels = P["input_size"]

    positions = []
    for y in range(h):
        for x in range(w):
            positions.append((y, x))
    positions = torch.Tensor(positions).to('cuda')

    if P["batch_sampling_mode"] == BatchSamplingMode.nth_element.name:
        batches = (lambda source, slices: [source[i::slices] for i in range(slices)])(positions, P["n_batches"])  # FIXME: seems wrong
    elif P["batch_sampling_mode"] == BatchSamplingMode.whole.name:
        batches = [positions]
    elif P["batch_sampling_mode"] == BatchSamplingMode.sequence.name:
        batches = torch.split(positions, P["batch_size"])
    else:
        raise ValueError('BatchSamplingMode: unknown value')
    if shuffle:
        random.shuffle(batches)
    return batches


def get_optimizer(model_params, lr, optim_name):
    return torch.optim.Adam(model_params, lr) if optim_name == 'adam' \
        else torch.optim.AdamW(model_params, lr) if optim_name == 'adamW' \
        else torch.optim.SGD(model_params, lr)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr)
    # optimizer = adabound.AdaBound(model.parameters(), lr, final_lr=1)
