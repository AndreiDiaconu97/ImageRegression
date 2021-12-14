import time

import cv2
import numpy as np
import onnx
import onnxruntime as rt
import torch
from config.default import OUT_ROOT, hparams_xgboost, hparams_base
from utils import batch_generator, input_mapping, BatchSamplingMode


def main():
    sess = rt.InferenceSession(OUT_ROOT + "/base/model_base.onnx", providers=['CUDAExecutionProvider'])

    # get model metadata to enable mapping of new input to the runtime model.
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    onx_inputs = {input_name: mapped_input.numpy()}
    start = time.time()
    onx_outs = sess.run([label_name], onx_inputs)[0]
    print(f"Inference time: {time.time() - start:.4f}s")

    onx_outs.shape = P["image_shape"]
    cv2.imwrite(OUT_ROOT + f'/onnx_base.png', (onx_outs * 255))

    # retrieve prediction - passing in the input list (you can also pass in multiple inputs as a list of lists)
    # pred_onx = np.empty((336 * 504, 1))
    # for p in range(len(pred_onx)):
    #     pred_onx[p] = sess.run([label_name], {input_name: np.expand_dims(mapped_input[p], axis=0)})[0]
    # pred_onx.shape = (336, 504, 1)
    # cv2.imwrite(OUT_ROOT + f'/onnx_xgboost.png', (pred_onx * 255))


if __name__ == '__main__':
    P = hparams_base
    P["image_shape"] = 336, 504, 3
    P["batch_sampling_mode"] = BatchSamplingMode.whole.name
    P["input_layer_size"] = P["hidden_size"] * 2

    h, w, channels = P["image_shape"]
    positions = []

    batches = batch_generator(P, 'cpu', shuffle=False)
    # B = np.load(OUT_ROOT + "/B.npy")
    B = np.load(OUT_ROOT + "/base/B.npy")
    # B = P["B_scale"] * torch.randn((P["input_layer_size"] // 2, 2))

    mapped_input = input_mapping(batches[0], B)
    main()
