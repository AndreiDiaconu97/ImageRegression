import cv2
import torch
from pathlib import Path
from config.default import OUT_ROOT

# Warning: this is not supposed to work off the shelf, manual tinkering is required

if __name__ == '__main__':
    # LOAD target + prediction

    # names = ["crowd", "giraffe", "mountains", "unitn"]
    # target_img_paths = ["data/crowd_generator.bmp", "data/giraffe_low_poly.png",
    #                     "data/mountains_final.jpg", "data/unitn.png"]
    #
    # for i, name in enumerate(names):
    #     target_img_path = target_img_paths[i]
    #     for path in Path(f"{OUT_ROOT}/base/__final/{name}/").glob(f"*_{name}.png"):
    #         print(path)
    #         path = str(path)
    #         experiment = path[path.rfind("\\") + 1: path.rfind("_")]
    #
    #         image = cv2.imread(target_img_path)
    #         pred_img = cv2.imread(path)
    #
    #         img_error = torch.abs(torch.Tensor(pred_img) - torch.Tensor(image))
    #         cv2.imwrite(f"{OUT_ROOT}/base/__final/{name}/{experiment}_{name}_errorDiff.png", img_error.numpy())

    # XGBOOST #
    root = "out/xgboost/xgboost_mountains_final.jpg_PSNR26"
    target_img_path = f"{root}/sample.png"
    path = f"{root}/xgboost_RGB.png"

    image = cv2.imread(target_img_path)
    pred_img = cv2.imread(path)
    img_error = torch.abs(torch.Tensor(pred_img[:, :, 0]) - torch.Tensor(image[:, :, 0]))
    cv2.imwrite(f"{root}/_errorDiff.png", img_error.numpy())
