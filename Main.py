from Diffusion.Train import train, eval

import random
import numpy as np
import torch

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(model_config = None):
    seed = 1  
    set_random_seed(seed)

    modelConfig = {
        "state": "eval", # or eval
        "epoch": 8000,
        "batch_size": 1,
        "T": 1000,
        "channel": 64,
        "channel_mult": [1, 2, 3, 4, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 256,
        "grad_clip": 1.,
        "nrow": 8,
        "device": "cuda", 
        "training_load_weight": None,
        "save_weight_dir": "./model/",
        "BG-De_weight": "./BG-De_model/kvasir.pt",
        "Detection_weight": "./Detection_model/yolo.pt",
        "train_image_dir":".dataset/images/",
        "train_mask_dir":".dataset/masks/",
        "inpaint_image":"test/test1/ori.png",
        "inpaint_mask":"test/test1/mask.png",
        "sampled_dir": "./SampledImgs/"
        }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)

if __name__ == '__main__':
    main()
