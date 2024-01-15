import os
import multiprocessing

import addict
import fire
import torch
import pandas as pd
from torch.backends import cudnn
from sklearn.model_selection import train_test_split

from . import getters
from . import training
from .training.config import parse_config, save_config
from .training.runner import GPUNormRunner

cudnn.benchmark = True


def worker_init_fn(seed):
    import random
    import numpy as np
    import time
    seed = (seed + 1) * (int(time.time()) % 60)  # set random seed every epoch!
    random.seed(seed + 1)
    np.random.seed(seed)


def main(cfg):
    # set GPUS
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cfg.gpus)) if cfg.get("gpus") else ""

    # --------------------------------------------------
    # define model
    # --------------------------------------------------

    print('Creating model...')

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


    init_params_custom = cfg.model.init_params  # extract model initialization parameters
    init_params_custom["encoder_weights"] = None  # because we will load pretrained weights for whole model

    model = getters.get_model(architecture=cfg.model.architecture, init_params=init_params_custom)

    checkpoint_path = os.path.join(cfg.logdir, "checkpoints", "best.pth")
    state_dict = torch.load(checkpoint_path)["state_dict"]
    model.load_state_dict(state_dict)

    print('Moving model to device...')
    model.to(device)

    print('Collecting model parameters...')
    params = model.parameters()

    if len(cfg.gpus) > 1:
        print("Creating DataParallel Model on gpus:", cfg.gpus)
        model = torch.nn.DataParallel(model)
        model.to(device)

    # --------------------------------------------------
    # define datasets and dataloaders
    # --------------------------------------------------
    print('Creating datasets and loaders..')

    valid_dataset = getters.get_dataset(
        name=cfg.data.valid_dataset.name,
        init_params=cfg.data.valid_dataset.init_params
    )
    
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, **cfg.data.valid_dataloader
    )

    # --------------------------------------------------
    # define losses and metrics functions
    # --------------------------------------------------

    print('Defining metrics..')

    metrics = {}
    for output_name in cfg.training.metrics.keys():
        metrics[output_name] = []
        for metric in cfg.training.metrics[output_name]:
            metrics[output_name].append(
                getters.get_metric(metric.name, metric.init_params)
            )

    # --------------------------------------------------
    # start inference
    # --------------------------------------------------
    print('Start evaluating...')

    runner = GPUNormRunner(model, model_device=device)

    runner.compile(
        metrics=metrics,
    )

    model.eval()

    logs = runner.evaluate(valid_dataloader)

    print(logs)
    

if __name__ == "__main__":

    #multiprocessing.set_start_method("fork")
    cfg = addict.Dict(fire.Fire(parse_config))
    logdir = cfg.get("logdir", None)
    if logdir is not None:
        save_config(cfg.to_dict(), logdir, name="config.yml")
        print(f"Config saved to: {logdir}")

    main(cfg)
    os._exit(0)
