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
from .util import utils

# from torchviz import make_dot

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

    device = utils.get_device()

    print(f"Selected device: {device}")

    model = getters.get_model(architecture=cfg.model.architecture, init_params=cfg.model.init_params)

    num_parameters = utils.count_parameters(model)
    print(f"Number of trainable parameters: {num_parameters}")

    print('Moving model to device...')
    model.to(device)

    print('Collecting model parameters...')
    params = model.parameters()

    if len(cfg.gpus) > 1:
        print("Creating DataParallel Model on gpus:", cfg.gpus)
        model = torch.nn.DataParallel(model)
        model.to(device)

    x = torch.randn(1,3, 224, 224).to(device)

    # dot = make_dot(model(x), params=dict(model.named_parameters()))
    # dot.format = 'svg'
    # dot.render('unet.svg')

    # --------------------------------------------------
    # define datasets and dataloaders
    # --------------------------------------------------
    print('Creating datasets and loaders..')

    train_dataset = getters.get_dataset(
        name=cfg.data.train_dataset.name,
        init_params=cfg.data.train_dataset.init_params
    )

    # Get collate function if exists
    collate_fn = getters.get_method(train_dataset, cfg.data.train_dataset.collate_fn)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, **cfg.data.train_dataloader,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn
    )

    valid_dataset = getters.get_dataset(
        name=cfg.data.valid_dataset.name,
        init_params=cfg.data.valid_dataset.init_params
    )
    
    # Get collate function if exists
    collate_fn = getters.get_method(train_dataset, cfg.data.valid_dataset.collate_fn)

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, **cfg.data.valid_dataloader,
        collate_fn=collate_fn,
    )

    # --------------------------------------------------
    # define losses and metrics functions
    # --------------------------------------------------

    print('Defining losses and metrics..')
    losses = {}
    for output_name in cfg.training.losses.keys():
        loss_name = cfg.training.losses[output_name].name
        loss_init_params = cfg.training.losses[output_name].init_params
        losses[output_name] = getters.get_loss(loss_name, loss_init_params)

    metrics = {}
    for output_name in cfg.training.metrics.keys():
        metrics[output_name] = []
        for metric in cfg.training.metrics[output_name]:
            metrics[output_name].append(
                getters.get_metric(metric.name, metric.init_params)
            )

    # --------------------------------------------------
    # define optimizer and scheduler
    # --------------------------------------------------
    print('Defining optimizers and schedulers..')
    optimizer = getters.get_optimizer(
        cfg.training.optimizer.name,
        model_params=params,
        init_params=cfg.training.optimizer.init_params,
    )
    if cfg.training.get("scheduler", None):
        scheduler = getters.get_scheduler(
            cfg.training.scheduler.name,
            optimizer,
            cfg.training.scheduler.init_params,
        )
    else:
        scheduler = None

    # --------------------------------------------------
    # define callbacks
    # --------------------------------------------------
    print('Defining callbacks..')
    callbacks = []

    if cfg.training.callbacks:
        callbacks_dict = {}
        for output_name in cfg.training.callbacks.keys():
            callbacks_dict[output_name] = []
            for callback in cfg.training.callbacks[output_name]:
                callbacks.append(
                    getters.get_callback(callback.name, callback.init_params)
                )

    # add scheduler callback
    if scheduler is not None:
        callbacks.append(training.callbacks.Scheduler(scheduler))

    # add default logging and checkpoint callbacks
    # if cfg.logdir is not None:

    #     # checkpointing
    #     callbacks.append(training.callbacks.ModelCheckpoint(
    #         directory=os.path.join(cfg.logdir, 'checkpoints'),
    #         monitor="val_mask_" + metrics["mask"][0].__name__,
    #         save_best=True,
    #         save_top_k=20,
    #         mode="max",
    #         verbose=True,
    #     ))

    # --------------------------------------------------
    # start training
    # --------------------------------------------------
    print('Start training...')

    # Get model output keys
    input_keys = getattr(model, "input_keys", "image")
    output_keys = getattr(model, "output_keys", "mask")

    runner = GPUNormRunner(model, 
                           model_device=device, 
                           model_input_keys=input_keys,
                           model_output_keys=output_keys)
    runner.compile(
        optimizer=optimizer,
        loss=losses,
        metrics=metrics,
    )

    runner.fit(
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        callbacks=callbacks,
        **cfg.training.fit,
    )

if __name__ == "__main__":

    #multiprocessing.set_start_method("fork")
    cfg = addict.Dict(fire.Fire(parse_config))
    logdir = cfg.get("logdir", None)
    if logdir is not None:
        save_config(cfg.to_dict(), logdir, name="config.yml")
        print(f"Config saved to: {logdir}")

    main(cfg)
    os._exit(0)
