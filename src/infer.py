#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 14:18:02 2022

@author: pedrofRodenas
"""

import argparse
import os
import fire
import torch
import addict

from tqdm import tqdm
import cv2
import numpy as np

from .training.runner import GPUNormRunner
from .training.config import parse_config

from . import getters


def model_from_config(cfg):
    """Create model from configuration specified in config file and load checkpoint weights"""
    init_params = cfg.model.init_params  # extract model initialization parameters
    init_params["encoder_weights"] = None  # because we will load pretrained weights for whole model
    model = getters.get_model(architecture=cfg.model.architecture, init_params=init_params)
    checkpoint_path = os.path.join(cfg.logdir, "checkpoints", "best.pth")
    state_dict = torch.load(checkpoint_path)["state_dict"]
    model.load_state_dict(state_dict)
    return model

def model_from_epoch(cfg):
    """Create model from configuration specified in config file and load checkpoint weights"""
    init_params = cfg.model.init_params  # extract model initialization parameters
    init_params["encoder_weights"] = None  # because we will load pretrained weights for whole model
    model = getters.get_model(architecture=cfg.model.architecture, init_params=init_params)

    key = "k-ep[{0}]".format(cfg.prediction.epoch)

    checkpoint_folder = os.path.join(cfg.logdir, "checkpoints")

    file_names = next(os.walk(checkpoint_folder))[2]
    file_names = filter(lambda f: f.startswith(key), file_names)
    file_name = sorted(file_names)

    if not file_name:
        raise ValueError('Check config prediction.epoch value, there is no models on: {0} with epoch: {1}'.format(checkpoint_folder, key))
    
    checkpoint_path = os.path.join(checkpoint_folder, file_name[0])
    
    state_dict = torch.load(checkpoint_path)["state_dict"]
    model.load_state_dict(state_dict)
    return model



def main(args):

    cfg = addict.Dict(parse_config(config = args.config))

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cfg.gpus)) if cfg.get("gpus") else ""

    # --------------------------------------------------
    # define model
    # --------------------------------------------------

    print('Creating model...')

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    if not os.path.exists(args.dst_dir):
        os.makedirs(args.dst_dir)


    init_params_custom = cfg.model.init_params  # extract model initialization parameters
    init_params_custom["encoder_weights"] = None  # because we will load pretrained weights for whole model

    model = getters.get_model(architecture=cfg.model.architecture, init_params=init_params_custom)

    checkpoint_path = os.path.join(cfg.logdir, "checkpoints", "best.pth")
    state_dict = torch.load(checkpoint_path)["state_dict"]
    model.load_state_dict(state_dict)

    print('Moving model to device...')
    model.to(device)
    
    # --------------------------------------------------
    # start inference
    # --------------------------------------------------
    runner = GPUNormRunner(model, model_device=device)
    model.eval()
    
    valid_dataset = getters.get_dataset(
        name=cfg.data.test_dataset.name,
        init_params=cfg.data.test_dataset.init_params
    )
      
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, **cfg.data.valid_dataloader
    )

    for batch in tqdm(valid_dataloader):
        predictions = runner.predict_on_batch(batch)['mask']
        
        imgs = batch['image']
        ids = batch['id']
        tops = batch['top']
        lefts = batch['left']
        rs = batch['r']

        for prediction, img, id, top, left, r in zip(predictions, imgs, ids, tops, lefts, rs):

            prediction = prediction.round().int().cpu().numpy().astype("uint8").squeeze()
            top = top.cpu().numpy()
            left = left.cpu().numpy()
            r = r.cpu().numpy()
            shape = prediction.shape
            crop_pred = prediction[top:shape[0]-top, left:shape[1]-left]

            mask = cv2.resize(crop_pred,
                    (int(crop_pred.shape[1] / r), int(crop_pred.shape[0] / r)),
                    interpolation = cv2.INTER_NEAREST
                )

            cv2.imwrite(os.path.join(args.dst_dir, os.path.splitext(id)[0] + '.png'), mask*255)
           



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--dst_dir', type=str, required=True)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    
    main(args)
    os._exit(0)