#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import addict
import argparse
import sys
import os
import torch

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src import getters
from src.training.config import parse_config
from io import BytesIO



def main(args):

    cfg = addict.Dict(parse_config(config = args.config))

    weights_path = args.weights
    export_file = weights_path.replace('.pt', '.onnx')  # filename

    print('Creating model...')

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    init_params_custom = cfg.model.init_params  # extract model initialization parameters
    init_params_custom["encoder_weights"] = None  # because we will load pretrained weights for whole model

    model = getters.get_model(architecture=cfg.model.architecture, init_params=init_params_custom)

    state_dict = torch.load(weights_path)["state_dict"]
    model.load_state_dict(state_dict)

    print('Moving model to device...')
    model.to(device)

    # Set up model to inference mode
    model.eval()

    # Input to the model
    x = torch.randn(1, 3, 512, 512, requires_grad=False).to(device)
    dynamic_axes = None
    output_name = os.path.join(cfg.logdir, cfg.onnx.output_name)

    torch.onnx.export(model, x, export_file, verbose=False, opset_version=13,
                        training=torch.onnx.TrainingMode.EVAL,
                        do_constant_folding=True,
                        input_names=['input0'],
                        output_names=['output0'],
                        dynamic_axes=dynamic_axes)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--weights', type=str, required=True, help='weights path')
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    
    main(args)
    os._exit(0)