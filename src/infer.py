import argparse
import os

import addict
from tqdm import tqdm
import torch

from .training.config import parse_config
from .training.runner import GPUNormRunner
from . import getters

def main(args):

    cfg = addict.Dict(parse_config(config = args.config))
    weights = args.weights

    # --------------------------------------------------
    # Define model
    # --------------------------------------------------

    print('Creating model...')

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    init_params_custom = cfg.model.init_params  # extract model initialization parameters
    init_params_custom["encoder_weights"] = None  # because we will load pretrained weights for whole model

    model = getters.get_model(architecture=cfg.model.architecture, init_params=init_params_custom)

    # --------------------------------------------------
    # Load weights
    # --------------------------------------------------

    if not weights:
        checkpoint_path = os.path.join(cfg.logdir, "checkpoints", "best.pth")
        state_dict = torch.load(checkpoint_path, map_location=device)["state_dict"]
        model.load_state_dict(state_dict)
    else:
        state_dict = torch.load(weights, map_location=device)["state_dict"]
        model.load_state_dict(state_dict)

    # --------------------------------------------------
    # Start inference
    # --------------------------------------------------
    runner = GPUNormRunner(model, model_device=device)
    model.eval()
    
    valid_dataset = getters.get_dataset(
        name=cfg.data.valid_dataset.name,
        init_params=cfg.data.valid_dataset.init_params
    )
      
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, **cfg.data.valid_dataloader
    )

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------

    datainterpreter = getters.get_interpreter(
        name = cfg.prediction.datainterpreter.name,
        init_params=cfg.prediction.datainterpreter.init_params
    )

    for batch in tqdm(valid_dataloader):
        predictions = runner.predict_on_batch(batch)
        datainterpreter(batch, predictions)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--weights', type=str)
    args = parser.parse_args()
    
    main(args)
    os._exit(0)