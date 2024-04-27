import torch

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    return device