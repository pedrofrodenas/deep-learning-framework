def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params