import torch

def get_optim(args, params):

    optimizer_str = args.optimizer

    if optimizer_str == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=args.base_lr)
    elif optimizer_str == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.base_lr)

    return optimizer


