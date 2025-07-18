import torch


def target_distribution(q):
    ## p : ground truth distribution
    p = torch.pow(q, 2) / torch.sum(q, dim=0).view(1, -1)
    sum_columns_p = torch.sum(p, dim=1).view(-1, 1)
    p = p / sum_columns_p
    return p


def kl_loss_function(input, pred):
    out = input * torch.log((input) / (pred))
    return torch.mean(torch.sum(out, dim=1))
