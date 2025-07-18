import torch
import torch.nn.functional as F

def sample_gumbel(shape, device, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature=1.0):
    y = logits + sample_gumbel(logits.size(), device=logits.device)
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=1.0, hard=False):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    if not hard:
        return y
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y

def softmax_logits(encode_output, centroids, sigma=1.0):
    assert (encode_output.shape[1] == centroids.shape[1]), 'Dimensions Mismatch'
    dist_sq = (-1 / sigma**2) * (encode_output[:, None, :] - centroids[None, :, :]).norm(2, dim=2).square()
    logits = F.log_softmax(dist_sq, dim=1)
    return logits
