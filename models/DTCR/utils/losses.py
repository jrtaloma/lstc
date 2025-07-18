import numpy as np
import torch
import torch.nn as nn


class KMeansLoss:
    """
    Class for K-Means optimization
    """
    def __init__(self, n_samples, n_clusters=2, weight=1.0, device=torch.device('cuda')):
        super(KMeansLoss, self).__init__()
        self.n_samples = n_samples
        self.n_clusters = n_clusters
        self.weight = 0.5 * weight
        self.device = device
        self.F = torch.empty(self.n_samples, self.n_clusters, requires_grad=False, device=device)
        nn.init.orthogonal_(self.F)

    def __call__(self, h):
        H = h.T
        HTH = torch.matmul(h, H)
        FTHTHF = torch.matmul(torch.matmul(self.F.T, HTH), self.F)
        loss = self.weight * (torch.trace(HTH) - torch.trace(FTHTHF))
        return loss

    def update_kmeans_f(self, h):
        W = h.T
        U, sigma, VT = np.linalg.svd(W)
        sorted_indices = np.argsort(sigma)
        topk_evecs = VT[sorted_indices[:-self.n_clusters - 1:-1], :]
        F_new = topk_evecs.T
        self.F = torch.FloatTensor(F_new).to(self.device)
