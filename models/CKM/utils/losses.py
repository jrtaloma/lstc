import torch
import torch.nn as nn
import torch.nn.functional as F


class KMeansLoss(nn.Module):
    def __init__(self, centroids):
        super().__init__()
        # Initializing centroids
        self.centroids = nn.Parameter(torch.tensor(centroids))

    def forward(self, z, cluster_logits, temperature):
        one_hot = F.gumbel_softmax(cluster_logits, tau=temperature, hard=True)
        #assert (z.shape[0] == one_hot.shape[0]), 'Length Mismatch'
        #assert (z.shape[1] == self.centroids.shape[1]), 'Dimensions Mismatch'
        loss = torch.mean(torch.square(z - torch.matmul(one_hot, self.centroids)))
        return loss
