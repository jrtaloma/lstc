import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class InstanceContrastiveLoss(nn.Module):
    """
    Class for instance contrastive learning optimization
    """
    def __init__(self):
        super(InstanceContrastiveLoss, self).__init__()

    def forward(self, z, z_a, temperature=1.0):
        LARGE_NUM = 1e9

        # Normalize data rows
        z_ = F.normalize(z, dim=1)
        z_a_ = F.normalize(z_a, dim=1)

        batch_size = z_.size(0)
        labels = torch.arange(batch_size, device=z.device)
        masks = torch.eye(batch_size, device=z.device)

        logits_aa = torch.matmul(z_, z_.T) / temperature
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(z_a_, z_a_.T) / temperature
        logits_bb = logits_bb - masks * LARGE_NUM

        logits_ab = torch.matmul(z_, z_a_.T) / temperature
        logits_ba = torch.matmul(z_a_, z_.T) / temperature

        loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels)
        loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels)

        loss = loss_a + loss_b

        return loss


class ClusterContrastiveLoss(nn.Module):
    """
    Class for entropy of the cluster assignment
    """
    def __init__(self):
        super(ClusterContrastiveLoss, self).__init__()

    def forward(self, q, q_a, temperature=1.0):
        LARGE_NUM = 1e9

        # Probability distribution for each sample over k
        q_ = F.softmax(q, dim=1)
        q_a_ = F.softmax(q_a, dim=1)

        # Cluster entropy loss
        p_q = torch.sum(q_, dim=0)
        p_q = p_q / p_q.sum()
        p_q_a = torch.sum(q_a_, dim=0)
        p_q_a = p_q_a / p_q_a.sum()
        ne_loss = torch.sum(p_q * torch.log(p_q)) + torch.sum(p_q_a * torch.log(p_q_a))

        # Cluster contrastive loss
        # Normalize cluster columns
        #q_ = F.normalize(q_, dim=0)
        #q_a_ = F.normalize(q_a_, dim=0)

        q_ = q_.T
        q_a_ = q_a_.T

        n_clusters = q_.size(0)
        labels = torch.arange(n_clusters, device=q.device)
        masks = torch.eye(n_clusters, device=q.device)

        logits_aa = F.cosine_similarity(q_.unsqueeze(1), q_.unsqueeze(0), dim=2) / temperature
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = F.cosine_similarity(q_a_.unsqueeze(1), q_a_.unsqueeze(0), dim=2) / temperature
        logits_bb = logits_bb - masks * LARGE_NUM

        logits_ab = F.cosine_similarity(q_.unsqueeze(1), q_a_.unsqueeze(0), dim=2) / temperature
        logits_ba = F.cosine_similarity(q_a_.unsqueeze(1), q_.unsqueeze(0), dim=2) / temperature

        loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels)
        loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels)

        loss = loss_a + loss_b + ne_loss

        return loss
