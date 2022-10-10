import torch
import torch.nn as nn
import torch.nn.functional as F

class NewLossFn(nn.Module):
    def __init__(self, batch_size, temperature = 0.5, lambda_loss = 1.0):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        # self.n_temperature = n_temperature
        self.lambda_loss = lambda_loss
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
            mask[i, i] = 0
        return mask
    
    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2, l4fm1, l4fm2):
        N = 2 * self.batch_size
        z1 = F.normalize(z1, dim=0, p=2)
        #z1 = z1 - z1.mean(dim = 0)
        z2 = F.normalize(z2, dim=0, p=2)
        #z2 = z2 - z2.mean(dim = 0)
        z1mod = z1 - z1.mean(dim = 0)
        z2mod = z2 - z2.mean(dim = 0)
        crosscovmat = z1mod.T@z2mod
        loss = torch.square(self.off_diagonal(crosscovmat)).sum() + torch.square(torch.diag(crosscovmat)-1).sum()
        loss /= N

        return loss