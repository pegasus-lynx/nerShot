import torch
import torch.nn as nn

class CosineSim(nn.Module):

    def __init__(self, alpha:float=5.0, reduction:str='mean'):
        super(CosineSim, self).__init__()
        self.alpha = alpha
        self.reduction = reduction

        self.cosine_sim = nn.CosineSimilarity(dim=1)

    def forward(self, logits, labels):
        # logits : d1 * d2 * ... dk * D
        # labels : d1 * d2 * ... dk
        vec_dim = logits.shape[-1]
        original_shape = labels.shape

        labels_one_hot = torch.zeros_like(logits)
        labels_one_hot = labels_one_hot.view(-1, vec_dim)
        labels = labels.view(-1, 1)
        labels_one_hot.scatter_(1,labels,1)
        logits = logits.view(-1, vec_dim)

        sims = self.cosine_sim(logits, labels)

        loss_tensor = ((1-sims) / (1+sims)) * self.alpha

        if reduction == 'mean':
            loss = torch.mean(loss_tensor)
        elif reduction == 'sum':
            loss = torch.sum(loss_tensor)
        else:
            loss = loss_tensor

        return loss