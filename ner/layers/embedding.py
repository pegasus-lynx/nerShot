import torch.nn as nn
import torch.nn.functional as F
from layers.abstract.base import AbstractLayer

class EmbeddingLayer(AbstractLayer):
    def __init__(self, gpu, emb_dim:int, vocab_size:int, padding_idx:int=0, pretrained_mat=None, requires_grad:bool=False):
        super(EmbeddingLayer, self).__init__(gpu)
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.freeze = not requires_grad

        self.embeddings = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=padding_idx)
        if pretrained_mat is not None:
            self.embeddings.weight.data.copy_(torch.from_numpy(pretrained_mat))
        if self.freeze:
            self.embeddings.weight.requires_grad = False

    def is_cuda(self):
        return self.embeddings.weight.is_cuda

    def forward(self, seqs):
        embs = self.embeddings(seqs)
        return embs