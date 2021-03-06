import torch.nn as nn
import torch.nn.functional as F
from ner.layers.abstract.base import AbstractLayer

class CharCNN(AbstractLayer):
    def __init__(self, gpu, char_emb_dim:int, word_len:int, nfilters:int, kernel_size:int, word_len:int, pad_typr:str='same'):
        super(CharCNN, self).__init__(gpu)
        self.char_emb_dim = char_emb_dim
        self.word_len = word_len
        self.out_dim = char_emb_dim * nfilters
        self.nfilters  = filters
        self.kernel_size = kernel_size
        self.padding = 0
        if pad_type == 'same':
            self.padding = (kernel_size-1)//2
        self.conv_feature_len = word_len - kernel_size + 1 + 2*self.padding

        self.conv_layer = nn.Conv1d(in_channels = emb_dim,
                                    out_channels = self.out_dim,
                                    kernel_size = kernel_size,
                                    padding = self.padding,
                                    groups=emb_dim)
        self.max_pool_layer = nn.MaxPool1d(kernel_size=self.conv_feature_len)

    def forward(self, char_embs): # batch * seq_len * word_len * char_emb_dim
        batch, seq_len, word_len, char_emb_dim = char_embs.shape
        char_embs_flat = char_embs.view(batch*seq_len, word_len, char_emb_dim)
        char_embs_flat = char_embs_flat.permute(0,2,1)
        
        conv_features = self.conv_layer(char_embs_flat)
        char_features = self.max_pool_layer(conv_features) # batch*seq_len x emb_dim x 1

        char_features = char_features.permute(0,2,1)
        char_features = char_features.view(batch, seq_len, self.out_dim)
        return char_features # batch x seq_len x out_dim

    def is_cuda(self):
        return self.conv_layer.weight.is_cuda