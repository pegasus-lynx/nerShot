import torch
import torch.nn as nn
from layers.abstract.rnn_base import AbstractRNNLayer

class BiLSTMLayer(AbstractRNNLayer):
    def __init__(self, gpu, inp_dim:int, hid_dim:int, nlayers:int=1, sort_batch:bool=True):
        super(BiLSTMLayer, self).__init__(gpu, inp_dim, hid_dim, nlayers, bi_dir=True)
        self.sort_batch = sort_batch
        self.rnn = nn.LSTM(inp_dim, hid_dim, nlayers, 
                            batch_first=True, bidirectional=True)

    def custom_init(self):
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
        for names in self.rnn._all_weights:
            for name in filter(lambda n: 'bias' in n, names):
                bias = getattr(self.rnn, name)
                bias.data.fill_(0.0)
                sz = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.0)

    def forward(self, inp_tensor, mask_tensor):
        batch_size, seq_len, _ = inp_tensor.shape

        h0 = self.set_device(torch.zeros(self.nlayers * self.ndirs, batch_size, self.hid_dim))
        c0 = self.set_device(torch.zeros(self.nlayers * self.ndirs, batch_size, self.hid_dim))

        if self.sort_batch:
            inp_packed, rev_sort_index = self.pack_sorted(inp_tensor, mask_tensor)
            out_packed, states = self.rnn(inp_packed, (h0, c0))
            out_tensor = self.unpack_sorted(out_packed, seq_len, rev_sort_index)
        else:
            inp_packed = self.pack(inp_tensor, mask_tensor)
            out_packed, states = self.rnn(inp_packed, (h0, c0))
            out_tensor = self.unpack(out_packed, seq_len)

        return out_tensor, states

    def is_cuda(self):
        return self.rnn.weight_hh_l0.is_cuda

class BiGRULayer(AbstractRNNLayer):
    def __init__(self, gpu, inp_dim:int, hid_dim:int, nlayers:int=1, sort_batch:bool=True):
        super(BiLSTMLayer, self).__init__(gpu, inp_dim, hid_dim, nlayers, bi_dir=True)
        self.sort_batch = sort_batch
        self.rnn = nn.GRU(inp_dim, hid_dim, nlayers, 
                            batch_first=True, bidirectional=True)

    def forward(self, inp_tensor, mask_tensor):
        batch_size, seq_len, _ = inp_tensor.shape
        h0 = self.set_device(torch.zeros(self.nlayers * self.ndirs, batch_size, self.hid_dim))

        if self.sort_batch:
            inp_packed, rev_sort_index = self.pack_sorted(inp_tensor, mask_tensor)
            out_packed, states = self.rnn(inp_packed, h0)
            out_tensor = self.unpack_sorted(out_tensor, seq_len, rev_sort_index)
        else:
            inp_packed = self.pack(inp_tensor, mask_tensor)
            out_packed, states = self.rnn(inp_packed, h0)
            out_tensor = self.unpack(out_packed, seq_len)

        return out_tensor, states

    def is_cuda(self):
        return self.rnn.weight_hh_l0.is_cuda

class BiRNNLayer(AbstractRNNLayer):
    
    def __init__(self, gpu, inp_dim:int, hid_dim:int, nlayers:int=1):
        super(BiRNNLayer, self).__init__(gpu, inp_dim, hid_dim, nlayers, bi_dir=True)
        self.rnn = nn.RNN(inp_dim, hid_dim, nlayers, 
                            batch_first=True, bidirectional=True)

    def forward(self, inp_tensor, mask_tensor):
        batch_size, seq_len, _ = inp_tensor.shape
        h0 = self.set_device(torch.zeros(self.nlayers * self.ndirs, batch_size, self.hid_dim))
        out_tensor, states = self.rnn(inp_tensor, h0)
        return self.apply_mask(out_tensor, mask_tensor), states

    def is_cuda(self):
        return self.rnn.weight_hh_l0.is_cuda