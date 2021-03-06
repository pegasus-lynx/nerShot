import torch.nn as nn
import torch.nn.functional as F
from ner.layers.abstract.rnn_base import AbstractRNNLayer

class CharBiRNNLayer(AbstractRNNLayer):
    def __init__(self, gpu, emb_dim:int, hid_dim:int, rnn_type:str='lstm' nlayers:int=1, bi_dir:bool=True):
        super(CharRNN, self).__init__(gpu, emb_dim, hid_dim, nlayers, bi_dir=True)
        assert rnn_type in ['lstm', 'gru', 'rnn']
        
        self.rnn_type = rnn_type
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(emb_dim, hid_dim, nlayers, 
                                batch_first=True, bidirectional=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(emb_dim, hid_dim, nlayers,
                                batch_first=True, bidirectional=True)
        else:
            self.rnn = nn.RNN(emb_dim, hid_dim, nlayers,
                                batch_first=True, bidirectional=True)

    def forward(self, char_seqss): # char_seqss : batch_size x max_seq_len x word_len x char_emb_dim
        batch_size, seq_len, word_len, char_emb_dim = char_seqss.shape
        char_seqss_flat = char_seqss.view(-1, word_len, char_emb_dim)
        inp_tensor = self.set_device(char_seqss_flat)

        h0 = self.set_device(torch.zeros(self.nlayers * self.ndirs, batch_size, self.hid_dim))
        inp_states = h0
        if self.rnn_type == 'lstm':
            c0 = self.set_device(torch.zeros(self.nlayers * self.ndirs, batch_size, self.hid_dim))
            inp_states = (h0, c0)

        _, out_states = self.rnn(inp_tensor, inp_states)

        hid_states = out_states
        if self.rnn_type == 'lstm':
            hid_states = out_states[0]
        
        char_fets = hid_states.view(batch_size, seq_len, -1)
        return char_fets
        