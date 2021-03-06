import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from ner.layers.abstract.base import AbstractLayer

class AbstractRNNLayer(AbstractLayer):
    def __init__(self, gpu, inp_dim:int, hid_dim:int, nlayers:int, bi_dir:bool=False):
        super(AbstractRNNLayer, self).__init__(gpu)
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.nlayers = nlayers
        self.ndirs = 2 if bi_dir else 1
        self.out_dim = (2 if bi_dir else 1) * hid_dim

    def pack_sorted(self, inp_tensor, mask_tensor):
        seq_lens = self.get_seq_len_from_mask_tensor(mask_tensor)
        sorted_seq_lens, sort_index, rev_sort_index = self.sort_seq_lens(seq_lens)
        sorted_inp_tensor = torch.index_select(inp_tensor, dim=0, index=sort_index)
        return pack_padded_sequence(sorted_inp_tensor, lengths=sorted_seq_lens, batch_first=True, 
                                    enforce_sorted=True), rev_sort_index

    def unpack_sorted(self, out_packed, max_seq_len, rev_sort_index):
        sorted_out_tensor, _ = pad_packed_sequence(out_packed, batch_first=True, 
                                            total_length=max_seq_len)
        out_tensor = torch.index_select(sorted_out_tensor, dim=0, index=rev_sort_index)
        return out_tensor

    def pack(self, inp_tensor, mask_tensor):
        seq_lens = self.get_seq_len_from_mask_tensor(mask_tensor)
        return pack_padded_sequence(inp_tensor, lengths=seq_lens, batch_first=True, 
                                    enforce_sorted=False)

    def unpack(self, out_packed, max_seq_len):
        out_tensor, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=max_seq_len)
        return out_tensor
        
    def sort_seq_lens(self, seq_lens):
        batch_size = len(seq_lens)
        sort_indices = sorted(range(batch_size), key=seq_lens.__getitem__, reverse=True)
        rev_sort_indices = [-1 for _ in range(batch_size)]
        for i in range(batch_size):
            rev_sort_indices[sort_indices[i]] = i        
        sort_index = self.set_device(torch.tensor(sort_indices, dtype=torch.long))
        rev_sort_index = self.set_device(torch.tensor(rev_sort_indices, dtype=torch.long))
        return sorted(seq_lens, reverse=True), sort_index, rev_sort_index

    def get_seq_len_from_mask_tensor(self, mask_tensor):
        batch_size = mask_tensor.shape[0]
        return [int(mask_tensor[k].sum().item()) for k in range(batch_size)]