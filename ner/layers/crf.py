import torch
import torch.nn as nn
from layers.abstract.base import AbstractLayer

class CRFLayer(AbstractLayer):
    def __init__(self, gpu, nstates:int, pad_idx, sos_idx, tag_vocab):
        super(CRFLayer, self).__init__(gpu)
        self.nstates = nstates
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.tags = tag_vocab

        self.transition_mat = nn.Parameter(torch.zeros(nstates, nstates, dtype=torch.float))
        nn.init.normal_(self.transition_mat, -1, 0.1)

        self.transition_mat.data[self.sos_idx, :] = -9999.0
        self.transition_mat.data[self.pad_idx, :] = -9999.0
        self.transition_mat.data[:, self.pad_idx] = -9999.0
        self.transition_mat.data[self.pad_idx, self.pad_idx] = 0.0

    def __repr__(self):
        return f'{self.__class__.__name__}(num_tags={self.nstates-2})'

    def get_emp_transition_mat(self, tag_seqs):
        emp_trans_mat = torch.zeros(self.nstates, self.nstates, dtype=torch.long)
        for tag_seq in tag_seqs:
            st = tag_seq[0]
            emp_trans_mat[st, self.sos_idx] += 1
            for p, tag in enumerate(tag_seq):
                if p + 1 >= len(tag_seq):
                    break
                next_tag = tag_seq[p+1]
                emp_trans_mat[next_tag, tag] += 1
        return emp_trans_mat

    def init_transition_mat_emp(self, tag_seqs):
        emp_trans_mat = self.get_emp_transition_mat(tag_seqs)
        for i in range(nstates):
            for j in range(nstates):
                if i == self.sos_idx or j == self.sos_idx:
                    continue
                if emp_trans_mat[i,j] == 0:
                    self.transition_mat[i,j] = -9999.0

    def is_cuda(self):
        return self.transition_mat.is_cuda

    def forward(self, out_fet_seqs, tag_seqs, mask_tensor=None, return_batch:bool=False):
        batch_size, max_seq_len, _ = out_fet_seqs.shape
        if mask_tensor is None:
            mask_tensor = self.set_device(torch.zeros(batch_size, max_seq_len, dtype=torch.float).fill_(1.0))
        numerator = self.numerator(out_fet_seqs, tag_seqs, mask_tensor)
        denominator = self.denominator(out_fet_seqs, mask_tensor)
        nll_loss = -torch.mean(numerator - denominator)
        
        if return_batch:
            decoded_batch = self.decode_viterbi(out_fet_seqs, mask_tensor)
            return nll_loss, decoded_batch
            
        return nll_loss

    def numerator(self, out_fet_seqs, tag_seqs, mask_tensor): 
        # out_fet_seqs : batch_size x max_seq_len x nstates
        # tag_seqs : batch_size x max_seq_len
        # mask_tensor : batch_size x max_seq_len
        batch_size, max_seq_len = mask_tensor.shape
        score = self.set_device(torch.zeros(batch_size, dtype=torch.float))
        start_tags = self.set_device(torch.zeros(batch_size, 1, dtype=torch.long).fill_(self.sos_idx))
        tag_seqs = torch.cat([start_tags, tag_seqs], 1)
        for n in range(max_seq_len):
            curr_mask = mask_tensor[:, n]
            curr_emms = self.set_device(torch.zeros(batch_size, dtype=torch.float))
            curr_transition = self.set_device(torch.zeros(batch_size, dtype=torch.float))
            for k in range(batch_size):
                curr_emms[k] = out_fet_seqs[k, n, tag_seqs[k, n+1]].unsqueeze(0)
                curr_tag_seq = tag_seqs[k]
                curr_transition[k] = self.transition_mat[curr_tag_seq[n+1], curr_tag_seq[k]].unsqueeze(0)
            score = score + curr_emms*curr_mask + curr_transition*curr_mask
        return score

    def denominator(self, out_fet_seqs, mask_tensor):
        batch_size, max_seq_len = mask_tensor.shape
        score = self.set_device(torch.zeros(batch_size, self.nstates, dtype=torch.float).fill_(-9999.0))
        score[:,self.sos_idx] = 0.0
        for n in range(max_seq_len):
            curr_mask = mask_tensor[:,n].unsqueeze(-1).expand_as(score)
            curr_score = score.unsqueeze(-1).expand(-1, *self.transition_mat.size())
            curr_emms = out_fet_seqs[:,n].unsqueeze(-1).expand_as(curr_score)
            curr_transition = self.transition_mat.unsqueeze(0).expand_as(curr_score)
            curr_score = log_sum_exp(curr_score + curr_emms + curr_transition)
            score = curr_score*curr_mask + score*(1-curr_mask)
        score = log_sum_exp(score)
        return score

    def decode_viterbi(self, out_fet_seqs, mask_tensor):
        batch_size, max_seq_len = mask_tensor.shape
        seqs_len = [int(mask_tensor[k].sum().item()) for k in range(batch_size)]

        score = self.set_device(torch.Tensor(batch_size, self.nstates).fill_(-9999.0))
        score[:, self.sos_idx] = 0.0
        back_ptrs = self.set_device(torch.LongTensor(batch_size, max_seq_len, self.nstates))
        for n in range(max_seq_len):
            curr_emms = out_fet_seqs[:, n]
            curr_score = self.set_device(torch.Tensor(batch_size, self.nstates))
            curr_back_ptrs = self.set_device(torch.LongTensor(batch_size, self.nstates))
            for curr_state in range(self.nstates):
                T = self.transition_mat[curr_state, :].unsqueeze(0).expand(batch_size, self.nstates)
                max_values, max_indices = torch.max(score + T, 1)
                curr_score[:, curr_state] = max_indices
            curr_mask = mask_tensor[:, n].unsqueeze(1).expand(batch_size, self.nstates)
            score = score * (1-curr_mask) + (curr_score + curr_emms) * curr_mask
            back_ptrs[:, n, :] = curr_back_ptrs
        best_score_batch, last_best_score_batch = torch.max(score, 1)

        best_path_batch = [[state] for state in last_best_score_batch.tolist()]
        for k in range(batch_size):
            curr_best_state = last_best_score_batch[k]
            curr_seq_len = seqs_len[k]
            for n in reversed(range(1,curr_seq_len)):
                curr_best_state = back_ptrs[k, n, curr_best_state].item()
                best_path_batch[k].insert(0, curr_best_state)
        return best_path_batch