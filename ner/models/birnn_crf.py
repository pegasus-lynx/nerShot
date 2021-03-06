import torch
import torch.nn as nn

from ner.layers.birnns import BiGRULayer, BiLSTMLayer, BiRNNLayer
from ner.layers.crf import CRFLayer
from ner.layers.embedding import EmbeddingLayer
from ner.models.abstract_tagger import AbstractNERTagger, TaggerSchema

class BiRNNCRFSchema(TaggerSchema):
    def __init__(self):
        super(BiRNNSchema, self).__init__()
        self.forward_list = ['seqs']
        self.get_loss_list = ['seqs', 'tagseqs']
        self.predict_list = ['seqs']
        self.forward_out_list = ['indexes']
        self.labels_list = ['tagseqs']
        self.decoder_type = ['crf']

class BiRNNCRFTagger(AbstractNERTagger):

    def __init__(self, gpu, ntags:int, nwords:int, word_emb_dim:int,  
                word_rnn_hid_dim:int, word_emb_mat=None, activation_type:str='gelu', 
                rnn_type:str='lstm', criterion_type:str='nll', drop_ratio:float=0.1):
        super(BiRNNCRFTagger, self).__init__(gpu, ntags, activation_type, criterion_type)

        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.drop_ratio = drop_ratio

        self.word_emb_dim = word_emb_dim
        self.word_rnn_hid_dim = word_rnn_hid_dim

        self.criterion = None

        #Layers
        self.word_emb_layer = EmbeddingLayer(word_emb_dim, len(word_vocab), 
                                                pretrained_mat=word_emb_mat)
        self.drop_layer = nn.Dropout(p=drop_ratio)
        self.birnn_layer = self._get_birnn(rnn_type)
        self.lin_layer = nn.Linear(self.birnn_layer.out_dim, self.ntags+1)
        self.crf_layer = CRFLayer(gpu, self.ntags+1, 0, self.ntags)

        if gpu >= 0:
            self.cuda(device=self.gpu)

    def forward(self, seqs_tensor, tagseqs_tensor):
        rnn_outs = self._forward_encoder(seqs_tensor)
        mask = self.make_mask(seqs_tensor)
        crf_loss, idx_seqs = self.crf_layer(rnn_outs, tagseqs_tensor, 
                                            mask, return_batch=True)        
        return ((idx_seqs), crf_loss)

    def _forward_encoder(self, seqs_tensor):
        mask = self.make_mask(seqs_tensor)
        seqs_emb = self.word_emb_layer(seqs_tensor)
        seqs_emb = self.drop_layer(seqs_emb)
        rnn_out, rnn_states = self.birnn_layer(word_seq_tensor, mask)
        rnn_out = self.drop_layer(rnn_out)
        rnn_cmp = self.lin_layer(rnn_out)
        return self.apply_mask(rnn_cmp, mask)

    def get_loss(self, seqs_tensor, tagseqs_tensor):
        rnn_outs = self._forward_encoder(seqs_tensor)
        mask = self.make_mask(seqs_tensor)
        crf_loss = self.crf_layer(rnn_outs, tagseqs_tensor, mask)        
        return crf_loss

    def predict(self, seqs_tensor):
        self.eval()
        rnn_outs = self._forward_encoder(seqs_tensor)
        mask = self.make_mask(seqs_tensor)
        idx_seqs = self.crf_layer.decode_viterbi(rnn_outs, mask)
        return idx_seqs

    def _get_birnn(self, rnn_type):
        if rnn_type == 'gru':
            return BiGRULayer(gpu, self.word_emb_dim, self.word_rnn_hid_dim, 
                                self.nlayers, sort_batch=self.sort_batch)
        elif rnn_type == 'lstm':
            return BiLSTMLayer(gpu, self.word_emb_dim, self.word_rnn_hid_dim,
                                self.nlayers, sort_batch=self.sort_batch)
        else:
            return BiRNNLayer(gpu, self.word_emb_dim, self.word_rnn_hid_dim, 
                                nlayers=self.nlayers)

                