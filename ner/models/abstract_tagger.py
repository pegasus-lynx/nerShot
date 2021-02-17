import math
import torch
import torch.nn as nn
import numpy as np

from ..factories.activation import ActivationFactory
from ..factories.criterion import CriterionFactory

class TaggerSchema():

    def __init__(self):
        """ 
            The elements of the lists ( except forward_out_list ) 
            must be the names of the keys from the dataset_file 
        """
        self.forward_list = []
        self.get_loss_list = []
        self.predict_list = []
        self.forward_out_list = []
        self.labels_list = []

    def get_keys(self):
        keys = []
        keys.extend(self.forward_list)
        keys.extend(self.get_loss_list)
        keys.extend(self.predict_list)
        keys.extend(self.labels_list)
        return set(keys)

    @property
    def forward(self):
        return self.forward_list

    @forward.setter
    def forward(self, value):
        self.forward_list = value

    @property
    def get_loss(self):
        return self.get_loss_list

    @get_loss.setter
    def get_loss(self, value):
        self.get_loss_list = value

    @property
    def predict(self):
        return self.predict_list

    @predict.setter
    def predict(self, value):
        self.predict_list = value

    @property
    def forward_out(self):
        return self.forward_out_list

    @forward_out.setter
    def forward_out(self, value):
        self.forward_out_list = value

    @property
    def labels(self):
        return self.labels_list

    @labels.setter
    def labels(self, value):
        self.labels_list = value

class AbstractNERTagger(nn.Module):

    def __init__(self, gpu, ntags:int, nwords:int, activation_type='gelu', criterion_type='nll'):
        self.gpu = gpu
        self.ntags = ntags
        self.act_layer = ActivationFactory.create(activation_type)
        self.criterion = CriterionFactory.create(criterion_type)

    def set_device(self, tensor):
        if self.gpu >= 0:
            return tensor.cuda(device=self.gpu)
        else:
            return tensor

    def set_device_(self):
        if self.gpu >=0:
            self.cuda(device=self.gpu)
        else:
            self.cpu()

    def ensure_tensor(self, inp, trainable:bool=True):
        if isinstance(inp, np.ndarray):
            inp = torch.from_numpy(inp)
        if trainable:
            inp.requires_grad_ = True
        return self.set_device(inp)

    def forward(self, *input):
        pass

    def save(self, checkpoint_fn):
        self.cpu()
        torch.save(self, checkpoint_fn)
        self.set_device_()  

    def make_mask(self, word_seq):
        dim = len(word_seq.shape)
        mask = self.set_device()
        return mask

    def apply_mask(self, inp_tensor, mask_tensor):
        inp_tensor = self.set_device(inp_tensor)
        mask_tensor = self.set_device(mask_tensor)
        return inp_tensor*mask_tensor.unsqueeze(-1).expand_as(inp_tensor)
