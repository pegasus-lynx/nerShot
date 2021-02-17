import torch
import torch.nn as nn

from ..layers.csim_loss import CosineSim

class CriterionFactory():

    criterion_types = [
        'nll', 'kld', 'csim'
    ]

    @staticmethod
    def create(criterion_type:str):
        criterion_type = criterion_type.lower()
        assert criterion_type in CriterionFactory.criterion_types

        if criterion_type == 'nll':
            criterion = nn.NLLLoss()
        elif criterion_type == 'kld':
            criterion = nn.KLDivLoss()
        elif criterion_type == 'csim':
            criterion = CosineSim()

        return criterion