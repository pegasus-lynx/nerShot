import torch
import torch.nn as nn

class ActivationFactory():

    activation_types = [
        'relu', 'gelu', 'celu', 'rrelu', 'elu', 'leakyrelu',
        'prelu', 'selu', 'relu6', 'silu', 'tanh', 'sigmoid', 'logsigmoid'
    ]

    @staticmethod
    def create(activation_type:str):
        act_type = activation_type.lower()
        assert act_type in ActivationFactory.activation_types

        if act_type =='relu':
            act = nn.ReLU()
        elif act_type =='gelu':
            act = nn.GELU()
        elif act_type =='celu':
            act = nn.CELU()
        elif act_type =='rrelu':
            act = nn.RReLU()
        elif act_type =='elu':
            act = nn.ELU()
        elif act_type =='leakyrelu':
            act = nn.LeakyReLU()
        elif act_type =='prelu':
            act = nn.PReLU()
        elif act_type =='selu':
            act = nn.SELU()
        elif act_type =='relu6':
            act = nn.ReLU6()
        elif act_type =='silu':
            act = nn.SiLU()
        elif act_type =='tanh':
            act = nn.Tanh()
        elif act_type =='sigmoid':
            act = nn.Sigmoid()
        elif act_type == 'logsigmoid':
            act = nn.LogSigmoid()
        else:
            raise(ValueError('Not found/supported'))
        return act