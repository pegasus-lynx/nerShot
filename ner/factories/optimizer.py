import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR


class OptimizerFactory():
    """OptimizerFactory contains wrappers to create various optimizers."""
    @staticmethod
    def create(optim_args, model):
        opt_name, args = optim_args
        if opt_name == 'sgd':
            optimizer = optim.SGD(list(model.parameters()), lr=args.get('lr'), momentum=args.get('momentum'))
        elif opt_name == 'adam':
            optimizer = optim.Adam(list(model.parameters()), lr=args.get('lr'), betas=(0.9, 0.999))
        else:
            raise ValueError('Unknown optimizer, must be one of "sgd"/"adam".')
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + args.get('lr_decay', 0.05) * epoch))
        return optimizer, scheduler