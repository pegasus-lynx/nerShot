import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from ner import log

class OptimizerFactory():
    """OptimizerFactory contains wrappers to create various optimizers."""
    @staticmethod
    def create(optim_args, model):
        opt_name, args = optim_args
        opt_name = opt_name.lower()

        log.info(f'Optimizer : {opt_name.upper()}')
        log.info(f'Optimizer Args : {args}')

        assert args.get('lr')

        if opt_name == 'sgd':
            optimizer = optim.SGD(list(model.parameters()), lr=args.get('lr'), momentum=args.get('momentum'))
        elif opt_name == 'adam':
            optimizer = optim.Adam(list(model.parameters()), lr=args.get('lr'), betas=(0.9, 0.999))
        else:
            raise ValueError('Unknown optimizer, must be one of "sgd"/"adam".')
        
        lr_decay = args.get('lr_decay', 0.05)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + lr_decay * epoch))
        return optimizer, scheduler