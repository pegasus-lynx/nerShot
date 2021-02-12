import torch.nn as nn

class AbstractLayer(nn.Module):
    """Abstract base class for all type of layers."""
    def __init__(self, gpu):
        super(LayerBase, self).__init__()
        self.gpu = gpu

    def set_device(self, tensor):
        if self.is_cuda():
            return tensor.cuda(device=self.gpu)
        else:
            return tensor.cpu()

    def apply_mask(self, input_tensor, mask_tensor):
        input_tensor = self.set_device(input_tensor)
        mask_tensor = self.set_device(mask_tensor)
        return input_tensor*mask_tensor.unsqueeze(-1).expand_as(input_tensor)
