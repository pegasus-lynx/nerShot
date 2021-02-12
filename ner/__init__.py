import os
import logging
from rtg.tool.log import Logger

log = Logger(console_level=logging.INFO)

import torch

device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)
cpu_device = torch.device('cpu')

from ruamel.yaml import YAML
yaml = YAML()
