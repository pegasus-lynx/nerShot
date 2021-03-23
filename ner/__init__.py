import os
import logging
from .tool.log import Logger

log = Logger(console_level=logging.INFO)

import torch

device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)
cpu_device = torch.device('cpu')

from ruamel.yaml import YAML
yaml = YAML()

from typing import List, Union, Dict
from pathlib import Path


# Defined General Types
FileType = Union[str, Path]
FileList = List[FileType]
FileDict = Dict[str,FileType]
FileAny  = Union[FileList, FileDict]