from abc import ABC, abstractmethod, abstractstaticmethod, abstractclassmethod
from pathlib import Path
from typing import Dict, Union

class AbstractParser(ABC):
    @abstractclassmethod
    def parse():
        pass

    @abstractclassmethod
    def write():
        pass

    @abstractclassmethod
    def read():
        pass

    @abstractclassmethod
    def filter():
        pass