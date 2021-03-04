import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from exp import NERTaggingExperiment as NerExp
from preexp import NERPrepper as NerPrep
from models import ModelRegistry
from ner import log
from ner.file_io import load_conf
from ner.decoder import NERDecoder

import torch
import random

class DecPipe(object):

    def __init__(self, work_dir, conf_file, gpu:int=-1):
        self.work_dir = work_dir
        self.test_dir = work_dir / Path('test/')
        self.gpu = gpu
        self.config = load_conf
        self.decoder = None

    def make_decoder(self, beam_size:int=1, ensemble:int=1, use_conf:bool=True):
        if use_conf:
            tester_args = self.config.get('tester')
            assert tester_args is not None
            beam_size = tester_args.get('beam_size', 1)
            ensemble  = tester_args.get('ensemble', 5)
        
        model_name = self.config.get('model_name')
        assert model_name is not None
        assert model_name in ModelRegistry.names

        self.decoder = NERDecoder(self.work_dir, model_name, beam_size=beam_size,
                                    ensemble=ensemble, gpu=self.gpu)

    def run(self, files, suits):
        self.decoder.decode(suits)
        self.decoder.decode_files(files)

def parse_args():
    parser = argparse.ArgumentParser(prog='decoder')
    parser.add_argument('-w', '--work_dir', type=Path)
    parser.add_argument('-c', '--config_file', type=Path, nargs='?')
    parser.add_argument('-u', '--use_conf', type=bool, default=True)
    parser.add_argument('-t', '--exp_type', choices=['ner'], default='ner')
    parser.add_argument('-b', '--beam_size', type=int, default=1)
    parser.add_argument('-e', '--ensemble', type=int, default=1)
    parser.add_argument('-g', '--gpu', type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gpu >= 0:
        assert torch.cuda.is_available(), 'GPU Parameter passed but no GPU available'
        for i in range(torch.cuda.device_count()):
            log.info(f'Cuda {i}: {torch.cuda.get_device_properties(i)}')
        assert args.gpu < len(torch.cuda.device_count())
        log.info(f'Using GPU {args.gpu} : {torch.cuda.get_device_properties(args.gpu)}')

    conf_file: Path = args.config_file if args.config_file else args.work_dir / 'conf.yml'
    assert conf_file.exists(), f'NOT FOUND: {conf_file}'

    decpipe = DecPipe(work_dir, conf_file, gpu=args.gpu)
    decpipe.make_decoder(beam_size=args.beam_size, ensemble=args.ensemble, use_conf=args.use_conf) 
    decoder.run()

if __name__ == '__main__':
    main()