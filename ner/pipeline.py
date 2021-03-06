import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from ner.exp import NERTaggingExperiment as NerExp
from ner.preexp import NERPrepper as NerPrep
from ner.models import ModelRegistry
from ner import log
from ner.decoder import NERDecoder

import torch
import random


class Pipeline(object):

    def __init__(self, exp):
        self.exp = exp
        self.config = exp.config

    def pre_checks(self):
        assert self.exp.work_dir.exists()
        conf = self.exp.config
        assert conf.get('trainer') is not None
        assert conf.get('model_args') is not None
        assert conf.get('model_name') is not None
        assert conf.get('model_name') in ModelRegistry.names

    def run(self):
        self.pre_checks()
        self.exp.load()
        # if not self.exp.has_prepared():
            # self.preprocess()
        self.exp.train_model()
        self.run_tests()

    def run_tests(self):
        tester_args = self.config.get('tester')
        decoder = NERDecoder(self.exp.work_dir, self.config.get('model_name'),
                            beam_size=tester_args.get('beam_size'), 
                            ensemble=tester_args.get('ensemble'), gpu=self.exp.gpu)
        decoder.load()
        if tester_args.get('files'):
            decoder.decode_files(tester_args.get('files'))
        if tester_args.get('suits'):
            decoder.decode(tester_args.get('suits'))

    def preprocess(self):
        work_dir = self.exp.work_dir
        prep_file = work_dir / Path('prep.yml')
        prepper = NerPrep(work_dir, config=prep_file)
        prepper.pre_process()

def parse_args():
    parser = argparse.ArgumentParser(prog='pipe')
    parser.add_argument('-w', '--work_dir', type=Path)
    parser.add_argument('-c', '--config_file', type=Path, nargs='?')
    parser.add_argument('-p', '--prep_file', type=Path, nargs='?')
    parser.add_argument('-t', '--exp_type', choices=['ner'], default='ner')
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

    exp = NerExp(args.work_dir, config=conf_file)

    pipe = Pipeline(exp)
    pipe.run()

if __name__ == '__main__':
    main()