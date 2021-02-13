import json
from tqdm import tqdm
from .parser_base import AbstractParser
from typing import List, Dict, Union
from ner import FileList
from .dataset import Dataset
from ..functionals import Functionals as Fn

class CoNLL03Parser(AbstractParser):
    @classmethod
    def parse(cls, filepaths:FileList, keys:List[str]=['keys', 'tagseqs'], filter:bool=False):
        data = Dataset(keys)
        seq, tagseq = [], []
        for filepath in filepaths:
            with open(filepath, 'r') as fr:
                for line in fr:
                    if 'DOCSTART' in line:
                        continue
                    line = line.strip()
                    if len(line) == 0:
                        if len(seq) != 0:
                            data.append([seq, tagseq])
                            seq, tagseq = [], []
                        continue
                    try:
                        token, _, _, nertag = line.split()
                        seq.append(token)
                        tagseq.append(nertag)
                    except Exception:
                        print(line)
                        raise(e)
        return data

    @classmethod
    def write(cls, dataset:Dataset, files:FileAny):
        keys = dataset.keys
        if isinstance(files, list):
            files = { key:filepath for key,filepath in zip(keys, files)}
        fws = dict()
        for key in keys:
            meta = dict(total=len(dataset), key=key, dim=1)
            fw = open(files[key], 'w')
            # fw.write(f'#{json.dumps(meta)}\n')
            fws[key] = fw
        for row in dataset:
            for key, cell in zip(keys, row):
                fws[key].write(f'{Fn.format(cell, dim=1)}\n')

    @classmethod
    def read(cls, files:FileAny, keys:List[str]=['seqs', 'tagseqs']):
        dataset = Dataset(keys)
        if isinstance(files, list):
            files = { key:filepath for key,filepath in zip(keys, files)}
        # frs, metas = dict(), dict()
        cols = dict()
        for key in keys:
            fr = open(files[key],'r')
            cols[key] = fr.readlines()
            # line = fr.readline().strip()
            # frs[key] = fr
            # if line.startswith('#'):
            #     meta = json.load(line[1:])
            #     metas[key] = meta
        nlines = min([len(cols[key]) for key in keys])
        curr = 0
        while curr < nlines:
            row = [cols[key][curr] for key in keys]
            row = [Fn.unformat(x,dim=1) for x in row]
            curr += 1
            dataset.append(row)

    @classmethod
    def filter(cls):
        pass