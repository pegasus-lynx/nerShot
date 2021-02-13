from tqdm import tqdm
from .parser_base import AbstractParser
from typing import List, Dict, Union
from ner import FileType, FileAny
from ..functionals import Functionals as Fn

class FIREParser(AbstractParser):

    tagmap = {
        'other' : 'O',
        'location' : 'LOC',
        'organization': 'ORG',
        'name': 'NAME',
        'things': "TNG",
        'number': 'NUM',
        'datenum': 'DATE',
        'occupation': 'OCC',
        'event': 'EVE'        
    }

    @classmethod
    def parse(cls, filepaths:Dict[str, File], keys:List[str]=['seqs', 'tagseqs'], filter:bool=False):
        dataset = Dataset(keys)
        seq, tagseq = [], []
        count = 0
        skipline = False
        for filepath in filepaths:
            with open(filepath, "r+") as fs:
                for line in fs:
                    line = line.strip()
                    count += 1
                    if line == 'newline':
                        if not skipline:
                            dataset.append([seq, tagseq])
                        seq, tagseq = [], []
                        skipline = False
                        continue
                    try:
                        token, nertag = line.split('\t')
                        if filter and not cls.filter(token):
                            skipline = True
                        seq.append(token)
                        tagseq.append(cls.tagmap[nertag])
                    except Exception:
                        token = line.strip()
                        if filter and not cls.filter(token):
                            skipline = True
                        seq.append(token)
                        tagseq.append('O')
        dataset = cls.iobmark(dataset)
        return dataset

    @classmethod
    def iobmark(cls, dataset):
        for ix, tags in enumerate(dataset.cols['tagseqs']):
            iob_tags = []
            start = ""
            for tag in tags:
                if tag == 'O':
                    start = ""
                    iob_tags.append(tag)
                elif start!=tag:
                    iob_tags.append("B-{}".format(tag))
                    start = tag
                else:
                    iob_tags.append("I-{}".format(tag))            
            dataset.cols['tagseqs'][ix] = iob_tags
        return dataset

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