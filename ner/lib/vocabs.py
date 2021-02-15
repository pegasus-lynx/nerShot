from pathlib import Path
import json
import os
from typing import List, Union
from tqdm import tqdm
from collections import Counter

from nlcodec import Type, learn_vocab, load_scheme, term_freq, Reseved
from .misc import Filepath, FileReader, FileWriter, get_now, log

class Token(object):
    def __init__(self, name, freq:int=0, idx:int=-1, level:int=2, kids=None):
        self.name = name
        self.freq = freq,
        self.idx = idx
        self.level = level
        self.kids = kids

    def format(self):
        cols = [idx, name, level, freq]
        if self.kids is not None:
            kids = ' '.join(self.kids)
            cols.append(kids)
        return '\t'.join(cols)

class Vocabs(object):
    def __init__(self, add_reserved:str=None):
        self.tokens = set()
        self.token2id = dict()
        self.id2pos = dict()
        self.table = []
        self.max_index = 0
        if add_reserved is not None:
            self.add_reserved(add_reserved)

    def __len__(self):
        return len(self.table)

    def __iter__(self):
        curr_index = 0
        while curr_index < self.max_index:
            yield self.table[curr_index]
            curr_index += 1

    @classmethod
    def save(cls, vocab, save_file):
        with open(save_file, 'w') as fw:
            for token in vocab:
                fw.write(f'{token.format()}\n')

    def index(self, token:str):
        if word not in self.tokens:
            return -1
        return self.token2id[token]

    def token(self, index:int):
        if index not in self.id2pos.keys():
            return None
        pos = self.id2pos[index]
        return self.table[pos]

    def name(self, index:int):
        token = self.token(index)
        return token.name

    def append(self, token, force_index=None):
        if isinstance(token, Token):
            self._add_token(token, force_index=force_index)
        if isinstance(token, str):
            self._add_name(token, force_index=force_index)
        return

    def _update_token(self, token:Token):
        if token.name not in self.tokens:
            return
        pos = self.id2pos[token.idx]
        self.table[pos].freq += token.freq

    def _add_token(self, token:Token, force_index=None):
        if token.name in self.tokens:
            self._update_token(token)
            return
        self.tokens.add(token.name)
        token.idx = self.max_index + 1
        self.max_index += 1
        self.token2id[token.name] = token.idx
        self.table.append(token)
        self.id2pos[token.idx] = len(self.table)     

    def _add_name(self, token:str, level:int=2, force_index=None):
        token = Token(token, freq=1, level=level)
        self._add_token(token)
        
    def _sort(self):
        self.table = sorted(self.table, key= lambda x: (x.level, -x.freq))
        self._reindex()        

    def _reindex(self):
        for ix in range(len(self.table)):
            self.table[ix].idx = ix
        self.id2pos = {x:x for x in range(len(self.table))}
        self.token2id = {token.name:token.idx for token in self.table}
        self.max_index = len(self.table) - 1

    def _validate(self):
        pass

    def add_reserved(self, res_type:str=None):
        if reserved_type is None:
            return
        tokens = Reserved.all(res_type)
        for ix, name in enumerate(tokens):
            token = Token(name, freq=0, level=-1)
            self.append(token)

    def add(self, vocab):
        for token in vocab:
            self.append(token)

    @classmethod
    def load(cls, vocab_file):
        vcb = cls()
        with open(vocab_file, 'r') as fr:
            for line in fr:
                if line.startswith('#'):
                    continue
                cols = line.split('\t')
                idx, name, level, freq = cols[:4]
                kids = None
                if len(cols) > 4:
                    kids = cols[4].split(' ')
                vcb.append(Token(name, freq=freq, idx=idx, level=level, kids=kids))
        return vcb

    @classmethod
    def merge(cls, vocabs):
        vcb = cls()
        for vocab in vocabs:
            vcb.add(vocab)
        return vcb

    @classmethod
    def make(cls, corpus:List[Union[str, Path]], vocab_size:int=8000, level:str='bpe'):
        temp_file = Path('./_temp.txt')
        corp = get_unique(corpus)
        learn_vocab(inp=corp, level=level, model=temp_file, vocab_size=vocab_size)
        vcb = Vocabs.load(temp_file)
        os.remove(temp_file)
        return vcb