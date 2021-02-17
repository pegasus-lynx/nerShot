from pathlib import Path
import json
import os
from typing import List, Union
from tqdm import tqdm
from collections import Counter

from nlcodec import Type, learn_vocab, load_scheme, term_freq, Reseved
from .misc import Filepath, FileReader, FileWriter, get_now, log
from ..tool.file_io import FileReader

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

        self.reserved = dict()
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
        if res_type is None:
            return
        
        tokens = Reserved.all(res_type)
        if len(tokens)==0:
            return

        for ix, name in enumerate(tokens):
            token = Token(name, freq=0, level=-1)
            self.append(token)
        
        self.reserved['pad'] = Reserved.PAD_TOK[0]
        if res_type in ['word', 'char']:
            keys = ['pad', 'oov', 'sos', 'eos', 'brk', 'spc']
            for key, tok in zip(keys, Reserved.TAG_TOKS):
                self.reserved[key] = tok
        elif res_type in 'ner':
            self.reserved['pad'] = Reserved.PAD_TOK[0]

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

    def reserved_idx(self, res_type:str):
        if res_type in self.reserved.keys():
            return self.index(self.reserved[res_type])
        return None

    @classmethod
    def lowered(cls, vocab):
        vmap = [-1] * len(vocab)
        rvmap = dict()
        lvcb = Vocabs()
        for i, x in enumerate(vocab):
            lx = x.name.lower()
            if lvcb.index(lx) is not None:
                vmap[i] = lvcb.index(lx)
            else:
                index = len(lvcb)
                lvcb.append(lx)
                vmap[i] = index        
        for i, v in enumerate(vmap):
            if v not in rvmap.keys():
                rvmap[v] = []
            rvmap[v].append(i)
        return lvcb, rvmap

class Embeddings(object):
    
    def __init__(self):
        self.emb_file = emb_file if type(emb_file) == Path else Path(emb_file)
        self.emb_dim = emb_dim
        self.mat = None

        if vocab is not None:
            self.vocab = vocab
            self.found = [False] * len(vocab)   
            self.mat = np.zeros((len(vocab), self.emb_dim), dtype=np.float64)

        # Models for generating embeddings : FastText, Mimick
        self.ft_model = None
        self.mk_model = None

    @property
    def vocabulary(self):
        return self.vocab 

    @vocabulary.setter
    def vocabulary(self, vocab):
        if type(vocab) == Path or type(vocab) == str:
            vocab = Vocabs.load(vocab)
        self.vocab = vocab
        self.found = [False] * len(vocab)   
        self.mat = np.zeros((len(vocab), self.emb_dim), dtype=np.float64)

    @property
    def oovs(self):
        if self.found is None or self.vocab is None:
            return []
        return [x for i, x in enumerate(self.vocab) if not self.found[i]]

    @classmethod
    def load(cls, in_file: Union[Path or str], is_expanded:bool=False):
        if not is_expanded:
            with np.load(in_file) as data:
                return data["embeddings"]
        emb = cls(in_file)
        emb._build()
        return emb  

    @classmethod
    def save(cls, emb, out_file:Union[Path or str], save_expanded:bool=False):
        if not save_expanded:
            np.savez_compressed(out_file, embeddings=emb.mat)
            return
        
        assert emb.mat is not None
        assert len(vocab) == emb.mat.shape[0]
        
        with open(emb_file, "w") as ef:
            ef.write(f'{emb.mat.shape[0]} {emb.mat.shape[1]}\n')
            for ix, word in enumerate(emb.vocab):
                ef.write(f'{word} {" ".join(str(x) for x in emb.mat[ix])}\n')
        return None

    def normalize(self, normalize_to:float = 1.0):
        if self.mat is None:
            return
        size = self.mat.shape[0]
        for ix in range(size):
            norm = np.linalg.norm(self.mat[ix])
            if norm == 0:
                norm = 1
            self.mat[ix] = self.mat[ix] / norm

    def _build(self):
        fs = open(self.emb_file, 'r')
        first = True
        size, emb_dim, pos = 0, 0, 0
        for line in fs:
            if first:
                first = False
                size, emb_dim = [int(x) for x in line.strip().split()]
                self.vocab = Vocabs()
                self.found = [False] * size
                self.mat = np.zeros((size, emb_dim), dtype=np.float64)      
            tokens = line.strip().split()
            word, vec = tokens[0], tokens[1:]
            self.vocab.append(word)
            self.found[pos] = True
            self.mat[pos] = vec
            pos += 1

    def make(self, lowered=True):
        assert self.vocab is not None
        assert self.emb_file.exists()

        if not self.found or self.mat is None:
            self.found = [False] * len(self.vocab)
            self.mat = np.zeros((len(self.vocab), self.emb_dim), dtype=np.float64)

        if lowered:
            lvocab, lmap = Vocabs.lowered(self.vocab)
            lowered_found = set()

        print(f'Scanning {self.emb_file} ... ')
        with open(self.emb_file, 'r') as ef:
            for line in tqdm(ef):
                tokens = line.strip().split()
                if len(tokens) != self.emb_dim + 1:
                    continue
                word, vec = tokens[0], tokens[1:]
                pos = self.vocab.index(word)
                if pos:
                    if not self.found[pos]:
                        self.found[pos] = True
                        self.mat[pos] = vec
                        if lowered and pos in lowered_found:
                            lowered_found.remove(pos)
                if lowered:
                    if lvocab.index(word) is not None:
                        lpos = lvocab.index(word)
                        for pos in lmap[lpos]:
                            if not self.found[pos] and pos not in lowered_found:
                                lowered_found.add(pos)
                                self.mat[pos] = vec
        print(f'Found tokens : {sum(self.found)}' )
        if lowered:
            print(f'Found on lowering : {len(lowered_found)}')
            for pos in lowered_found:
                self.found[pos] = True
        print(f'Unfound tokens : {len(self.found) - sum(self.found)}')

    @classmethod
    def train(cls, dataset_files, emb_dim, min_freq=10, epochs=5):
        from gensim.models.fasttext import FastText
        model = FastText(sg=1, size=emb_dim, min_count=min_freq)
        with FileReader.get_liness(dataset_files) as reader:
            model.build_vocab(sentences = reader)
            model.train(sentences=reader, total_examples=model.corpus_count, epochs=epochs)
        return model    

    @classmethod
    def generate(cls, model_path, vocab):
        import fasttext
        emb = Embeddings('none', vocab)
        ft = fasttext.load_model(str(model_path))
        for pos, token in enumerate(vocab):
            self.mat[pos] = ft.get_word_vector(token.name)
            self.found[pos] = True
        return emb

    def add_model(self, model, model_type:Union['fasttext' or 'mimick'] = 'fasttext'):
        if model_type == 'fasttext':
            self.ft_model = model
        elif model_type == 'mimick':
            self.mk_model = model