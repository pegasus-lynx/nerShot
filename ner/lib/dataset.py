import random
import json
from pathlib import Path
from collections import Counter, OrderedDict
from typing import Iterable, Iterator, List, Tuple, Union
from .functionals import Formatter as Fr

class Dataset(object):

    def __init__(self, keys:List[str], dims=None):
        self.cols = OrderedDict()
        self.keys = keys
        self.length = 0
        for key in keys:
            self.cols[key] = []
        self._set_dims(dims)

        self.is_shuffled = False
        self.permutation = None

    def __len__(self):
        return self.length

    def __iter__(self):
        curr_index = 0
        while curr_index < self.length:
            yield self._get_row(curr_index)
            curr_index += 1

    def _get_row(self, index):
        assert index >=0 and index < len(self)
        return [self.cols[key][index] for key in self.keys] 

    def _validate_keys(self, keys, ordered:bool=True):
        try:
            assert len(keys) == len(self.keys)
            if ordered:
                for p in range(len(keys)):
                    assert keys[p] == self.keys[p]
            else:
                for key in keys:
                    assert key in self.keys
            return True
        except Exception:
            return False

    def _set_dims(self, dims):
        self.dims = dict()
        if isinstance(dims, dict):
            self.dims = dims
        if isinstance(dims, list):
            for key, dim in zip(self.keys, dims):
                self.dims[key] = dim

    def _get_meta(self):
        return dict(total=len(self), keys=self.keys, dims=self.dims)

    def _set_meta(self, meta):
        self.dims = meta.dims

    def format(self, row=None, index:int=-1):
        if row is None
            try:
                row = self._get_row(index)
            except Exception:
                return ''
        parts = [Fr.format(cell, self.dims[key]) for key, cell in zip(self.keys, row)]
        return '\t'.join(parts)

    def parse(self, line):
        cols = line.strip().split('\t')
        row = []
        for key, col in zip(self.keys, cols):
            dim = self.dims[key]
            row.append(Fr.parse(col, dim))
        return row

    def append(self, data_row):
        if isinstance(data_row, dict):
            if not self._validate_keys(data_row.keys(), ordered=False):
                return
            data_row = [data_row[key] for key in self.keys]
        for key, data_point in zip(self.keys, data_row):
            self.cols[key].append(data_point)

    def shuffle(self):
        self.is_shuffled = True
        self.permutation = random.sample([x for x in range(len(self))], len(self))

    def unshuffle(self):
        self.is_shuffled = False
        self.permutation = [x for x in range(len(self))]

    def add(self, dataset):
        if not self._validate_keys(dataset.keys)
            print('Keys don\'t match')
            return
        for row in dataset:
            self.append(row)

    @classmethod
    def merge(cls, keys, datasets):
        dataset = Dataset(keys)
        for ds in datasets:
            dataset.add(ds)
        return ds

    @classmethod
    def load(cls, data_file):
        fr = open(data_file, 'r')
        line = fr.readline()
        line = line.strip()[1:]
        meta = json.load(line)
        dataset = Dataset(meta.keys, meta.dims)
        for line in fr:
            row = cls.parse(line)
            dataset.append(row)
        return dataset

    @classmethod
    def save(cls, dataset, filepath):
        with open(filepath, 'w') as fw:
            meta = dataset._get_meta()
            fw.write(f'#{json.dumps(meta)}\n')
            for row in dataset:
                fw.write(f'{cls.format(row=row)}\n')

    # def minibatches(self, size):
    #     ''' Returns batches from the dataset '''
    #     if self.shuffled is None:
    #         self.shuffled = [ x for x in range(len(self))]
    #     batch = [[] for x in range(len(self.lists))]
    #     curr = 0
    #     for ix in self.shuffled:
    #         if curr == size:
    #             curr = 0
    #             yield tuple(batch)
    #             batch = [[] for x in range(len(self.lists))]
    #         for p, key in enumerate(self.lists.keys()):
    #             batch[p].append(self.lists[key][ix])
    #         curr += 1
