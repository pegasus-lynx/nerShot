import random
import math
import json
from pathlib import Path
from typing import Dict
from collections import Counter, OrderedDict
from typing import Iterable, Iterator, List, Tuple, Union
from .functionals import Formatter as Fr
from .functionals import Indexer as Ir
from .functionals import Padder as Pd
from .functionals import Converter as Cr

class Dataset(object):

    def __init__(self, keys:List[str], dims=None):
        self.cols = OrderedDict()
        self.keys = keys
        self.length = 0
        for key in keys:
            self.cols[key] = []

        if dims is not None:
            self._set_dims(dims)

    def __len__(self):
        return len(self.cols[self.keys[0]])

    def __iter__(self):
        curr_index = 0
        while curr_index < len(self):
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
        if row is None:
            try:
                row = self._get_row(index)
            except Exception:
                return ''
        parts = [Fr.format(cell, self.dims[key]-1, modifier=str) for key, cell in zip(self.keys, row)]
        return '\t'.join(parts)

    def parse(self, line):
        cols = line.strip().split('\t')
        row = []
        for key, col in zip(self.keys, cols):
            dim = self.dims[key]
            row.append(Fr.parse(col, dim-1))
        return row

    def append(self, data_row):
        if isinstance(data_row, dict):
            if not self._validate_keys(data_row.keys(), ordered=False):
                return
            data_row = [data_row[key] for key in self.keys]
        for key, data_point in zip(self.keys, data_row):
            self.cols[key].append(data_point)

    def add(self, dataset):
        if not self._validate_keys(dataset.keys):
            print('Keys don\'t match')
            return
        for row in dataset:
            self.append(row)

    def read(self, filenames):
        for key in filenames.keys():
            assert key in self.keys        
        for key, filename in filenames.items():
            fr = open(filename, 'r')
            lines = fr.readlines()
            for line in lines:
                line = line.strip()
                self.cols[key].append(line.split())
            self.length = len(self.cols[key])
            fr.close()

    def add_bos(self, key, vocab, add_type:str='token'):
        # Supports addition for 1D lists only
        assert key in self.keys
        assert 'sos' in vocab.reserved.keys()
        value = self.reserved['sos']
        if add_type == 'index':
            value = self.reserved_idx('sos')
        for p, cell in enumerate(self.cols[key]):
            cell.insert(0, value)
            self.cols[key][p] = cell

    def add_eos(self, key, vocab, add_type:str='token'):
        # Supports addition for 1D lists only
        assert key in self.keys
        assert 'eos' in vocab.reserved.keys()
        value = self.reserved['eos']
        if add_type == 'index':
            value = self.reserved_idx('eos')
        for p, cell in enumerate(self.cols[key]):
            cell.append(value)
            self.cols[key][p] = cell

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
        meta = json.loads(line)
        dataset = Dataset(meta.get('keys'), dims=meta.get('dims'))
        for line in fr:
            row = dataset.parse(line)
            dataset.append(row)
        fr.close()
        return dataset

    @classmethod
    def save(cls, dataset, filepath):
        with open(filepath, 'w') as fw:
            meta = dataset._get_meta()
            fw.write(f'#{json.dumps(meta)}\n')
            for row in dataset:
                fw.write(f'{dataset.format(row=row)}\n')


class Batch(object):
    
    def __init__(self, mats, keys, dims, schema, gpu:int=-1):
        self.schema = schema
        self.mats = dict()
        self.dims = dims
        for key, mat in zip(keys, mats):
            self.mats[key] = self.process(mat, gpu)

    @property
    def forward(self):
        return [self.mats[key] for key in self.schema.forward]

    @property
    def labels(self):
        return [self.mats[key] for key in self.schema.labels]

    @property
    def get_loss(self):
        return [self.mats[key] for key in self.schema.get_loss]

    @property
    def predict(self):
        return [self.mats[key] for key in self.schema.predict]

    def process(self, mat, gpu:int=-1):
        mat = Cr.list2tensor(mat, gpu)
        return mat


class BatchIterable(object):
    
    def __init__(self, dataset:Dataset, schema=None, batch_size:int=32, shuffle:bool=True, 
                        indexed:bool=False, gpu:int=-1):
        self.batch_size = batch_size

        self.keys = dataset.keys
        self.dataset = dataset

        self.shuffled = False
        self.permuation = []
        if shuffle: 
            self.shuffled = True
            self.permuation = self._permute()

        self.gpu = gpu
        self.schema = schema
        self.indexed = indexed
        self.vocabs = dict()

    def __iter__(self):
        if not self.indexed:
            raise ValueError('Batch not indexed. Batch must be indexed before iterating')
        if self.schema is None:
            raise ValueError('Schema is not set')

        mats = [[] for x in range(len(self.keys))]
        cur_row = 0
        for p in self.permuation:
            if cur_row == self.batch_size:
                yield Batch(mats, self.keys, self.dataset.dims, self.schema, gpu=self.gpu)
                mats = [[] for x in range(len(self.keys))]
                cur_row = 0
            for x, key in enumerate(self.keys):
                mats[x].append(self.dataset.cols[key][p])
            cur_row += 1
        yield Batch(mats, self.keys, self.dataset.dims, self.schema, gpu=self.gpu)

    def __len__(self):
        return int(math.ceil(len(self.dataset) / self.batch_size))

    def set_schema(self, schema):
        keys = schema.get_keys()
        for key in keys:
            if key not in self.keys:
                print('Cannot set schema. The key mentioned in schema is not present in the dataset')
                return
        self.schema = schema    

    def set_indexer(self, dataset_key:str, vocab):
        if dataset_key not in self.keys:
            return
        self.vocabs[dataset_key] = vocab

    def add_bos_eos(self, add_bos:bool=True, add_eos:bool=True, keys:List[str]=None):
        if keys is None:
            keys = self.keys
        if add_bos:
            for key in keys:
                self.dataset.add_bos(key, self.vocabs[key])
        if add_eos:
            for key in keys:
                self.dataset.add_eos(key, self.vocabs[key])

    def index(self):
        for key in self.keys:
            assert key in self.vocabs.keys()
        if self.indexed:
            return
        keys = self.keys
        dims = self.dataset.dims
        for p, row in enumerate(self.dataset):
            for key, cell in zip(keys, row):
                self.dataset.cols[key][p] = Ir.index(cell, dims[key]-1, self.vocabs[key])
        return

    def pad(self, padshapes:Dict[str,int], pad_idx:int=0):
        for key in self.keys:
            assert key in padshapes.keys()
            assert len(padshapes[key]) == self.dataset.dims[key] - 1
        dims = self.dataset.dims
        for p, row in enumerate(self.dataset):
            for key, cell in zip(self.keys, row):
                self.dataset.cols[key][p] = Pd.pad(cell, dims[key]-1, pad_idx, padshapes[key])
        
    def _permute(self):
        permutation = random.sample([x for x in range(len(self.dataset))], len(self.dataset))
        return permutation 