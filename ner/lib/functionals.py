import copy
from typing import List, Dict
from lib.tokenizer import Reserved

class Formatter(object):

    seps = [ '', ' ', ' | ', ' || ', ' ||| ' ]

    @classmethod
    def format(cls, cell, dim:int, modifier=None):
        if dim == 1:
            cell = list(map(modifier, cell))
            return cls.seps[dim].join(cell)
        for p, x in enumerate(cell):
            cell[p] = cls.format(cell[p], dim-1, modifier=modifier)
        return cls.seps[dim].join(cell)

    @classmethod
    def parse(text:str, dim:int, modifier=None):
        if dim==1:
            splits = text.split(cls.seps[dim])
            if modifier:
                splits = list(map(modifier, splits))
            return splits
        parts = text.split(cls.seps[dim])
        for p, part in enumerate(parts):
            parts[p] = cls.parse(part, dim-1, modifier=modifier)
        return parts

class Indexer(object):

    @classmethod
    def index(cls, cell, dim:int, vocab):
        if dim == 1:
            icell = []
            for x in cell:
                index = vocab.index(x)
                if index is None:
                    index = vocab.index(Reserved.OOV_TOK[0])
                icell.append(index)
            return icell

        for p, x in enumerate(cell):
            cell[p] = cls.index(x,dim-1,vocab)
        return cell

    @classmethod
    def deindex(cls, cell, dim:int, vocab):
        if dim == 1:
            icell = []
            for x in cell:
                token = vocab.name(x)
                icell.append(token)
            return icell
            
        for p, x in enumerate(cell):
            cell[p] = cls.index(x,dim-1,vocab)
        return cell

class Converter(object):

    @classmethod
    def list2numpy(cls, mat):
        return np.asarray(mat)

    @classmethod
    def list2tensor(cls, mat, gpu:int=-1, requires_grad:bool=False):
        nmat = cls.list2numpy(mat)
        tmat = torch.from_numpy(mat).long()
        if requires_grad:
            tmat.requires_grad_ = True
        if gpu >=0:
            tmat = tmat.cuda(gpu)
        return tmat
        
    @classmethod
    def tensor2numpy(cls, tensor):
        return tensor.cpu().detach().numpy()
    
class Padder(object):

    @classmethod
    def pad(cls, mat, dim:int, pad_idx:int, padshape:List[int]=None):
        if padshape is None:
            padshape = cls.get_padshape(mat, dim)

        if dim == 1:
            if padshape[0] <= len(mat):
                mat = mat[:padshape[0]]
            else:
                mat.extend([pad_idx for _ in range(padshape[0]-len(mat))])
            return mat
        
        for p, x in enumerate(mat):
            mat[p] = cls.pad(x,dim-1, pad_idx, padshape=padshape[1:])
        if len(mat) < padshape[0]:
            mat.append(cls._make_empty(dim-1, padshape[1:], pad_idx))
        else:
            mat = mat[:padshape[0]]
        return mat

    @classmethod
    def unpad(cls, mat, dim:int, pad_idx:int):
        mask = cls.make_mask(mat, dim, pad_idx)
        unpadded = []
        if dim == 1:
            for flag, x in zip(mask, mat):
                if flag:
                    unpadded.append(x)
            if len(unpadded) == 0:
                return None
            return unpadded

        for p, x in enumerate(mat):
            submat = cls.unpad(x, dim-1, pad_idx)
            if submat is not None:
                unpadded.append(submat)
        return unpadded

    @classmethod
    def get_padshape(cls, mat, dim:int):
        if dim==1:
            return [len(mat)]
        
        padshape = [0] * dim
        padshape[0] = len(mat)
        max_shape = [0] * (dim-1)
        for p, x in enumerate(mat):
            curr_shape = cls.get_padshape(x, dim-1)
            max_shape = [max(max_shape[i], curr_shape[i]) for i in range(dim-1)]
        for i in range(dim-1):
            padshape[i+1] = max_shape[i]
        return padshape

    @classmethod
    def make_mask(cls, mat, dim:int, pad_idx:int, inverse:bool=False):
        if dim == 1:
            mask_mat = [0 if x == pad_idx else 1 for x in mat]
            if inverse:
                mask_mat = [1-x for x in mask_mat]
            return mask_mat
        mask_mat = []
        for p, x in enumerate(mat):
            mask_mat.append(cls.make_mask(x, dim-1, pad_idx, inverse=inverse))
        return mask_mat

    @classmethod
    def _make_empty(cls, dim, shape, pad_idx):
        if dim == 1:
            return [pad_idx] * shape[0]
        mat = []
        for p in range(shape[0]):
            mat.append(cls._make_empty(dim-1, shape[1:], pad_idx))
        return mat
