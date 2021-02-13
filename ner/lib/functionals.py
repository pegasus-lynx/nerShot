

class Formatter(object):

    seps = [ '', ' ', ' | ', ' || ', ' ||| ' ]

    @classmethod
    def format(cls, cell, dim:int):
        if dim == 1:
            return cls.seps[dim].join(cell)
        for p, x in enumerate(cell):
            cell[p] = cls.format(cell[p], dim-1)
        return cls.seps[dim].join(cell)

    @classmethod
    def parse(text:str, dim:int):
        if dim==1:
            return text.split(cls.seps[dim])
        parts = text.split(cls.seps[dim])
        for p, part in enumerate(parts):
            parts[p] = cls.parse(part, dim-1)
        return parts

        ds_keys = dataset.lists.keys()
        ds = Dataset(ds_keys)
        for row in dataset:
            filter_out = False
            for key, val in zip(ds_keys, row):
                if key not in lens.keys():
                    continue
                if len(val) > lens[key]:
                    filter_out = True
                    break
            if filter_out:
                continue
            ds.append(row, ds_keys)
        return ds