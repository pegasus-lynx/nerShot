from pathlib import Path

def get_unique(filepaths):
    flat = set()
    for line in FileReader.get_liness(filepaths):
        line = line.strip()
        flat.add(line)
    return flat

class FileReader:
    def __init__(self, path, text=True):
        self.path = path
        self.mode = 'rt' if text else 'rb'
        self.fd = None

    def __enter__(self):
        if 'b' in self.mode:
            self.fd = self.path.open(self.mode)
        else:
            self.fd = self.path.open(self.mode, encoding=self.encoding)
        return self.fd

    def __exit__(self):
        self.fd.close()

    @classmethod
    def get_lines(cls, path, col=0, delim='\t', line_mapper=None, newline_fix=True):
        with cls(path) as inp:
            if newline_fix and delim != '\r':
                inp = (line.replace('\r', '') for line in inp)
            if col >= 0:
                inp = (line.split(delim)[col].strip() for line in inp)
            if line_mapper:
                inp = (line_mapper(line) for line in inp)
            yield from inp

    @classmethod
    def get_liness(cls, paths, **kwargs):
        for path in paths:
            yield from cls.get_lines(path, **kwargs)

class FileWriter:
    def __init__(self, path, text=True, append=False):
        self.path = path
        self.mode =  'a' if append else 'w' + 't' if text else 'b'
        self.fd = self.path.open(self.mode)
   
    def write(self, text):
        self.fd.write(text)
    
    def writeline(self, text):
        self.fd.write(text)
        self.fd.write('\n')

    def writelines(self, text_list):
        for text in text_list:
            self.writeline(text)

    def newline(self, count:int=1):
        for i in range(count):
            self.write('\n')

    def close(self):
        self.fd.close()

class MetaWriter(FileWriter):
    def __init__(self, path, text=True):
        super(MetaWriter, self).__init__(path, text=text)
        self.indent = 0

    def add_indent(self):
        self.write(f'{"    "*self.indent}')

    def textline(self, text):
        self.add_indent()
        self.writeline(text)

    def textlines(self, texts):
        for text in texts:
            self.textline(text)

    def heading(self, text):
        self.textline(text)
        self.time()
        self.dashline()
        self.newline()

    def time(self):
        def _get_now():
            datetime_str = datetime.now().isoformat().replace('T', ' ')
            pos = datetime_str.index('.')
            return datetime_str[:pos]

        self.textline(f'Time {_get_now()}')

    def dashline(self, txt='-', length=50):
        self.writeline(f'{txt*length}')

    def sectionstart(self, text):
        self.textline(text)
        self.dashline()
        self.indent += 1

    def sectionclose(self):
        self.dashline()
        self.newline()
        self.indent -= 1

    def section(self, heading:str, lines:List):
        self.sectionstart(heading)
        self.textlines(lines)
        self.sectionclose()
