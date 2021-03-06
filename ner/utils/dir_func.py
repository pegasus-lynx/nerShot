import shutil
from pathlib import Path

class DirFuncs(object):

    @classmethod
    def copy_file(cls, src_file, tgt_file):
        if isinstance(src_file, str):
            src_file = Path(src_file)
        if isinstance(tgt_file, str):
            tgt_file = Path(tgt_file)
        shutil.copy(src_file, tgt_file)
        return tgt_file

    @classmethod
    def make_dir(cls, dir_path):
        if isinstance(dir_path, str):
            dir_path = Path(dir_path)
        if not dir_path.exists():
            dir_path.mkdir()
        return dir_path