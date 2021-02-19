import argparse
from pathlib import Path
from preexp import NERPrepper

def parse_args():
    parser = argparse.ArgumentParser(prog='prep')
    parser.add_argument('-w', '--work_dir', type=Path)
    parser.add_argument('-t', '--prep_type', type=str, choices=['ner'], default='ner')
    parser.add_argument('-c', '--conf_file', type=Path, nargs='?')
    return parser.parse_args()

def main():
    args = parse_args()
    conf_file: Path = args.conf_file if args.conf_file else args.work_dir / 'prep.yml'
    assert conf_file.exists()
    prepper = NERPrepper(args.work_dir, config=conf_file)
    return prepper.pre_process()

if __name__ == '__main__':
    main()