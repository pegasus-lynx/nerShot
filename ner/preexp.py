from pathlib import Path
from nlcodec import load_scheme
from ner.lib.vocabs import Vocabs, Embeddings
from ner.lib.tokenizer import Reserved
from ner.lib.dataset import Dataset, BatchIterable
from ner.tool.file_io import load_conf, FileReader
from ner import log, FileList, FileAny, FileType

class BasePrepper(object):
    def __init__(self, work_dir, config=None):
        if type(work_dir) is str:
            work_dir = Path(work_dir)

        log.info(f'Initializing data preparation. Directory : {work_dir}')
        self.work_dir = work_dir
        self.log_dir = work_dir / Path('log')
        self.log_file = self.log_dir / Path('prep.log')
        self.data_dir = self.work_dir / Path('data/')
        self._config_file = work_dir / Path('prep.yml')

        if isinstance(config, str) or isinstance(config, Path):
            config = load_conf(config)
        self.config = config if config else load_conf(self._config_file)
        
        self._prepared_flag = self.data_dir / Path('_PREPARED')
        self.make_dirs()

    def make_dirs(self):
        if not self.log_dir.exists():
            self.log_dir.mkdir()
        if not self.data_dir.exists():
            self.data_dir.mkdir()

    def has_prepared(self):
        return self._prepared_flag.exists()

class NERPrepper(BasePrepper):
    def __init__(self, work_dir, config=None):

        super(NERPrepper, self).__init__(work_dir, config)

        data_dir = self.data_dir
        # Vocab
        self.vocab_files = {
            'seqs' : data_dir / Path('word.vocab'),
            'subwordseqss' : data_dir / Path('subword.vocab'),
            'charseqs' : data_dir / Path('char.vocab'),
            'tagseqs' : data_dir / Path('tag.vocab')
        }

        # Embedding file
        self._word_emb_file = data_dir / Path('word.emb.npz')
        self._subword_emb_file = data_dir / Path('subword.emb.npz')

        # Dataset files : Train and Validation
        self._train_dataset_file = data_dir / Path('train.tsv')
        self._valid_dataset_file = data_dir / Path('valid.tsv')

    def pre_process(self, config=None):
        self.prechecks(config)

        vocab_args = config.get('vocabs')
        self.make_vocabs(**vocab_args)

        emb_args = config.get('embedding', {})
        self.make_embedding(**emb_args)

        data_args = config.get('data')
        train_files = { 'seqs': data_args.get('train_seqs'), 
                        'tagseqs': data_args.get('train_tags') }
        val_files = { 'seqs': data_args.get('val_seqs'), 
                        'tagseqs': data_args.get('val_tags') }
        for key in ['train_seqs', 'train_tags', 'val_seqs', 'val_tags']:
            data_args.pop(key)
        self.prep_data(train_files, val_files, **data_args)
        self._prepared_flag.touch()

    def prechecks(self, config=None):
        if config is None:
            config = self.config

        vocab_args = config.get('vocabs')
        assert vocab_args
        assert vocab_args.get('corpus_files')
        
        tag_args = config.get('tags')
        assert tag_args
        assert tag_args.get('corpus_files')

        data_args = config.get('data')
        assert data_args
        assert data_args.get('train_seqs')
        assert data_args.get('train_tags')
        assert data_args.get('val_seqs')
        assert data_args.get('val_tags')

    def make_vocabs(self, corpus_files:FileList, tag_files:FileList, include_chars:bool=False, 
                    include_subwords:bool=False, max_words:int=32000, max_subwords:int=8000, **kwargs):
        if config is None:
            config = self.config

        log.info('Making word vocabs ...')
        word_vocab = Vocabs.make(corpus_files, vocab_size=max_words, level='word')
        Vocabs.save(word_vocab, self._word_vocab_file)

        if include_chars:
            log.info('Making char vocabs ...')
            char_vocab = Vocabs.make(corpus_files, level='char')
            Vocabs.save(char_vocab, self._char_vocab_file)
        
        if include_subwords:
            log.info('Making subword vocabs ...')
            subword_vocab = Vocabs.make(corpus_files, vocab_size=max_subwords)
            Vocabs.save(subword_vocab, self._subword_vocab_file)

        log.info('Making tag vocabs ...')
        tag_vocab = Vocabs(add_reserved='ner')
        ner_tags = Vocabs.make(tag_files, level='word')
        for token in ner_tags:
            if token.level == -1:
                continue
            tag_vocab.append(token)
        Vocabs.save(tag_vocab, self._tag_vocab_file)

    def split_dataset():
        pass

    def merge_vocabs():
        pass

    def index_dataset(self, dataset):
        pass

    def make_embedding(self, word_args=None, subword_args=None, normalize:float=0):
        if emb_args is None or len(emb_args) == 0:
            log.info('Embedding args not found. Skipping preembeding mats.')
            return
        emb_args = { 'word': word_args, 'subword': subword_args}
        for key, filename in zip(['word', 'subword'], [self._word_emb_file, self._subword_emb_file]):
            if emb_args[key]:
                args = emb_args[key]
                emb_dim  = args.get('dim', 300)
                emb_file = args.get('pretrained_file', None)
                # emb_model = args.get('pretrained_model', None)
                
                if not emb_file.exists():
                    log.warning(f'Embedding file : {emb_file} does not exists')
                    continue

                vocab = Vocabs.load(self.work_dir / Path(f'{key}.vocab'))
                emb = Embeddings(vocab, emb_dim=emb_dim, emb_file=emb_file)
                emb.make(lowered=args.get('lowered', True))
                if normalize != 0.0:
                    emb.normalize(normalize_to=normalize)
                Embeddings.save(emb, filename)
            else:
                log.info(f'{key} embedding args not found')

    def prep_test(self, test_file:FileType, include_chars:bool=False, include_subwords:bool=False,
                    max_seq_len:int=64, max_word_len:int=20, max_subword_len:int=10, truncate:bool=True):
        keys = []
        include = { 'seqs':True, 'charseqss':include_chars, 'subwordseqss':include_subwords }
        vocab_files = self.vocab_files

        vocabs, padshapes, dims = {}, {}, {}
        
        for key, flag in include.items():
            if flag:
                keys.append(key)
                dims[key] = 3 if key.endswith('ss') else 2
                vocabs[key] = Vocabs.load(vocab_files[key])
                shape = [max_seq_len]
                if dims[key] == 3:
                    shape.append(max_word_len if key == 'charseqss' else max_subword_len)
                padshapes[key] = shape

        test_ds = Dataset(keys, dims=dims)
        with FileReader(test_file) as ft:
            for line in ft:
                line = line.strip().split()
                test_ds.cols['seqs'].append(line)


        if include_chars:
            test_ds.cols['charseqss'] = NERPrepper._make_charseqss(test_ds.cols['seqs'])
        if include_subwords:
            test_ds.cols['subwordseqss'] = NERPrepper._make_subwordseqss(test_ds.cols['seqs'])

        loader = BatchIterable(test_ds)
        loader.vocabs = vocabs
        loader.index()
        loader.pad(padshapes)

        ds = loader.dataset
        return loader.dataset

    def prep_data(self, train_files:FileAny, val_files:FileAny, include_chars:bool=False, 
                    include_subwords:bool=False, max_seq_len:int=64, max_word_len:int=20, 
                    max_subword_len:int=10, truncate:bool=True, include_seqs:bool=True, **kwargs):

        keys = []
        include = { 'seqs' : True, 
                    'charseqss' : True if include_subwords else False, 
                    'subwordseqss' : True if include_subwords else False, 
                    'tagseqs':True }
        vocab_files = self.vocab_files

        vocabs, padshapes, dims = {}, {}, {}
        for key, flag in include.items():
            if flag:
                keys.append(key)
                dims[key] = 3 if key.endswith('ss') else 2
                vocabs[key] = Vocabs.load(vocab_files[key])
                shape = [max_seq_len]
                if dims[key] == 3:
                    shape.append(max_word_len if key == 'charseqss' else max_subword_len)
                padshapes[key] = shape

        train_ds = Dataset(keys, dims=dims)
        val_ds = Dataset(keys, dims=dims)

        datasets, dataloaders = dict(), dict()
        for key in ['train', 'val']:
            log.info(f'Prepping {key} data ...')
            datasets[key] = Dataset(keys, dims=dims)
            log.info(f'\tReading {key} data files ...')
            datasets[key].read( train_files if key == 'train' else val_files)    
            if include_chars:
                datasets[key].cols['charseqss'] = NERPrepper._make_charseqss(datasets[key].cols['seqs'])
            if include_subwords:
                datasets[key].cols['subwordseqss'] = NERPrepper._make_subwordseqss(datasets[key].cols['seqs'])

            log.info(f'\tMaking {key} dataloader . Indexing and Padding ...')
            dataloaders[key] = BatchIterable(datasets[key])
            dataloaders[key].vocabs = vocabs
            dataloaders[key].index()
            dataloaders[key].pad(padshapes)

            datasets[key] = dataloaders[key].dataset
            log.info(f'\tSaving prepped {key} datafile')
            save_file = self._train_dataset_file if key == 'train' else self._valid_dataset_file
            Dataset.save(datasets[key], save_file)

    @staticmethod
    def _make_charseqss(seqs):
        char_seqss = []
        for seq in seqs:
            char_seqs = NERPrepper._make_charseqs(seq)
            char_seqss.append(char_seqs)
        return char_seqss

    @staticmethod
    def _make_charseqs(seq):
        char_seqs = []
        for word in seq:
            char_seqs.append(list(word))
        return char_seqs

    @staticmethod
    def _make_subwordseqss(seqs):
        subword_seqss = []
        for seq in seqs:
            subword_seqs = NERPrepper._make_subwordseqs(seq)
            subword_seqss.append(subword_seqs)
        return subword_seqss

    @staticmethod
    def _make_subwordseqs(seq):
        if not self._subword_vocab_file.exists():
            return []

        codec = load_scheme(self._subword_vocab_file)
        vocab = Vocabs.load(self._subword_vocab_file)        
        
        subword_seqs = []
        tok_list = codec.encode(' '.join(seq))
        tok_list = [vocab.name(x) for x in tok_list]

        boundaries = []
        for p, x in enumerate(tok_list):
            if x.endswith(Reserved.SPACE_TOK[0]):
                boundaries.append(p)
        start = 0
        for end in boundaries:
            subword_seqs.append(tok_list[start:end+1])
        return subword_seqs

    def generate_embedding():
        pass