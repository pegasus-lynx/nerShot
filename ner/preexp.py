from pathlib import Path
from nlcodec import load_scheme
from lib.vocabs import Vocabs, Embeddings
from lib.tokenizer import Reserved
from lib.dataset import Dataset, BatchIterable
from tool.file_io import load_conf

class BasePrepper(object):
    def __init__(self, work_dir, config=None):
        if type(work_dir) is str:
            work_dir = Path(work_dir)

        self.work_dir = work_dir
        self.log_dir = work_dir / 'log'
        self.log_file = self.log_dir / 'prep.log'
        self.data_dir = self.work_dir / 'data'
        self._config_file = work_dir / 'prep.yml'
        if isinstance(config, str) or isinstance(config, Path):
            config = load_conf(config)
        self.config = config if config else load_conf(self._config_file)
        self._prepared_flag = self.data_dir / '_PREPARED'

    def has_prepared(self):
        return self._prepared_flag.exists()

class NERPrepper(BasePrepper):
    def __init__(self, work_dir, config=None):
        super(NERPrepper, self).__init__(work_dir, config)
        if type(work_dir) is str:
            work_dir = Path(work_dir)

        self._prepared_flag = work_dir / Path('_PREPARED')

        exp_dir = work_dir
        work_dir = work_dir / Path('data/')

        if not work_dir.exists():
            work_dir.mkdir()

        self.work_dir = work_dir
        self._word_vocab_file = work_dir / Path('word.vocab')
        self._subword_vocab_file = work_dir / Path('subword.vocab')
        self._char_vocab_file = work_dir / Path('char.vocab')
        self._tag_vocab_file = work_dir / Path('tag.vocab')

        self._word_emb_file = work_dir / Path('word.emb.npz')
        self._subword_emb_file = work_dir / Path('subword.emb.npz')

        self._train_dataset_file = work_dir / Path('train.tsv')
        self._valid_dataset_file = work_dir / Path('valid.tsv')

    def pre_process(self, config=None):
        self.make_vocabs()
        self.make_embedding()
        self.prep_data()
        self._prepared_flag.touch()

    def make_vocabs(self, config=None):
        if config is None:
            config = self.config

        vocab_args = config.get('vocabs')        
        corpus_files = vocab_args.get('corpus_files', [])
        word_vocab = Vocabs.make(corpus_files, vocab_size=config.get('max_word_tokens', 30000), level='word')
        Vocabs.save(word_vocab, self._word_vocab_file)

        include_chars, include_subwords = config.get('include_chars', False), config.get('include_subwords', False)
        if include_chars:
            char_vocab = Vocabs.make(corpus_files, level='char')
            Vocabs.save(char_vocab, self._char_vocab_file)
        if include_subwords:
            subword_vocab = Vocabs.make(corpus_files, vocab_size=config.get('max_word_tokens', 8000))
            Vocabs.save(subword_vocab, self._subword_vocab_file)

        tag_args = config.get('tags', {})
        tag_corpus_files = tag_args.get('corpus_files')
        tag_vocab = Vocabs(add_reserved='ner')
        tags = Vocabs.make(tag_corpus_files, level='word')
        for token in tags:
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

    def make_embedding(self, config=None):
        if config is None:
            config = self.config
        embedding_args = config.get('embedding')
        for key, filename in zip(['word', 'subword'], [self._word_emb_file, self._subword_emb_file]):
            if key in embedding_args.keys():
                args = embedding_args.get(key)
                emb_dim = args.get('dim')
                emb_file = args.get('pretrained_file', None)
                vocab = Vocabs.load(self.work_dir / Path(f'{key}.vocab'))
                emb = Embeddings(vocab, emb_dim=emb_dim, emb_file=emb_file)
                emb.make(lowered=args.get('lowered', True))
                Embeddings.save(emb, filename)

    def prep_data(self, config=None):
        if config is None:
            config = self.config

        data_args = config.get('data', {})
        vocab_args = config.get('vocabs', {})

        max_seq_len = data_args.get('max_seq_len', 64)
        max_word_len = data_args.get('max_word_len', 24)
        max_subword_len = data_args.get('max_subword_len', 5)
        truncate: data_args.get('truncate', True)

        keys = ['seqs']
        dims = {'seqs':2}
        vocabs = {'seqs': Vocabs.load(self._word_vocab_file)}
        padshapes = {'seqs' : [max_seq_len]}
        if vocab_args.get('include_chars', False):
            keys.append('charseqss')
            dims['charseqss'] = 3
            vocabs['charseqss'] = Vocabs.load(self._char_vocab_file)
            padshapes['charseqss'] = [max_seq_len, max_word_len]
        if vocab_args.get('include_subwords', False):
            keys.append('subwordseqss')
            dims['subwordseqss'] = 3
            vocabs['subwordseqss'] = Vocabs.load(self._subword_vocab_file)
            padshapes['subwordseqss'] = [max_seq_len, max_subword_len]
        keys.append('tagseqs')
        dims['tagseqs'] = 2
        vocabs['tagseqs'] = Vocabs.load(self._tag_vocab_file)
        padshapes['tagseqs'] = [max_seq_len]


        train_ds = Dataset(keys, dims=dims)
        val_ds = Dataset(keys, dims=dims)

        train_ds.read({'seqs':data_args.get('train_seqs'), 'tagseqs':data_args.get('train_tags')})
        val_ds.read({'seqs':data_args.get('valid_seqs'), 'tagseqs':data_args.get('valid_tags')})

        if vocab_args.get('include_chars', False):
            train_ds.cols['charseqss'] = self._make_charseqss(train_ds.cols['seqs'])
            val_ds.cols['charseqss'] = self._make_charseqss(val_ds.cols['seqs'])

        if vocab_args.get('include_subwords', False):
            train_ds.cols['subwordseqss'] = self._make_subwordseqss(train_ds.cols['seqs'])
            val_ds.cols['subwordseqss'] = self._make_subwordseqss(val_ds.cols['seqs'])

        train_loader = BatchIterable(train_ds)
        val_loader = BatchIterable(val_ds)
        train_loader.vocabs = vocabs
        val_loader.vocabs = vocabs

        train_loader.index()
        train_loader.pad(padshapes)
        val_loader.index()
        val_loader.pad(padshapes)

        train_ds = train_loader.dataset
        val_ds = val_loader.dataset

        Dataset.save(train_ds, self._train_dataset_file)
        Dataset.save(val_ds, self._valid_dataset_file)

    def _make_charseqss(self, seqs):
        char_seqss = []
        for seq in seqs:
            char_seqs = []
            for word in seq:
                char_seqs.append(list(word))
            char_seqss.append(char_seqs)
        return char_seqss

    def _make_subwordseqss(self, seqs):
        codec = load_scheme(self._subword_vocab_file)
        vocab = Vocabs.load(self._subword_vocab_file)
        subword_seqss = []
        for seq in seqs:
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
            subword_seqss.append(subword_seqs)
        return subword_seqss

    def generate_embedding():
        pass