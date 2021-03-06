from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
# from sklear
import torch
import hashlib

from ner import FileType, log
from ner.factories.criterion import CriterionFactory
from ner.factories.model import ModelFactory, ModelRegistry
from ner.lib.dataset import Dataset
from ner.lib.functionals import Converter as Cr
from ner.lib.tokenizer import Reserved, Tokenizer
from ner.lib.vocabs import Vocabs
from ner.preexp import NERPrepper
from ner.tool.file_io import FileReader, FileWriter, load_conf
from ner.tool.log import _get_now
from ner.utils.dir_func import DirFuncs as Df
from ner.utils.path_func import PathFuncs as Pf


class BaseDecoder(object):
    
    def __init__(self, work_dir, model_name=None, beam_size:int=5, ensemble:int=1, gpu:int=-1):
        if isinstance(work_dir, str):
            work_dir = Path(work_dir)

        self.work_dir = work_dir
        self.model_dir = work_dir / Path('models')

        self.model = None
        self.model_name = None
        self.model_schema = None

        self.beam_size = beam_size
        self.pad_idx   = Reserved.PAD_IDX
        self.ensemble  = ensemble

        self.gpu = gpu
        self.device = torch.device('cpu' if gpu < 0 else f'cuda:{gpu}')

        self._config_file = work_dir / 'conf.yml'
        self.config = load_conf(self._config_file)

        self.toker = Tokenizer()

    @property
    def decode_args(self):
        return {
            'beam_size': self.beam_size,
            'ensemble': self.ensemble,
            'model_name': self.model_name,
            'device': self.device
        }

    def load(self, model_name=None, model=None, model_args=None):
        if not model_name:
            model_name = self.config['model_name']
            self.model_name = model_name
            self.model_schema = ModelRegistry.schemas[model_name]
        if not model_args:
            model_args = self.config['model_args']
        if model is None:
            model = ModelFactory.create_tagger(model_name, model_args, gpu=self.gpu)
            model_paths = list(self.model_dir.glob('model_*.pkl'))
            state = BaseDecoder.maybe_ensemble_state(model_paths, self.ensemble, device=self.device)
            model.load_state_dict(state)
            log.info('Successfully restored the model state')    
        model = model.eval().to(device=self.device)
        self.model = model

    @staticmethod
    def average_states(model_paths, device:str='cpu'):
        assert model_paths
        for i, mp in enumerate(model_paths):
            next_state = BaseDecoder._ckpt_to_model_state(mp, device=device)
            if i < 1:
                state_dict = next_state
                key_set = set(state_dict.keys())
            else:
                assert key_set == set(state_dict.keys())
                for key in key_set:
                    state_dict[key] = (i*state_dict[key] + next_state[key]) / (i + 1)
        return state_dict

    @staticmethod
    def maybe_ensemble_state(model_paths, ensemble:int=1, device:str='cpu'):
        if len(model_paths) == 1:
            log.info(f" Restoring state from requested model {model_paths[0]}")
            return BaseDecoder._ckpt_to_model_state(model_paths[0], device=device)
        elif ensemble <= 1:
            model_path = model_paths[0]
            log.info(f" Restoring state from best known model: {model_path}")
            return BaseDecoder._ckpt_to_model_state(model_path, device=device)
        else:
            digest = hashlib.md5(";".join(str(p) for p in model_paths).encode('utf-8')).hexdigest()
            model_dir = model_paths[0].parent
            cache_file = model_dir / f'avg_state{len(model_paths)}_{digest}.pkl'
            if cache_file.exists():
                log.info(f"Cache exists: reading from {cache_file}")
                state = BaseDecoder._ckpt_to_model_state(cache_file, device=device)
            else:
                log.info(f"Averaging {len(model_paths)} model states :: {model_paths}")
                state = BaseDecoder.average_states(model_paths)
                if len(model_paths) > 1:
                    log.info(f"Caching the averaged state at {cache_file}")
                    torch.save(state, str(cache_file))
            return state

    @staticmethod
    def _ckpt_to_model_state(checkpt_path: str, device:str='cpu'):
        if isinstance(device, str):
            device = torch.device(device)
        state = torch.load(checkpt_path, map_location=device)
        if 'model_state' in state:
            state = state['model_state']
        return state

    def tokenize_file(self, inp_file:FileType, lang:str='en'):
        tok_file = inp_file.with_suffix('.tok')
        fw = open(tok_file, 'w')
        with FileReader(inp_file) as fr:
            for line in fr:
                line = line.strip()
                tokens = self.toker.tokenize(line, lang=lang)
        fw.close()

    def detokenize_file(self, out_file:FileType, lang:str='en'):
        detok_file = out_file.with_suffix('.detok')
        fw = open(detok_file, 'w')
        with FileReader(out_file) as fr:
            for line in fr:
                line.strip().split()
                pass
        fw.close()

class NERDecoder(BaseDecoder):
    
    def __init__(self, work_dir, model_name:str=None, beam_size:int=5, ensemble:int=1, gpu:int=-1):
        super(NERDecoder, self).__init__(work_dir, model_name=model_name, beam_size=beam_size,
                                            ensemble=ensemble, gpu=gpu)
        self.data_dir = self.work_dir / Path('data/')
        self.test_dir = Df.make_dir(self.work_dir / Path('test/'))
        self.vocab_files_ = {
            'word': self.data_dir / Path('word.vocab'),
            'subword': self.data_dir / Path('subword.vocab'),
            'char': self.data_dir / Path('char.vocab'),
            'tag' : self.data_dir / Path('tag.vocab')
        }
        self.vocabs = self._load_vocabs()
        self.prepper = NERPrepper(self.work_dir)
        self.criterion = CriterionFactory.create(self.config['optim'].get('criterion', 'nll'))

    def _load_vocabs(self):
        vocabs = { key:Vocabs.load(val) if val.exists() else None 
                            for key, val in self.vocab_files_.items()}
        return vocabs

    def eval_decoded(self, out_file, tag_file, avg='macro', max_len:int=64):
        with open(out_file, 'r') as fo:
            y_preds = []
            for line in fo:
                line = line.strip().split()
                # y_pred = [ 0 for x in range(max_len)]
                # for p, word in enumerate(line):
                    # if p >= max_len:
                        # break
                    # y_pred[p] = self.vocabs['tag'].index(word)
                # y_preds.append([self.vocabs['tag'].index(x) for x in line])
                y_preds.extend([self.vocabs['tag'].index(x) for x in line])
                # y_preds.append(y_pred)
            # y_preds = Cr.list2numpy(y_preds)


        with open(tag_file, 'r') as ft:
            y_trues = []
            for line in ft:
                line = line.strip().split()
                y_true = [ 0 for x in range(max_len)]
                for p, word in enumerate(line):
                    if p >= max_len:
                        break
                    y_true[p] = self.vocabs['tag'].index(word)
                # y_trues.append([self.vocabs['tag'].index(x) for x in line])
                # y_trues.extend([self.vocabs['tag'].index(x) for x in line])
                y_trues.extend(y_true)
            # y_trues = Cr.list2numpy(y_trues)

        labels = self.vocabs['tag'].vocabulary
        scores = precision_recall_fscore_support(y_trues, y_preds, average=avg, labels=range(1,len(labels)))
        return scores

    def decode(self, test_suits, test_name:str=None):
        log.info(f'Decoder Args : {self.decode_args}')
        if not test_name:
            test_name = _get_now()        
        test_dir = Df.make_dir(self.test_dir / Path(f'test_{test_name}'))
        decoder_func = self._get_decoder_func()
        for key in test_suits.keys():
            suit_name = key
            log.info(f'Decoding suit : {suit_name}')
            seq_file, tagseq_file = test_suits[key]
            scores = self.decode_suit(test_dir, suit_name, seq_file, 
                                    tagseq_file, decoder_func, avg='macro')
            precision, recall, macro_f1, _ = scores
            log.info(f'Scores - Precision : {precision} | Recall : {recall} | Macro F1 : {macro_f1}')

    def decode_files(self, dec_files):
        log.info(f'Decoder Args : {self.decode_args}')
        test_dir = Df.make_dir(self.test_dir / Path(f'files'))
        decoder_func = self._get_decoder_func()
        for dec_file in dec_files:
            log.info(f'Decoding file : {dec_file.name}')
            self.decode_file(test_dir, suit_name, seq_file, tagseq_file, decoder_func)  

    def _get_decoder_func(self):
        # dec_types = self.model_schema.decoder_type
        # if 'beam' in dec_types and self.beam_size > 1:
            # return self.beam_decode
        return self.greedy_decode

    def greedy_decode(self, row):
        tensors = [Cr.list2tensor([x], gpu=self.gpu) for x in row]
        out_tensor = self.model.predict(*tensors)
        out_tensor = out_tensor.squeeze()
        out_array = Cr.tensor2numpy(out_tensor)
        # print(out_array)
        return out_array.tolist()

    def decode_suit(self, test_dir, name, seq_file, tagseq_file, decoder_func=None, **kwargs):
        pref = f'suit_{name}'
        seq_file = Df.copy_file(seq_file, test_dir / Path(f'{pref}_seq.txt'))
        tag_file = Df.copy_file(tagseq_file, test_dir / Path(f'{pref}_tag.txt'))
        out_file = test_dir / Path(f'{pref}_tag.out.txt')
        
        if not decoder_func:
            decoder_func = self._get_decoder_func()
        dec_ds = self.decode_file_(seq_file, decoder_func)
        self._write_decoded(dec_ds, out_file)
        scores = self.eval_decoded(out_file, tagseq_file, avg=kwargs.get('avg', 'macro'))
        return scores
        
    def decode_file(self, test_dir, seq_file, decoder_func=None):
        seq_file = Df.copy_file(seq_file, test_dir / Path(f'{seq_file.name}.seq.txt'))
        out_file = test_dir / Path(f'{seq_file.name}.out.txt')
        if not decoder_func:
            decoder_func = self._get_decoder_func()
        dec_ds = self.decode_file_(seq_file, decoder_func)
        self._write_decoded(dec_ds, out_file)

    def _write_decoded(self, dec_ds, out_file):
        fw = FileWriter(out_file)
        for row in dec_ds:
            dec = row[0]
            fw.writeline(' '.join(dec))
        fw.close()

    def decode_file_(self, seq_file, decoder_func):
        data_args = self.prepper.config['data']
        vocab_args = self.prepper.config['vocabs']

        include_chars = vocab_args.get('include_chars', False)
        include_subwords = vocab_args.get('include_subwords', False)

        seq_len = data_args.get('max_seq_len', 64)
        word_len = data_args.get('max_word_len', 20)
        subword_len = data_args.get('max_subword_len', 10)

        test_ds = self.prepper.prep_test(seq_file, include_chars=include_chars,
                            include_subwords=include_subwords, max_seq_len=seq_len,
                            max_word_len=word_len, max_subword_len=subword_len)

        Dataset.save(test_ds, seq_file.parent / Path('temp.txt'))
        out_ds = Dataset(['decs'])
        for row in test_ds:
            out = decoder_func(row)
            dec = [self.vocabs['tag'].name(x) for x in out]
            out_ds.cols['decs'].append(dec)
        return out_ds                        
