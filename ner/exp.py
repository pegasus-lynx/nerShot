import copy
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, Any
import time

import numpy as np
import torch

from __init__ import log, yaml
from tool.file_io import FileReader, FileWriter
from lib.vocabs import Vocabs, Embeddings
from lib.dataset import Dataset, Batch, BatchIterable
from factories import CriterionFactory, OptimizerFactory, ModelFactory
from models import ModelRegistry
from utils.states import TrainerState, EarlyStopper
from tool.file_io import load_conf

class BaseExperiment(object):
    def __init__(self, work_dir, config=None):
        if type(work_dir) is str:
            work_dir = Path(work_dir)

        log.info(f"Initializing an experiment. Directory = {work_dir}")
        self.work_dir = work_dir
        self.log_dir = work_dir / 'logs'
        self.log_file = self.log_dir / 'exp.log'
        self.data_dir = work_dir / 'data'
        self.model_dir = work_dir / 'models'
        self._config_file = work_dir / 'conf.yml'
        if isinstance(config, str) or isinstance(config, Path):
            config = load_conf(config)
        self.config = config if config else load_conf(self._config_file)

        self._prepared_flag = self.data_dir / '_PREPARED'
        self._trained_flag = self.data_dir / '_TRAINED'

        self.device, self.gpu = self._set_self()
        

    def has_prepared(self):
        return self._prepared_flag.exists()

    def has_trained(self):
        return self._trained_flag.exists()

    def store_model(self, epoch: int, model, train_score: float, val_score: float, keep: int,
                    prefix='model', keeper_sort='step'):
        """
        saves model to a given path
        :param epoch: epoch number of model
        :param model: model object itself
        :param train_score: score of model on training split
        :param val_score: score of model on validation split
        :param keep: number of good models to keep, bad models will be deleted
        :param prefix: prefix to store model. default is "model"
        :param keeper_sort: criteria for choosing the old or bad models for deletion.
            Choices: {'total_score', 'step'}
        :return:
        """
        # TODO: improve this by skipping the model save if the model is not good enough to be saved
 
        name = f'{prefix}_{epoch:03d}_{train_score:.6f}_{val_score:.6f}.pkl'
        path = self.model_dir / name
        log.info(f"Saving epoch {epoch} to {path}")
        torch.save(model, str(path))

        store = False
        if keeper_sort == 'total_score':
            curr_models, scores = self.list_models(sort='total_score', desc=False)
            if train_score + valid_score > scores[curr_models[-1]]['total_score']:
                store = True 
        elif keeper_sort == 'step':
            store = True
        
        if len(curr_models) < keep:
            store = True

        del_models = []
        if keeper_sort == 'total_score':
            del_models = self.list_models(sort='total_score', desc=False)[keep:]
        elif keeper_sort == 'step':
            del_models = self.list_models(sort='step', desc=True)[keep:]
        else:
            Exception(f'Sort criteria{keeper_sort} not understood')
        for d_model in del_models:
            log.info(f"Deleting model {d_model} . Keep={keep}, sort={keeper_sort}")
            os.remove(str(d_model))

        fw = FileWriter(os.path.join(self.model_dir, 'scores.tsv'), append=True)
        cols = [str(epoch), datetime.now().isoformat(), name, f'{train_score:g}', f'{val_score:g}']
        fw.writeline('\t'.join(cols))
        fw.close()

    @staticmethod
    def _path_to_validn_score(path):
        parts = str(path.name).replace('.pkl', '').split('_')
        valid_score = float(parts[-1])
        return valid_score

    @staticmethod
    def _path_to_total_score(path):
        parts = str(path.name).replace('.pkl', '').split('_')
        tot_score = float(parts[-2]) + float(parts[-1])
        return tot_score

    @staticmethod
    def _path_to_step_no(path):
        parts = str(path.name).replace('.pkl', '').split('_')
        step_no = int(parts[-3])
        return step_no

    def list_models(self, sort: str = 'step', desc: bool = True) -> List[Path]:
        """
        Lists models in descending order of modification time
        :param sort: how to sort models ?
          - valid_score: sort based on score on validation set
          - total_score: sort based on validation_score + training_score
          - mtime: sort by modification time
          - step (default): sort by step number
        :param desc: True to sort in reverse (default); False to sort in ascending
        :return: list of model paths
        """
        paths = self.model_dir.glob('model_*.pkl')
        sorters = {
            'valid_score': self._path_to_validn_score,
            'total_score': self._path_to_total_score,
            'mtime': lambda p: p.stat().st_mtime,
            'step': self._path_to_step_no
        }

        scores = dict()
        for path in paths:
            steps, train_score, valid_score = str(path).replace('.pkl', '').split('_')[-3:]
            scores[path] = { 'steps': steps, 'valid_score':valid_score, 
                                    'total_score': train_score + valid_score }

        if sort not in sorters:
            raise Exception(f'Sort {sort} not supported. valid options: {sorters.keys()}')
        return sorted(paths, key=sorters[sort], reverse=desc), scores

    def _get_first_model(self, sort: str, desc: bool) -> Tuple[Optional[Path], int]:
        """
        Gets the first model that matches the given sort criteria
        :param sort: sort mechanism
        :param desc: True for descending, False for ascending
        :return: Tuple[Optional[Path], step_num:int]
        """
        models, _ = self.list_models(sort=sort, desc=desc)
        if models:
            step, train_score, valid_score = models[0].name.replace('.pkl', '').split('_')[-3:]
            return models[0], int(step)
        else:
            return None, 0

    def get_best_known_model(self) -> Tuple[Optional[Path], int]:
        """Gets best Known model (best on lowest scores on training and validation sets)
        """
        return self._get_first_model(sort='total_score', desc=False)   

    def get_last_saved_model(self) -> Tuple[Optional[Path], int]:
        return self._get_first_model(sort='step', desc=True)     

    @property
    def model_args(self) -> Optional[Dict]:
        """
        Gets args from file
        :return: args if exists or None otherwise
        """
        return self.config.get('model_args')

    @model_args.setter
    def model_args(self, model_args):
        """
        set model args
        """
        self.config['model_args'] = model_args

    @property
    def optim_args(self) -> Tuple[Optional[str], Dict]:
        """
        Gets optimizer args from file
        :return: optimizer args if exists or None otherwise
        """
        opt_conf = self.config.get('optim')
        if opt_conf:
            return opt_conf.get('name'), opt_conf.get('args')
        else:
            return None, {}

    @optim_args.setter
    def optim_args(self, optim_args: Tuple[str, Dict]):
        """
        set optimizer args
        """
        name, args = optim_args
        self.config['optim'].update({'name': name, 'args': args})

    def train_mode(self, mode: bool):
        torch.set_grad_enabled(mode)
        self.model.train(mode)

    def _set_self(self):
        if torch.cuda.is_available():
            return 'cuda', 0
        return 'cpu', -1

    @property
    def model_name(self):
        return self.config.get('model_name')

    @model_name.setter
    def model_name(self, model_name):
        self.config['model_name'] = model_name


class NERTaggingExperiment(BaseExperiment):

    def __init__(self, work_dir, config=None):
        super(NERTaggingExperiment, self).__init__(work_dir, config=config)
        if type(work_dir) is str:
            work_dir = Path(work_dir)

        self._word_vocab_file = self.data_dir / Path('word.vocab')
        self._subword_vocab_file = self.data_dir / Path('subword.vocab')
        self._char_vocab_file = self.data_dir / Path('char.vocab')
        self._tag_vocab_file = self.data_dir / Path('tag.vocab')

        self._word_emb_file = self.data_dir / Path('word.emb.npz')
        self._subword_emb_file = self.data_dir / Path('subword.emb.npz')

        self._train_dataset_file = self.data_dir / Path('train.tsv')
        self._valid_dataset_file = self.data_dir / Path('valid.tsv')

        self.schema = ModelRegistry.schemas[self.model_name]()

        self.model = None
        self.optimizer = None
        self.criterion = None

    def train_model(self, args=None):
        train_args = copy.deepcopy(self.config.get('trainer', {}))  
        if args:
            train_args.update(args)
        
        train_steps = train_args['steps']

        _, last_step = self.get_last_saved_model()
        if self._trained_flag.exists():
            try:
                last_step = max(last_step, yaml.load(self._trained_flag.read_text())['steps'])
            except Exception as _:
                pass

        if last_step >= train_steps and (finetune_steps is None or last_step >= finetune_steps):
            log.warning(
                f"Already trained upto {last_step}; Requested: train={train_steps}, finetune={finetune_steps} Skipped")
            return

        if last_step < train_steps:
            stopped = self.trainer(**train_args)
            status = dict(steps=train_steps, early_stopped=stopped)
            yaml.dump(status, stream=self._trained_flag)

    def trainer(self, steps:int, check_point:int, batch_size:int, **args):
        self.train_state = TrainerState(len(self.train_loader), 
                                        check_point, self.last_step)
        eargs = args.get('early_stop', {})
        self.stopper = EarlyStopper(cur_steps=self.last_step, **eargs)
        unsaved_state = False
        early_stopped = False
        for step in range(self.last_step+1, steps):
            self._train()
            self.last_step = step
            unsaved_state = True
            if self.train_state.is_check_point():
                train_loss = self.train_state.reset()
                val_loss = self._predict()
                self.make_check_point(train_loss, val_loss, keep=keep_models)
                unsaved_state = False
                self.train_mode(True)
                stopper.step()
                stopper.validation(val_loss)
                if stopper.is_stop():
                    log.info(f"Stopping at {stopper.cur_step} because {stopper.by}"
                                f" didnt improve over {stopper.patience} checkpoints")
                    early_stopped = True
                    break
        if unsaved_state:
            train_loss = self.train_state.reset()
            val_loss = self._predict()
            self.make_check_point(train_loss, val_loss, keep=keep_models)
        return early_stopped

    def _train(self):
        self.train_mode(True)
        for p, batch in enumerate(self.train_loader):
            outs, loss = self.model(*batch.forward)
            labels = batch.labels[0]
            if loss is None:
                loss = self.criterion(outs.view(-1, self.model.ntags), 
                                        labels.view(-1))
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.model.zero_grad()
            self.train_state.step(loss.item())
        return self.running_loss()

    def _predict(self, get_outs:bool=False):
        self.train_mode(False)
        val_loss = 0
        val_outs = []
        with torch.no_grad():
            for p, batch in enumerate(self.valid_loader):
                outs, loss = self.model(*batch.forward)
                if loss is None:
                    loss = self.criterion(outs.view(-1, self.ntags), 
                                            labels.view(-1))
                val_loss += loss.item()
                if get_outs:
                    val_outs.append(outs)
        total_loss = val_loss / len(self.valid_loader)
        if get_outs:
            outputs = torch.cat(val_outs, dim=0)
            return total_loss, outputs
        return total_loss

    def load(self, model_name:str=None, indexed:bool=True):
        if not self.has_prepared():
            log.warning('Not able to load the model. Prepared Flag Missing')
        self.load_vocabs()
        self.load_embeddings()
        self.load_data(indexed=indexed)
        self.load_model(model_name)

    def load_vocabs(self):
        log.info('Loading Vocabs ...')
        self.word_vocab, self.subword_vocab, self.char_vocab, self.tag_vocab = [
            Vocabs.load(f) if f.exists() else None for f in (
                self._word_vocab_file, self._subword_vocab_file,
                self._char_vocab_file, self._tag_vocab_file )]

    def load_data(self, indexed:bool=True):
        log.info('Loading Datasets : Train , Valid ...')
        train_dataset, valid_dataset = [
            Dataset.load(f) if f.exists() else None for f in (
                self._train_dataset_file, self._valid_dataset_file)]

        batch_size = self.config['trainer'].get('batch_size', 32)
        log.info('Making dataloaders ...')
        self.train_loader = BatchIterable(train_dataset, schema=self.schema, batch_size=batch_size, 
                                            indexed=indexed, gpu=self.gpu)
        self.val_loader = BatchIterable(valid_dataset, schema=self.schema, batch_size=batch_size,
                                            indexed=indexed, gpu=self.gpu)

    def load_model(self, model_name:str=None): 
        if model_name is None:
            model_name = self.model_name
        log.info(f'Loading Model : {model_name} ...')
        
        self.model = ModelFactory.create_tagger(model_name, 
                                                self.model_args, 
                                                gpu=self.gpu)

        _, opt_args = self.optim_args
        self.optimizer, self.scheduler = OptimizerFactory.create(self.optim_args, self.model)
        self.criterion = CriterionFactory.create(opt_args['criterion'])
        self.last_step = 0

        last_model, self.last_step = self.get_last_saved_model()
        if last_model:
            log.info(f"Resuming training from step:{self.last_step}, model={last_model}")
            state = torch.load(last_model, map_location=self.device)
            model_state = state['model_state'] if 'model_state' in state else state
            if 'optim_state' in state:
                optim_state = state['optim_state']

            self.model.load_state_dict(model_state)
            self.model = self.model.to(self.device)

            self.optimizer.load_state_dict(optim_state)
        else:
            log.info("No earlier check point found. Looks like this is a fresh start")

    def load_embeddings(self):
        if self._word_emb_file.exists():
            self.word_emb = Embeddings.load(self._word_emb_file)
        if self._subword_emb_file.exists():
            self.subword_emb = Embeddings.load(self._subword_emb_file)

    def make_check_point(self, train_loss, valid_loss, keep:int=5):
        step_num = self.last_step
        log.info(f"Checkpoint at optimizer step {step_num}. Training Loss {train_loss:g},"
                 f" Validation Loss:{val_loss:g}")
        model = self.model
        state = {
            'model_state': model.state_dict(),
            'optim_state': self.optimizer.state_dict(),
            'step': step_num,
            'train_loss': train_loss,
            'val_loss': valid_loss,
            'time': time.time(),
            'model_name': self.model_name,
            'model_args': self.model_args
        } 

        self.store_model(step_num, state, train_score=train_loss,
                        val_score=valid_loss, keep=keep)
        