from ..models import ModelRegistry


class ModelFactory(object):
    
    @staticmethod
    def create_tagger(name, model_args, **kwargs):
        args = model_args
        if name == 'birnn':
            ModelFactory.check(model_args, ['ntags', 'nwords', 'word_emb_dim', 'word_rnn_hid_dim'])
            tagger = ModelRegistry.classes[name](kwargs.get('gpu', -1), 
                                                args['ntags'], args['nwords'], 
                                                args['word_emb_dim'],
                                                args['word_rnn_hid_dim'], 
                                                word_emb_mat = kwargs.get('word_emb_mat', None)
                                                activation_type = args.get('activation', 'gelu'),
                                                rnn_type = args.get('rnn_type', 'lstm'),
                                                drop_ratio = args.get('dropout', 0))
        elif name == 'birnn_rnn':
            pass
        elif name == 'birnn_crf':
            assert 'tag_seqs' in kwargs.keys()
            ModelFactory.check(model_args, ['ntags', 'nwords', 'word_emb_dim', 'word_rnn_hid_dim'])
            tagger = ModelRegistry.classes[name](kwargs.get('gpu', -1),
                                                args['ntags'], args['nwords'], 
                                                args['word_emb_dim'],
                                                args['word_rnn_hid_dim'], 
                                                word_emb_mat = kwargs.get('word_emb_mat', None)
                                                activation_type = args.get('activation', 'gelu'),
                                                rnn_type = args.get('rnn_type', 'lstm'),
                                                drop_ratio = args.get('dropout', 0))
            tagger.crf_layer.init_transition_mat_emp(kwargs.get('tag_seqs'))
        elif name == 'birnn_cnn_crf':
            pass
        else:
            raise ValueError("Can't find the model name in the registry")

        return tagger

    @staticmethod
    def check(model_args, keys):
        for key in keys:
            assert key in model_args.keys()
