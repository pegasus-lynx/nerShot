from .birnn import BiRNNTagger, BiRNNSchema
from .birnn_crf import BiRNNCRFTagger, BiRNNCRFSchema


class ModelRegistry():

    names = [ 'birnn', 'birnn_rnn', 'birnn_crf', 'birnn_cnn_crf' ] 

    schemas = {
        'birnn': BiRNNSchema,
        'birnn_rnn': None,
        'birnn_crf': BiRNNCRFSchema,
        'birnn_cnn_crf': None          
    }

    classes = {
        'birnn': BiRNNTagger,
        'birnn_rnn': None,
        'birnn_crf': BiRNNCRFTagger,
        'birnn_cnn_crf': None         
    }