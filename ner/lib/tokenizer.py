""" Utils for tokenizing the text """

__author__ = "Dipesh Kumar"

import json
import re
from collections import Counter
from typing import List, Union

from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize, sentence_tokenize
from sacremoses import MosesDetokenizer, MosesTokenizer


class CharTokens(object):

    # Symbols
    symbols = set([
        '.', ',', '(', ')', '[', ']', '{', '}', '!',
        ':', '-', '"', "'", ';', '<', '>', '?', '&',
        '–', '@', ' ', '\t', '\n'
    ])

    # Eglish Digits and Chars
    en_digits = set(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    en_chars  = set([
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
        'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
    ])

    # Hindi Digits and Chars
    hi_digits = set(['०', '१', '२', '३', '४', '५', '६', '७', '८', '९'])
    hi_chars  = set([
        'ऀ', 'ँ', 'ं', 'ः', 'ऄ', 'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ऌ', 
        'ऍ', 'ऎ', 'ए', 'ऐ', 'ऑ', 'ऒ', 'ओ', 'औ', 'क', 'ख', 'ग', 'घ', 'ङ', 
        'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 
        'ध', 'न', 'ऩ', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ऱ', 'ल', 'ळ', 
        'ऴ', 'व', 'श', 'ष', 'स', 'ह', 'ऺ', 'ऻ', '़', 'ऽ', 'ा', 'ि', 'ी', 'ु', 
        'ू', 'ृ', 'ॄ', 'ॅ', 'ॆ', 'े', 'ै', 'ॉ', 'ॊ', 'ो', 'ौ', '्', 'ॎ', 'ॏ', 'ॐ', 
        '॑', '॒', '॓', '॔', 'ॕ', 'ॖ', 'ॗ', 'क़', 'ख़', 'ग़', 'ज़', 'ड़', 'ढ़', 'फ़', 
        'य़', 'ॠ', 'ॡ', 'ॢ', 'ॣ', '।', '॥', '॰', 'ॱ', 'ॲ', 'ॳ', 'ॴ', 'ॵ', 
        'ॶ', 'ॷ', 'ॸ', 'ॹ', 'ॺ', 'ॻ', 'ॼ', 'ॽ', 'ॾ', 'ॿ'
    ])

    langs = {
        'en':{'digits':en_digits, 'chars':en_chars}, 
        'hi':{'digits':hi_digits, 'chars':hi_chars}
    }

    @classmethod
    def eng(cls, text:str):
        for x in text:
            digit = x in cls.en_digits
            letter = x in cls.en_chars
            symbol = x in cls.symbols
            valid = (digit or letter or symbol)
            if not valid:
                return False
        return True

    @classmethod
    def hin(cls, text:str):
        for x in text:
            digit = x in cls.hi_digits
            digit = digit or (x in cls.en_digits)
            letter = x in cls.hi_chars
            symbol = x in cls.symbols
            valid = (digit or letter or symbol)
            if not valid:
                return False
        return True

    @classmethod
    def is_symbol(cls, text:str):
        return text in cls.symbols

    @classmethod
    def check_lang(cls, text, lang=None:Union[str,List[str]]):
        if len(text) != 1:
            return None

        # Setting the keys for matching       
        keys = cls.langs.keys()
        if lang is not None:
            if type(lang) == list:
                keys = lang
            else:
                keys = [lang]

        for key in keys:
            if key not in cls.langs.keys():
                continue
            if text in cls.langs[key]['digits']:
                return key
            if text in cls.langs[key]['chars']:
                return key

        if text in cls.symbols:
            return 'sym'

        return None

class Reserved(object):
    PAD_TOK = "<pad>", 0
    OOV_TOK = "<unk>", 1
    SOS_TOK = "<s>", 2
    EOS_TOK = "</s>", 3
    CLS_TOK = "<cls>", 4
    SPACE_TOK = "▁", 5
    BRK_TOK = "<brk>", 6

    PAD_IDX, OOV_IDX, SOS_IDX, EOS_IDX, CLS_IDX, SPACE_IDX, BRK_IDX = 0,1,2,3,4,5,6

    TAG_TOKS = [ PAD_TOK[0], OOV_TOK[0], SOS_TOK[0], EOS_TOK[0], BRK_TOK[0], SPACE_TOK[0] ]
    ENT_TOKS = [ ("<num{}>", 10), ("<alf{}>", 5), ("<date{}>", 5)]

    @classmethod
    def all(cls, res_type = 'word'):
        if res_type == 'word':
            return cls._all_word()
        elif res_type == 'char':
            return cls._all_char()
        elif res_type == 'ner':
            return cls._all_ner()
        else:
            return []

    @classmethod
    def _all_word(cls):
        tokens = cls.TAG_TOKS
        for tok, freq in cls.ENT_TOKS:
            for p in range(freq):
                tokens.append(tok.format(p))
        return tokens

    @classmethod
    def _all_char(cls):
        return cls.TAG_TOKS

    @classmethod
    def _all_ner(cls):
        return [PAD_TOK[0]]

    @classmethod
    def validate(cls, word):
        if word in cls.TAG_TOKS:
            return True

        for ent, freq in cls.ENT_TOKS:
            for i in range(freq):
                if word == ent.format(i):
                    return True

        return False

class Tokenizer(object):

    def __init__(self):
        # For indic languages
        factory = IndicNormalizerFactory()
        self.normalizer = factory.get_normalizer("hi")

        # Moses Tokenizer
        self.mts = dict()
        self.mts["en"] = MosesTokenizer(lang="en")

    def tokenize(self, text: str, lang: str = "en"):
        text = text.strip()
        if len(text) == 0:
            return []

        if lang in ["en", "ru"]:
            return self.moses(text, lang)
        elif lang == "hi":
            return self.hindi(text)
        else:
            raise(NotImplementedError("The tokenizer works for : English, Hindi and Russian"))

    def moses(self, text:str, lang:str):
        if lang not in self.mst.keys():
            self.mts['lang'] = MosesTokenizer(lang=lang)
        return self.mts[lang].tokenize(text, )
        
    def hindi(self, text:str, normalize=False):
        if normalize:
            text = self.normalizer.normalize(text)
        return indic_tokenize.trivial_tokenize(text)

class Masker(object):
    
    default_regexs = {
        'num': [
            "^-?[0-9]+(\\.[0-9]+)?$",
            "^-?([0-9]+,)+[0-9]+(\\.[0-9]+)?$"
        ],
        'date': [
            "^[0-9]{4} ?[-/.] ?[0-9]{1,2} ?[-/.] ?[0-9]{1,2}$",
            "^[0-9]{1,2} ?[-/.] ?[0-9]{1,2} ?[-/.] ?[0-9]{4}$"
        ]
        'alf': [
            ".*[0-9]+.*[a-zA-Z]*.*",
            ".*[a-zA-Z]*.*[0-9]+.*"
        ]
    }

    def __init__(self, regexs:Dict[str,List[str]]=None):
        self.regexs = self.default_regexs
        if regexs is not None:
            for key, regex in regexs.items():
                self.regexs[key] = regex

    def match(self, token:str):
        if Reserved.validate(token):
            return 'res'
        for key in self.regexs.keys():
            for regex in self.regexs[key]:
                if re.search(regex, token) is not None:
                    return key
        return None

    