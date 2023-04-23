# -*- coding: utf-8 -*-

"""
Language-related data loading helper functions and class wrappers.
"""

import re, pdb, json
import torch
import codecs
from transformers import AutoTokenizer, DistilBertModel

# UNK_TOKEN = '<unk>'
UNK_TOKEN = '[UNK]'
PAD_TOKEN = '<pad>'
SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')     # re.compile()用以编译正则表达式，r’(\W+)’表示匹配一个或多个非字母数字及下划线(任何非单词字符)，相当于‘(\\W+)’


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def __getitem__(self, a):
        if isinstance(a, int):
            return self.idx2word[a]
        elif isinstance(a, list):
            return [self.idx2word[x] for x in a]
        elif isinstance(a, str):
            return self.word2idx[a]
        else:
            print(a)
            raise TypeError("Query word/index argument must be int or str")

    def __contains__(self, word):
        return word in self.word2idx


class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()

    def set_max_len(self, value):
        self.max_len = value

    def load_file(self, filename):
        with codecs.open(filename, 'r', 'utf-8') as f:
            for line in f:
                line = line.strip()
                self.add_to_corpus(line)
        self.dictionary.add_word(UNK_TOKEN)
        self.dictionary.add_word(PAD_TOKEN)

    def add_to_corpus(self, line):
        """Tokenizes a text line."""
        # Add words to the dictionary
        words = line.split()
        # tokens = len(words)
        for word in words:
            word = word.lower()
            self.dictionary.add_word(word)

    def tokenize(self, line, max_len):
        # Tokenize line contents
        words = SENTENCE_SPLIT_REGEX.split(line.strip())    # 删除头尾空格
        words = [w.lower() for w in words if len(w.strip()) > 0]    # lower() tramsfer to lower case; strip() delete the blanks

        # if words[-1] == '.':    # 去除符号
        #     words = words[:-1]

        if max_len > 0:
            if len(words) > max_len:
                words = words[:max_len]
            elif len(words) < max_len:
                words = [PAD_TOKEN] * (max_len - len(words)) + words

        tokens = len(words)
        ids = torch.LongTensor(tokens)
        token = 0
        for word in words:
            if word not in self.dictionary:
                word = UNK_TOKEN
            ids[token] = self.dictionary[word]
            token += 1

        return ids
    
    def untokenize(self, tokens):
        sent = []
        for token in tokens:
            word = self.dictionary[int(token)]
            if word == PAD_TOKEN:
                continue
            else:
                sent.append(word)
        
        return ' '.join(sent)
                
    def __len__(self):
        return len(self.dictionary)
    
    