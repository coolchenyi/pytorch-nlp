# -*- coding: utf-8 -*-

import torch
import re
import os
import unicodedata

from nlp.chatbot.config import MAX_LENGTH

SOS_token = 0
EOS_token = 1
PAD_token = 2


class Voc(object):
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2:"PAD"}
        self.n_words = 3  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def read_vocs(corpus, corpus_name):
    print("Reading lines...")
    # combine every two lines into pairs and normalize
    with open(corpus) as f:
        content = f.readlines()

    lines = [x.strip() for x in content]
    it = iter(lines)
    i = 0
    pairs = []
    for line in it:
        if i % 3 == 0:
            i += 1
            continue
        else:
            pairs.append([" ".join(line.split()[1].split('/')), " ".join(next(it).split()[1].split('/'))])
            i += 2
    voc = Voc(corpus_name)
    return voc, pairs


def filter_pair(p):
    # input sequences need to preserve the last word for EOS_token
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(corpus, corpus_name, save_dir):
    voc, pairs = read_vocs(corpus, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filter_pairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.add_sentence(pair[0])
        voc.add_sentence(pair[1])
    print("Counted words:", voc.n_words)
    directory = os.path.join(save_dir, 'training_data', corpus_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(voc, os.path.join(directory, '{!s}.tar'.format('voc')))
    torch.save(pairs, os.path.join(directory, '{!s}.tar'.format('pairs')))
    return voc, pairs


def load_prepare_data(corpus, save_dir):
    corpus_name = corpus.split('/')[-1].split('.')[0]
    try:
        print("Start loading training data ...")
        voc = torch.load(os.path.join(save_dir, 'training_data', corpus_name, 'voc.tar'))
        pairs = torch.load(os.path.join(save_dir, 'training_data', corpus_name, 'pairs.tar'))
    except FileNotFoundError:
        print("Saved data not found, start preparing training data ...")
        voc, pairs = prepare_data(corpus, corpus_name, save_dir)
    return voc, pairs