#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
custom_check.py: custom checks for assignment 5
Usage:
    custom_check.py 1f
    custom_check.py 1g
"""
import json
import math
import pickle
import sys
import time

import numpy as np

from docopt import docopt
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from utils import pad_sents_char, batch_iter, read_corpus
from vocab import Vocab, VocabEntry

from char_decoder import CharDecoder
from highway import Highway
from cnn import CNN

import torch
import torch.nn as nn
import torch.nn.utils

#----------
# CONSTANTS
#----------
BATCH_SIZE = 5
EMBED_SIZE = 3
HIDDEN_SIZE = 4
DROPOUT_RATE = 0.0

class DummyVocab():
    def __init__(self):
        self.char2id = json.load(open('./sanity_check_en_es_data/char_vocab_sanity_check.json', 'r'))
        self.id2char = {id: char for char, id in self.char2id.items()}
        self.char_pad = self.char2id['∏']
        self.char_unk = self.char2id['Û']
        self.start_of_word = self.char2id["{"]
        self.end_of_word = self.char2id["}"]

def question_1f_sanity_check():
    """ Custom check for highway.py
    """
    print ("-"*80)
    print("Running Custom Checks for Question 1f: Highway")
    print ("-"*80)
    inpt = torch.zeros(BATCH_SIZE, EMBED_SIZE, dtype=torch.float)
    highway = Highway(EMBED_SIZE)
    output = highway(inpt)
    output_expected_size = [BATCH_SIZE, EMBED_SIZE]
    assert(list(output.size()) == output_expected_size), "output shape is incorrect: it should be:\n {} but is:\n{}".format(output_expected_size, list(output.size()))
    
    sentence_length = 10
    inpt = torch.zeros(sentence_length, BATCH_SIZE, EMBED_SIZE, dtype=torch.float)
    inpt = inpt.contiguous().view(sentence_length*BATCH_SIZE, EMBED_SIZE)
    highway = Highway(EMBED_SIZE)
    output = highway(inpt)
    output_expected_size = [sentence_length*BATCH_SIZE, EMBED_SIZE]
    assert(list(output.size()) == output_expected_size), "output shape is incorrect: it should be:\n {} but is:\n{}".format(output_expected_size, list(output.size()))
    
    inpt = torch.zeros(BATCH_SIZE, EMBED_SIZE, dtype=torch.float)
    highway = Highway(EMBED_SIZE)
    x_proj, x_gate, x_highway = highway(inpt, testing=True)
    x_proj_expected_size = [BATCH_SIZE, EMBED_SIZE]
    x_gate_expected_size = [BATCH_SIZE, EMBED_SIZE]
    x_highway_expected_size = [BATCH_SIZE, EMBED_SIZE]
    assert(list(x_highway.size()) == x_highway_expected_size), "output shape is incorrect: it should be:\n {} but is:\n{}".format(x_highway_expected_size, list(x_highway.size()))
    assert(list(x_proj.size()) == x_proj_expected_size), "output shape is incorrect: it should be:\n {} but is:\n{}".format(x_proj_expected_size, list(x_proj.size()))
    assert(list(x_gate.size()) == x_gate_expected_size), "output shape is incorrect: it should be:\n {} but is:\n{}".format(x_gate_expected_size, list(x_gate.size()))
    print("All Custom Checks Passed for Question 1f: Highway!")
    print("-"*80)


def question_1g_sanity_check():
    """ Custom check for highway.py
    """
    print ("-"*80)
    print("Running Custom Checks for Question 1g: CNN")
    print ("-"*80)
    e_char = 10
    max_word_length = 21
    k = 5
    inpt = torch.zeros(BATCH_SIZE, e_char, max_word_length, dtype=torch.float)
    cnn = CNN(kernel_size=k, e_char=e_char, f=EMBED_SIZE)
    output = cnn(inpt)
    output_expected_size = [BATCH_SIZE, EMBED_SIZE]
    assert(list(output.size()) == output_expected_size), "output shape is incorrect: it should be:\n {} but is:\n{}".format(output_expected_size, list(output.size()))
    
    sentence_length = 10
    inpt = torch.zeros(sentence_length, BATCH_SIZE, e_char, max_word_length, dtype=torch.float)
    inpt = inpt.contiguous().view(sentence_length*BATCH_SIZE, e_char, max_word_length)
    cnn = CNN(kernel_size=k, e_char=e_char, f=EMBED_SIZE)
    output = cnn(inpt)
    output_expected_size = [sentence_length*BATCH_SIZE, EMBED_SIZE]
    assert(list(output.size()) == output_expected_size), "output shape is incorrect: it should be:\n {} but is:\n{}".format(output_expected_size, list(output.size()))
    
    inpt = torch.zeros(BATCH_SIZE, e_char, max_word_length, dtype=torch.float)
    cnn = CNN(kernel_size=k, e_char=e_char, f=EMBED_SIZE)
    x_conv, x_conv_out = cnn(inpt, testing=True)
    x_conv_expected_size = [BATCH_SIZE, EMBED_SIZE, max_word_length-k+3]
    x_conv_out_expected_size = [BATCH_SIZE, EMBED_SIZE]
    assert(list(x_conv.size()) == x_conv_expected_size), "output shape is incorrect: it should be:\n {} but is:\n{}".format(x_conv_expected_size, list(x_conv.size()))
    assert(list(x_conv_out.size()) == x_conv_out_expected_size), "output shape is incorrect: it should be:\n {} but is:\n{}".format(x_conv_out_expected_size, list(x_conv_out.size()))
    print("All Custom Checks Passed for Question 1g: CNN!")
    print("-"*80)


def main():
    """ Main func.
    """
    args = docopt(__doc__)

    # Check Python & PyTorch Versions
    assert (sys.version_info >= (3, 5)), "Please update your installation of Python to version >= 3.5"
    assert(torch.__version__ >= "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(torch.__version__)

    # Seed the Random Number Generators
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    vocab = Vocab.load('./sanity_check_en_es_data/vocab_sanity_check.json') 

    char_vocab = DummyVocab()

    if args['1f']:
        question_1f_sanity_check()
    elif args['1g']:
        question_1g_sanity_check()
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
