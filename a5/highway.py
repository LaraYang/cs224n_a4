#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Set, Union

class Highway(nn.Module):
    ### YOUR CODE HERE for part 1f
    def __init__(self, e_word):
        super(Highway, self).__init__()
        self.e_word = e_word
        
        self.proj = nn.Linear(e_word, e_word, bias=True)
        self.gate = nn.Linear(e_word, e_word, bias=True)
        return


    def forward(self, x_conv_out, testing=False) -> torch.Tensor:
        """
        @params x_conv_out (Tensor): shape (batch_size, e_word)
        @returns x_highway (Tensor): shape (batch_size, e_word)
        """
        x_proj = nn.functional.relu(self.proj(x_conv_out))
        x_gate = torch.sigmoid(self.gate(x_conv_out))
        x_highway = torch.mul(x_gate, x_proj) + torch.mul((torch.ones(x_gate.shape[0], x_gate.shape[1])-x_gate), x_conv_out)
        if testing:
            return (x_proj, x_gate, x_highway)
        return x_highway
    ### END YOUR CODE

