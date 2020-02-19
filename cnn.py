#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    ### YOUR CODE HERE for part 1g
    def __init__(self, kernel_size, e_char, f):
        """
        @params kernel_size (int): window size dictating size of window for computing features
        @params e_char (int): e_char, the length of the dense character embedding
        @params f (int): number of filters or number of output features, which is equal to e_word in our case
        """
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=e_char, out_channels=f, kernel_size=kernel_size, padding=1, bias=True)
    
    def forward(self, x_reshaped, testing=False) -> torch.Tensor:
        """
        @params x_reshaped (Tensor): shape (batch_size, e_char, m_word)
        @returns x_conv_out (Tensor): shape (batch_size, e_word)
        """
        # should have shape (batch_size, e_word, m_word-kernel_size+3)
        x_conv = self.conv(x_reshaped)
        # 1d max pooling over the last dimension of x_conv
        x_conv_out =  torch.max(nn.functional.relu(x_conv), -1)[0]
        if testing:
        	return (x_conv, x_conv_out)
        return x_conv_out
        ### END YOUR CODE

