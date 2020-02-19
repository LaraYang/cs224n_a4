#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import numpy as np
class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()
        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(self.target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(len(self.target_vocab.char2id), char_embedding_size, padding_idx=self.target_vocab.char_pad)

    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input (Tensor): tensor of integers, shape (length, batch_size)
        @param dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores (Tensor): called s_t in the PDF, shape (length, batch_size, self.vocab_size)
        @returns dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Implement the forward pass of the character decoder.
        X = self.decoderCharEmb(input)
        output, dec_hidden = self.charDecoder(X, dec_hidden)
        scores = self.char_output_projection(output)
        return (scores, dec_hidden)
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence (Tensor): tensor of integers, shape (length, batch_size). Note that "length" here and in forward() need not be the same.
        @param dec_hidden (tuple(Tensor, Tensor)): initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch_size, hidden_size)

        @returns The cross-entropy loss (Tensor), computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss. Check vocab.py to find the padding token's index.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} (e.g., <START>,m,u,s,i,c,<END>). Read the handout about how to construct input and target sequence of CharDecoderLSTM.
        ###       - Carefully read the documentation for nn.CrossEntropyLoss and our handout to see what this criterion have already included:
        ###             https://pytorch.org/docs/stable/nn.html#crossentropyloss
        loss = nn.CrossEntropyLoss(ignore_index=self.target_vocab.char_pad, reduction='sum')
        input_seq = char_sequence[:-1]
        target_seq_chars = char_sequence[1:]
        
        scores, dec_hidden = self.forward(input_seq, dec_hidden)
        # shape (batch_size, length-1, v_char)
        scores = torch.transpose(scores, 0, 1).contiguous()
        # shape (batch_size * (length-1), v_char)
        scores = torch.flatten(scores, start_dim=0, end_dim=1)
        
        # shape (batch_size, length-1)
        target_seq_chars = torch.transpose(target_seq_chars, 0, 1).contiguous()
        target_seq_chars = torch.flatten(target_seq_chars)
        output = loss(scores, target_seq_chars)
        return output
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates (tuple(Tensor, Tensor)): initial internal state of the LSTM, a tuple of two tensors of size (1, batch_size, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length (int): maximum length of words to decode

        @returns decodedWords (List[str]): a list (of length batch_size) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2c
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use initialStates to get batch_size.
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - You may find torch.argmax or torch.argmax useful
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        batch_size = initialStates[0].shape[1]
        dec_hidden = initialStates
        starting_chars = [self.target_vocab.start_of_word for i in range(batch_size)]
        decodedWords = []
        current_chars = torch.tensor(starting_chars, device=device)
        current_chars = torch.unsqueeze(current_chars, 0)
        softmax = nn.Softmax(dim=-1)
        output_words = None

        for i in range(max_length):
            scores, dec_hidden = self.forward(current_chars, dec_hidden)
            probs = softmax(scores)
            current_chars = torch.argmax(probs, dim=-1, keepdim=False)
            if i == 0:
                output_words = current_chars
            else:
                output_words = torch.cat((output_words, current_chars))

        decoded_char_ids = torch.transpose(output_words, 0, 1).flatten().tolist()
        decoded_char_ids = np.reshape(decoded_char_ids, (batch_size, max_length))
        for batch in decoded_char_ids:
            word = ''
            for char_id in batch:
                if char_id == self.target_vocab.end_of_word:
                    break
                elif char_id == self.target_vocab.start_of_word:
                    continue
                else:
                    word += self.target_vocab.id2char[char_id]
            decodedWords.append(word)
        return decodedWords
        ### END YOUR CODE

