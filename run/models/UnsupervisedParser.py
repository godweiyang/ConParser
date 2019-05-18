import functools
import time

import numpy as np
import _dynet as dy

from lib import *
from models import *


class UnsupervisedParser(BaseParser):
    '''
        Unsupervised parser
    '''

    def __init__(self, model, parameters):
        super().__init__(model, *parameters)
        self.spec = {'parameters': parameters}

        self.uni_lstm = dy.VanillaLSTMBuilder(
            2, self.word_embedding_dim, self.lstm_dim, self.model)
        self.W = self.model.add_parameters(
            (self.word_count, self.lstm_dim))
        self.b = self.model.add_parameters((self.word_count, ))
        self.f_syn_dis = Feedforward(
            self.model, self.lstm_dim, [self.fc_hidden_dim], 1)

    def parse(self, data, is_train=False):
        if is_train:
            self.uni_lstm.set_dropout(self.dropout)
        else:
            self.uni_lstm.disable_dropout()

        word_indices = data['w']
        # tag_indices = data['t']
        # gold_tree = data['tree']
        # sentence = gold_tree.sentence

        # embeddings = self.get_embeddings(word_indices, tag_indices, is_train)

        state = self.uni_lstm.initial_state()
        errs = []

        syntactic_distance = []
        for w_s, w_t in zip(word_indices[:-1], word_indices[1:-1]):
            word_embedding = self.word_embeddings[w_s]
            state = state.add_input(word_embedding)
            h_t = state.output()
            if is_train:
                h_t = dy.dropout(h_t, self.dropout)
            syn_dis = self.f_syn_dis(h_t).value()
            syntactic_distance.append(syn_dis.value())
            scores = self.W * h_t + self.b
            err = dy.pickneglogsoftmax(scores, int(w_t))
            errs.append(err)

        span_set = self.gen_tree_syn_dis(
            syntactic_distance[1:], 0, len(word_indices) - 3)
        sum_errs = dy.esum(errs)
        if is_train:
            return sum_errs, None, 1
        else:
            return [sum_errs, span_set]
