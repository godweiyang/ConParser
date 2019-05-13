import functools
import time

import _dynet as dy
import numpy as np

from lib import *
from models import *


class TopDownParser(BaseParser):
    '''
    TopDown greedy parser from stern 2017.
    label prediction: lstm(l, r)
    split prediction: lstm(l, r)
    '''

    def __init__(self, model, parameters):
        super().__init__(model, *parameters)
        self.spec = {'parameters': parameters}

    def parse(self, data, is_train=False):
        if is_train:
            self.lstm.set_dropout(self.dropout)
        else:
            self.lstm.disable_dropout()

        word_indices = data['w']
        tag_indices = data['t']
        gold_tree = data['tree']
        sentence = gold_tree.sentence

        embeddings = self.get_embeddings(word_indices, tag_indices, is_train)
        lstm_outputs = self.lstm.transduce(embeddings)

        def helper(left, right):
            label_scores = self.f_label(
                self.get_span_encoding(lstm_outputs, left, right))
            if is_train:
                oracle_label, oracle_label_index, crossing = self.get_oracle_label(
                    gold_tree, left, right)
                label_scores = self.augment(
                    label_scores, oracle_label_index, crossing)
            argmax_label, argmax_label_index = self.predict_label(
                label_scores, gold_tree, left, right)
            label = argmax_label

            if is_train:
                label_loss = (
                    label_scores[argmax_label_index] -
                    label_scores[oracle_label_index]
                    if argmax_label != oracle_label else dy.zeros(1))

            if left == right:
                tree = self.gen_leaf_tree(left, label)
                return [tree], label_loss if is_train else None

            split_scores = self.get_split_scores(lstm_outputs, left, right)
            if is_train:
                oracle_splits = gold_tree.span_splits(left, right)
                oracle_split = min(oracle_splits)
                oracle_split_index = oracle_split - (left + 1)
                split_scores = self.augment(split_scores, oracle_split_index)

            split_scores_np = split_scores.npvalue()
            argmax_split_index = int(split_scores_np.argmax())
            argmax_split = argmax_split_index + (left + 1)
            split = argmax_split

            if is_train:
                split_loss = (
                    split_scores[argmax_split_index] -
                    split_scores[oracle_split_index]
                    if argmax_split != oracle_split else dy.zeros(1))

            left_trees, left_loss = helper(left, split - 1)
            right_trees, right_loss = helper(split, right)

            childrens = left_trees + right_trees
            childrens = self.gen_nonleaf_tree(childrens, label)

            return childrens, label_loss + split_loss + left_loss + right_loss if is_train else None

        childrens, loss = helper(0, len(sentence) - 1)
        assert len(childrens) == 1
        tree = childrens[0]
        tree.propagate_sentence(sentence)

        if is_train:
            return loss, tree, 1
        else:
            return tree
