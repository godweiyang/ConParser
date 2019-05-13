import functools
import time

import _dynet as dy
import numpy as np

from lib import *
from models import *


class ChartParser(BaseParser):
    '''
    chart parser from stern 2017.
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

        def helper(force_gold):
            if force_gold:
                assert is_train

            chart = {}

            for length in range(1, len(sentence) + 1):
                for left in range(0, len(sentence) + 1 - length):
                    right = left + length - 1

                    label_scores = self.f_label(
                        self.get_span_encoding(lstm_outputs, left, right))
                    if is_train:
                        oracle_label, oracle_label_index, crossing = self.get_oracle_label(
                            gold_tree, left, right)
                    if force_gold:
                        label = oracle_label
                        label_score = label_scores[oracle_label_index]
                    else:
                        if is_train:
                            label_scores = self.augment(
                                label_scores, oracle_label_index, crossing)

                        argmax_label, argmax_label_index = self.predict_label(
                            label_scores, gold_tree, left, right)

                        label = argmax_label
                        label_score = label_scores[argmax_label_index]

                    if length == 1:
                        tree = self.gen_leaf_tree(left, label)
                        chart[left, right] = [tree], label_score
                        continue

                    if force_gold:
                        oracle_splits = gold_tree.span_splits(left, right)
                        oracle_split = min(oracle_splits)
                        best_split = oracle_split
                    else:
                        best_split = max(
                            range(left + 1, right + 1),
                            key=lambda split:
                                chart[left, split - 1][1].value() +
                                chart[split, right][1].value())

                    left_trees, left_score = chart[left, best_split - 1]
                    right_trees, right_score = chart[best_split, right]

                    childrens = left_trees + right_trees
                    childrens = self.gen_nonleaf_tree(childrens, label)

                    chart[left, right] = (
                        childrens, label_score + left_score + right_score)

            childrens, score = chart[0, len(sentence) - 1]
            assert len(childrens) == 1
            return childrens[0], score

        tree, score = helper(False)
        tree.propagate_sentence(sentence)

        if is_train:
            oracle_tree, oracle_score = helper(True)
            oracle_tree.propagate_sentence(sentence)

            assert str(oracle_tree) == str(gold_tree)

            correct = (str(tree) == str(oracle_tree))
            loss = dy.zeros(1) if correct else score - oracle_score

            return loss, tree, 1
        else:
            return tree
