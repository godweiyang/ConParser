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

    def parse(self, data, is_train=False):
        # if is_train:
        #     self.lstm.set_dropout(self.dropout)
        # else:
        #     self.lstm.disable_dropout()

        # word_indices = data['w']
        # tag_indices = data['t']
        # gold_tree = data['tree']
        # sentence = gold_tree.sentence

        # embeddings = self.get_embeddings(word_indices, tag_indices, is_train)
        # lstm_outputs = self.lstm.transduce(embeddings)

        # def helper(left, split, right_bound, left_trees=None, left_loss=None):
        #     if left == split:
        #         label_scores = self.f_label(
        #             self.get_span_encoding(lstm_outputs, left, split))
        #         if is_train:
        #             oracle_label, oracle_label_index, crossing = self.get_oracle_label(
        #                 gold_tree, left, split)
        #             label_scores = self.augment(
        #                 label_scores, oracle_label_index, crossing)
        #         argmax_label, argmax_label_index = self.predict_label(
        #             label_scores, gold_tree, left, split)

        #         if is_train:
        #             left_loss = (
        #                 label_scores[argmax_label_index] -
        #                 label_scores[oracle_label_index]
        #                 if argmax_label != oracle_label else dy.zeros(1))

        #         left_label = argmax_label
        #         left_trees = self.gen_leaf_tree(left, left_label)
        #         left_trees = [left_trees]

        #     if split == right_bound:
        #         return left_trees, left_loss

        #     parent_right_scores = self.get_right_boundary_scores(
        #         lstm_outputs, left, split, right_bound)
        #     if is_train:
        #         oracle_rights = gold_tree.parent_rights(
        #             left, split, right_bound)
        #         oracle_right = max(oracle_rights)
        #         oracle_right_index = oracle_right - (split + 1)
        #         parent_right_scores = self.augment(
        #             parent_right_scores, oracle_right_index)
        #     parent_right_scores_np = parent_right_scores.npvalue()
        #     argmax_right_index = int(parent_right_scores_np.argmax())
        #     argmax_right = argmax_right_index + (split + 1)

        #     right = argmax_right

        #     label_scores = self.f_label(
        #         self.get_span_encoding(lstm_outputs, left, right))
        #     if is_train:
        #         oracle_label, oracle_label_index, crossing = self.get_oracle_label(
        #             gold_tree, left, right)
        #         label_scores = self.augment(
        #             label_scores, oracle_label_index, crossing)
        #     argmax_label, argmax_label_index = self.predict_label(
        #         label_scores, gold_tree, left, right)
        #     label = argmax_label

        #     if is_train:
        #         parent_loss = (
        #             parent_right_scores[argmax_right_index] -
        #             parent_right_scores[oracle_right_index]
        #             if argmax_right != oracle_right else dy.zeros(1))
        #         label_loss = (
        #             label_scores[argmax_label_index] -
        #             label_scores[oracle_label_index]
        #             if argmax_label != oracle_label else dy.zeros(1))

        #     right_trees, right_loss = helper(
        #         split + 1, split + 1, right, None, None)

        #     childrens = left_trees + right_trees
        #     childrens = self.gen_nonleaf_tree(childrens, label)

        #     tree, loss = helper(left, right, right_bound, childrens, (
        #         parent_loss + label_loss + left_loss + right_loss) if is_train else None)

        #     return tree, loss

        # childrens, loss = helper(0, 0, len(sentence) - 1, None, None)
        # assert len(childrens) == 1
        # tree = childrens[0]
        # tree.propagate_sentence(sentence)

        # if is_train:
        #     return loss, tree, 1
        # else:
        #     return tree
