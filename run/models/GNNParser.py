import functools
import time
from collections import defaultdict

import _dynet as dy
import numpy as np

from lib import *
from models import *


class GNNParser(BaseParser):
    '''
        H = LSTM(sentence)
        G = H * W_G * H^T
        H2 = GCN(G, H)
        label = f(H2)
    '''

    def __init__(self, model, parameters):
        super().__init__(model, *parameters)
        self.spec = {'parameters': parameters}

    def get_span_encoding(self, H, left, right):
        span_encoding = H[left] + H[right]
        return span_encoding

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

        H = dy.transpose(dy.concatenate(lstm_outputs[1:-1], d=1))
        G_predict = dy.logistic(H * self.W_G * dy.transpose(H))

        def helper(G):
            G_np = self.gen_legal_graph(G)
            H_2 = self.GCN(G_np, H)

            stack = []
            total_label_loss = 0

            for right in range(len(sentence)):
                for left in range(right, -1, -1):
                    if G_np[left][right] > 0.8:
                        label_scores = self.f_label(
                            self.get_span_encoding(H_2, left, right))

                        if is_train:
                            oracle_label, oracle_label_index, crossing = self.get_oracle_label(
                                gold_tree, left, right)
                            label_scores = self.augment(
                                label_scores, oracle_label_index, crossing)

                        argmax_label, argmax_label_index = self.predict_label(
                            label_scores, gold_tree, left, right)

                        if is_train:
                            label_loss = (
                                label_scores[argmax_label_index] -
                                label_scores[oracle_label_index]
                                if argmax_label != oracle_label else dy.zeros(1))
                            total_label_loss += label_loss

                        label = argmax_label

                        if left == right:
                            tree = self.gen_leaf_tree(left, label)
                            stack.append((left, right, [tree]))
                            continue

                        tmp = []
                        while len(stack) > 0 and stack[-1][0] >= left:
                            tmp.append(stack.pop()[2])
                        childrens = []
                        for c in tmp[::-1]:
                            childrens += c

                        childrens = self.gen_nonleaf_tree(childrens, label)
                        stack.append((left, right, childrens))

            assert len(
                stack) == 1 and stack[-1][0] == 0 and stack[-1][1] == len(sentence) - 1
            childrens = stack[-1][2]
            assert len(childrens) == 1

            return childrens[0], total_label_loss

        tree, l_loss = helper(G_predict)
        tree.propagate_sentence(sentence)

        if is_train:
            G_oracle = self.tree2graph(gold_tree)
            s_loss = dy.binary_log_loss(G_predict, G_oracle)
            loss = s_loss + l_loss
            return loss, tree, 1
        else:
            return tree


class GNNParser1(BaseParser):
    '''
        H = LSTM(sentence)
        G = H * W_G * H^T
        label = f(H)
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

        H = dy.transpose(dy.concatenate(lstm_outputs[1:-1], d=1))
        G_predict = dy.logistic(H * self.W_G * dy.transpose(H))

        def helper(G):
            G_np = self.gen_legal_graph(G)

            stack = []
            total_label_loss = 0

            for right in range(len(sentence)):
                for left in range(right, -1, -1):
                    if G_np[left][right] > 0.8:
                        label_scores = self.f_label(
                            self.get_span_encoding(lstm_outputs, left, right))

                        if is_train:
                            oracle_label, oracle_label_index, crossing = self.get_oracle_label(
                                gold_tree, left, right)
                            label_scores = self.augment(
                                label_scores, oracle_label_index, crossing)

                        argmax_label, argmax_label_index = self.predict_label(
                            label_scores, gold_tree, left, right)

                        if is_train:
                            label_loss = (
                                label_scores[argmax_label_index] -
                                label_scores[oracle_label_index]
                                if argmax_label != oracle_label else dy.zeros(1))
                            total_label_loss += label_loss

                        label = argmax_label

                        if left == right:
                            tree = self.gen_leaf_tree(left, label)
                            stack.append((left, right, [tree]))
                            continue

                        tmp = []
                        while len(stack) > 0 and stack[-1][0] >= left:
                            tmp.append(stack.pop()[2])
                        childrens = []
                        for c in tmp[::-1]:
                            childrens += c

                        childrens = self.gen_nonleaf_tree(childrens, label)
                        stack.append((left, right, childrens))

            assert len(
                stack) == 1 and stack[-1][0] == 0 and stack[-1][1] == len(sentence) - 1
            childrens = stack[-1][2]
            assert len(childrens) == 1

            return childrens[0], total_label_loss

        tree, l_loss = helper(G_predict)
        tree.propagate_sentence(sentence)

        if is_train:
            G_oracle = self.tree2graph(gold_tree)
            s_loss = dy.binary_log_loss(G_predict, G_oracle)
            loss = s_loss + l_loss
            return loss, tree, 1
        else:
            return tree
