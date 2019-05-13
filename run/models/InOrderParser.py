import functools
import time

import numpy as np
import _dynet as dy

from lib import *
from models import *


class InOrderParser(BaseParser):
    '''
        InOrder greedy parser
        label prediction: lstm(l, r)
        right boundary point prediction: lstm(l, r)
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

        def helper(left, split, right_bound, left_trees=None, left_loss=None):
            if left == split:
                label_scores = self.f_label(
                    self.get_span_encoding(lstm_outputs, left, split))
                if is_train:
                    oracle_label, oracle_label_index, crossing = self.get_oracle_label(
                        gold_tree, left, split)
                    label_scores = self.augment(
                        label_scores, oracle_label_index, crossing)
                argmax_label, argmax_label_index = self.predict_label(
                    label_scores, gold_tree, left, split)

                if is_train:
                    left_loss = (
                        label_scores[argmax_label_index] -
                        label_scores[oracle_label_index]
                        if argmax_label != oracle_label else dy.zeros(1))

                left_label = argmax_label
                left_trees = self.gen_leaf_tree(left, left_label)
                left_trees = [left_trees]

            if split == right_bound:
                return left_trees, left_loss

            parent_right_scores = self.get_right_boundary_scores(
                lstm_outputs, left, split, right_bound)
            if is_train:
                oracle_rights = gold_tree.parent_rights(
                    left, split, right_bound)
                oracle_right = max(oracle_rights)
                oracle_right_index = oracle_right - (split + 1)
                parent_right_scores = self.augment(
                    parent_right_scores, oracle_right_index)
            parent_right_scores_np = parent_right_scores.npvalue()
            argmax_right_index = int(parent_right_scores_np.argmax())
            argmax_right = argmax_right_index + (split + 1)

            right = argmax_right

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
                parent_loss = (
                    parent_right_scores[argmax_right_index] -
                    parent_right_scores[oracle_right_index]
                    if argmax_right != oracle_right else dy.zeros(1))
                label_loss = (
                    label_scores[argmax_label_index] -
                    label_scores[oracle_label_index]
                    if argmax_label != oracle_label else dy.zeros(1))

            right_trees, right_loss = helper(
                split + 1, split + 1, right, None, None)

            childrens = left_trees + right_trees
            childrens = self.gen_nonleaf_tree(childrens, label)

            tree, loss = helper(left, right, right_bound, childrens, (
                parent_loss + label_loss + left_loss + right_loss) if is_train else None)

            return tree, loss

        childrens, loss = helper(0, 0, len(sentence) - 1, None, None)
        assert len(childrens) == 1
        tree = childrens[0]
        tree.propagate_sentence(sentence)

        if is_train:
            return loss, tree, 1
        else:
            return tree


class InOrderParser1(BaseParser):
    '''
        InOrder greedy parser
        historical_info_lstm:
            - without branch
            - input = label_embedding
        label prediction: [lstm(l, r); historical_info_lstm(t)]
        right boundary point prediction: lstm(l, r)
    '''

    def __init__(self, model, parameters):
        super().__init__(model, *parameters)
        self.spec = {'parameters': parameters}

        self.f_label = Feedforward(
            self.model, 3 * self.lstm_dim, [self.fc_hidden_dim], self.label_out)

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

        def helper(left, split, right_bound, child_state, left_trees=None, left_loss=None):
            if left == split:
                h_hat = child_state.output()
                span_encoding = dy.concatenate(
                    [self.get_span_encoding(lstm_outputs, left, split), h_hat])
                label_scores = self.f_label(span_encoding)
                if is_train:
                    oracle_label, oracle_label_index, crossing = self.get_oracle_label(
                        gold_tree, left, split)
                    label_scores = self.augment(
                        label_scores, oracle_label_index, crossing)
                argmax_label, argmax_label_index = self.predict_label(
                    label_scores, gold_tree, left, split)

                if is_train:
                    left_loss = (
                        label_scores[argmax_label_index] -
                        label_scores[oracle_label_index]
                        if argmax_label != oracle_label else dy.zeros(1))

                left_label = argmax_label
                left_trees = self.gen_leaf_tree(left, left_label)
                left_trees = [left_trees]
                child_state = child_state.add_input(
                    self.label_embeddings[argmax_label_index])

            if split == right_bound:
                return left_trees, left_loss, child_state

            parent_right_scores = self.get_right_boundary_scores(
                lstm_outputs, left, split, right_bound)
            if is_train:
                oracle_rights = gold_tree.parent_rights(
                    left, split, right_bound)
                oracle_right = max(oracle_rights)
                oracle_right_index = oracle_right - (split + 1)
                parent_right_scores = self.augment(
                    parent_right_scores, oracle_right_index)
            parent_right_scores_np = parent_right_scores.npvalue()
            argmax_right_index = int(parent_right_scores_np.argmax())
            argmax_right = argmax_right_index + (split + 1)

            right = argmax_right

            h_hat = child_state.output()
            span_encoding = dy.concatenate(
                [self.get_span_encoding(lstm_outputs, left, right), h_hat])
            label_scores = self.f_label(span_encoding)
            if is_train:
                oracle_label, oracle_label_index, crossing = self.get_oracle_label(
                    gold_tree, left, right)
                label_scores = self.augment(
                    label_scores, oracle_label_index, crossing)
            argmax_label, argmax_label_index = self.predict_label(
                label_scores, gold_tree, left, right)
            label = argmax_label
            child_state = child_state.add_input(
                self.label_embeddings[argmax_label_index])

            if is_train:
                parent_loss = (
                    parent_right_scores[argmax_right_index] -
                    parent_right_scores[oracle_right_index]
                    if argmax_right != oracle_right else dy.zeros(1))
                label_loss = (
                    label_scores[argmax_label_index] -
                    label_scores[oracle_label_index]
                    if argmax_label != oracle_label else dy.zeros(1))

            right_trees, right_loss, right_state = helper(
                split + 1, split + 1, right, child_state, None, None)

            childrens = left_trees + right_trees
            childrens = self.gen_nonleaf_tree(childrens, label)

            tree, loss, parent_state = helper(left, right, right_bound, right_state, childrens, (
                parent_loss + label_loss + left_loss + right_loss) if is_train else None)

            return tree, loss, parent_state

        leaf_state = self.historical_info_lstm.initial_state()
        leaf_state = leaf_state.add_input(
            self.label_embeddings[self.label_out])
        childrens, loss, _ = helper(
            0, 0, len(sentence) - 1, leaf_state, None, None)
        assert len(childrens) == 1
        tree = childrens[0]
        tree.propagate_sentence(sentence)

        if is_train:
            return loss, tree, 1
        else:
            return tree


class InOrderParser2(BaseParser):
    '''
        InOrder greedy parser
        historical_info_lstm:
            - with branch
            - input = label_embedding
        label prediction: [lstm(l, r); historical_info_lstm(t)]
        right boundary point prediction: lstm(l, r)
    '''

    def __init__(self, model, parameters):
        super().__init__(model, *parameters)
        self.spec = {'parameters': parameters}

        self.f_label = Feedforward(
            self.model, 3 * self.lstm_dim, [self.fc_hidden_dim], self.label_out)

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

        def helper(left, split, right_bound, child_state, left_trees=None, left_loss=None):
            if left == split:
                h_hat = child_state.output()
                span_encoding = dy.concatenate(
                    [self.get_span_encoding(lstm_outputs, left, split), h_hat])
                label_scores = self.f_label(span_encoding)
                if is_train:
                    oracle_label, oracle_label_index, crossing = self.get_oracle_label(
                        gold_tree, left, split)
                    label_scores = self.augment(
                        label_scores, oracle_label_index, crossing)
                argmax_label, argmax_label_index = self.predict_label(
                    label_scores, gold_tree, left, split)

                if is_train:
                    left_loss = (
                        label_scores[argmax_label_index] -
                        label_scores[oracle_label_index]
                        if argmax_label != oracle_label else dy.zeros(1))

                left_label = argmax_label
                left_trees = self.gen_leaf_tree(left, left_label)
                left_trees = [left_trees]
                child_state = child_state.add_input(
                    self.label_embeddings[argmax_label_index])

            if split == right_bound:
                return left_trees, left_loss, child_state

            parent_right_scores = self.get_right_boundary_scores(
                lstm_outputs, left, split, right_bound)
            if is_train:
                oracle_rights = gold_tree.parent_rights(
                    left, split, right_bound)
                oracle_right = max(oracle_rights)
                oracle_right_index = oracle_right - (split + 1)
                parent_right_scores = self.augment(
                    parent_right_scores, oracle_right_index)
            parent_right_scores_np = parent_right_scores.npvalue()
            argmax_right_index = int(parent_right_scores_np.argmax())
            argmax_right = argmax_right_index + (split + 1)

            right = argmax_right

            h_hat = child_state.output()
            span_encoding = dy.concatenate(
                [self.get_span_encoding(lstm_outputs, left, right), h_hat])
            label_scores = self.f_label(span_encoding)
            if is_train:
                oracle_label, oracle_label_index, crossing = self.get_oracle_label(
                    gold_tree, left, right)
                label_scores = self.augment(
                    label_scores, oracle_label_index, crossing)
            argmax_label, argmax_label_index = self.predict_label(
                label_scores, gold_tree, left, right)
            label = argmax_label
            child_state = child_state.add_input(
                self.label_embeddings[argmax_label_index])

            if is_train:
                parent_loss = (
                    parent_right_scores[argmax_right_index] -
                    parent_right_scores[oracle_right_index]
                    if argmax_right != oracle_right else dy.zeros(1))
                label_loss = (
                    label_scores[argmax_label_index] -
                    label_scores[oracle_label_index]
                    if argmax_label != oracle_label else dy.zeros(1))

            right_trees, right_loss, right_state = helper(
                split + 1, split + 1, right, child_state, None, None)

            childrens = left_trees + right_trees
            childrens = self.gen_nonleaf_tree(childrens, label)

            tree, loss, parent_state = helper(left, right, right_bound, child_state, childrens, (
                parent_loss + label_loss + left_loss + right_loss) if is_train else None)

            return tree, loss, parent_state

        leaf_state = self.historical_info_lstm.initial_state()
        leaf_state = leaf_state.add_input(
            self.label_embeddings[self.label_out])
        childrens, loss, _ = helper(
            0, 0, len(sentence) - 1, leaf_state, None, None)
        assert len(childrens) == 1
        tree = childrens[0]
        tree.propagate_sentence(sentence)

        if is_train:
            return loss, tree, 1
        else:
            return tree


class InOrderParser3(BaseParser):
    '''
        InOrder greedy parser
        historical_info_lstm:
            - with branch
            - input = [label_embedding; lstm(l, r)]
        label prediction: [lstm(l, r); historical_info_lstm(t)]
        right boundary point prediction: lstm(l, r)
    '''

    def __init__(self, model, parameters):
        super().__init__(model, *parameters)
        self.spec = {'parameters': parameters}

        self.historical_info_lstm = dy.VanillaLSTMBuilder(
            2, self.label_embedding_dim + 2 * self.lstm_dim, self.lstm_dim, self.model)

        self.f_label = Feedforward(
            self.model, 3 * self.lstm_dim, [self.fc_hidden_dim], self.label_out)

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

        def helper(left, split, right_bound, child_state, left_trees=None, left_loss=None):
            if left == split:
                h_hat = child_state.output()
                span_encoding = dy.concatenate(
                    [self.get_span_encoding(lstm_outputs, left, split), h_hat])
                label_scores = self.f_label(span_encoding)
                if is_train:
                    oracle_label, oracle_label_index, crossing = self.get_oracle_label(
                        gold_tree, left, split)
                    label_scores = self.augment(
                        label_scores, oracle_label_index, crossing)
                argmax_label, argmax_label_index = self.predict_label(
                    label_scores, gold_tree, left, split)

                if is_train:
                    left_loss = (
                        label_scores[argmax_label_index] -
                        label_scores[oracle_label_index]
                        if argmax_label != oracle_label else dy.zeros(1))

                left_label = argmax_label
                left_trees = self.gen_leaf_tree(left, left_label)
                left_trees = [left_trees]
                child_state = child_state.add_input(
                    dy.concatenate([self.label_embeddings[argmax_label_index], self.get_span_encoding(lstm_outputs, left, split)]))

            if split == right_bound:
                return left_trees, left_loss, child_state

            parent_right_scores = self.get_right_boundary_scores(
                lstm_outputs, left, split, right_bound)
            if is_train:
                oracle_rights = gold_tree.parent_rights(
                    left, split, right_bound)
                oracle_right = max(oracle_rights)
                oracle_right_index = oracle_right - (split + 1)
                parent_right_scores = self.augment(
                    parent_right_scores, oracle_right_index)
            parent_right_scores_np = parent_right_scores.npvalue()
            argmax_right_index = int(parent_right_scores_np.argmax())
            argmax_right = argmax_right_index + (split + 1)

            right = argmax_right

            h_hat = child_state.output()
            span_encoding = dy.concatenate(
                [self.get_span_encoding(lstm_outputs, left, right), h_hat])
            label_scores = self.f_label(span_encoding)
            if is_train:
                oracle_label, oracle_label_index, crossing = self.get_oracle_label(
                    gold_tree, left, right)
                label_scores = self.augment(
                    label_scores, oracle_label_index, crossing)
            argmax_label, argmax_label_index = self.predict_label(
                label_scores, gold_tree, left, right)
            label = argmax_label
            child_state = child_state.add_input(
                dy.concatenate([self.label_embeddings[argmax_label_index], self.get_span_encoding(lstm_outputs, left, right)]))

            if is_train:
                parent_loss = (
                    parent_right_scores[argmax_right_index] -
                    parent_right_scores[oracle_right_index]
                    if argmax_right != oracle_right else dy.zeros(1))
                label_loss = (
                    label_scores[argmax_label_index] -
                    label_scores[oracle_label_index]
                    if argmax_label != oracle_label else dy.zeros(1))

            right_trees, right_loss, right_state = helper(
                split + 1, split + 1, right, child_state, None, None)

            childrens = left_trees + right_trees
            childrens = self.gen_nonleaf_tree(childrens, label)

            tree, loss, parent_state = helper(left, right, right_bound, child_state, childrens, (
                parent_loss + label_loss + left_loss + right_loss) if is_train else None)

            return tree, loss, parent_state

        leaf_state = self.historical_info_lstm.initial_state()
        leaf_state = leaf_state.add_input(
            dy.concatenate([self.label_embeddings[self.label_out], dy.zeros(2 * self.lstm_dim)]))
        childrens, loss, _ = helper(
            0, 0, len(sentence) - 1, leaf_state, None, None)
        assert len(childrens) == 1
        tree = childrens[0]
        tree.propagate_sentence(sentence)

        if is_train:
            return loss, tree, 1
        else:
            return tree


class InOrderParser4(BaseParser):
    '''
        InOrder greedy parser
        historical_info_lstm:
            - with branch
            - input = lstm(l, r)
        label prediction: [lstm(l, r); historical_info_lstm(t)]
        right boundary point prediction: lstm(l, r)
    '''

    def __init__(self, model, parameters):
        super().__init__(model, *parameters)
        self.spec = {'parameters': parameters}

        self.historical_info_lstm = dy.VanillaLSTMBuilder(
            2, 2 * self.lstm_dim, self.lstm_dim, self.model)

        self.f_label = Feedforward(
            self.model, 3 * self.lstm_dim, [self.fc_hidden_dim], self.label_out)

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

        def helper(left, split, right_bound, child_state, left_trees=None, left_loss=None):
            if left == split:
                h_hat = child_state.output()
                span_encoding = dy.concatenate(
                    [self.get_span_encoding(lstm_outputs, left, split), h_hat])
                label_scores = self.f_label(span_encoding)
                if is_train:
                    oracle_label, oracle_label_index, crossing = self.get_oracle_label(
                        gold_tree, left, split)
                    label_scores = self.augment(
                        label_scores, oracle_label_index, crossing)
                argmax_label, argmax_label_index = self.predict_label(
                    label_scores, gold_tree, left, split)

                if is_train:
                    left_loss = (
                        label_scores[argmax_label_index] -
                        label_scores[oracle_label_index]
                        if argmax_label != oracle_label else dy.zeros(1))

                left_label = argmax_label
                left_trees = self.gen_leaf_tree(left, left_label)
                left_trees = [left_trees]
                child_state = child_state.add_input(
                    self.get_span_encoding(lstm_outputs, left, split))

            if split == right_bound:
                return left_trees, left_loss, child_state

            parent_right_scores = self.get_right_boundary_scores(
                lstm_outputs, left, split, right_bound)
            if is_train:
                oracle_rights = gold_tree.parent_rights(
                    left, split, right_bound)
                oracle_right = max(oracle_rights)
                oracle_right_index = oracle_right - (split + 1)
                parent_right_scores = self.augment(
                    parent_right_scores, oracle_right_index)
            parent_right_scores_np = parent_right_scores.npvalue()
            argmax_right_index = int(parent_right_scores_np.argmax())
            argmax_right = argmax_right_index + (split + 1)

            right = argmax_right

            h_hat = child_state.output()
            span_encoding = dy.concatenate(
                [self.get_span_encoding(lstm_outputs, left, right), h_hat])
            label_scores = self.f_label(span_encoding)
            if is_train:
                oracle_label, oracle_label_index, crossing = self.get_oracle_label(
                    gold_tree, left, right)
                label_scores = self.augment(
                    label_scores, oracle_label_index, crossing)
            argmax_label, argmax_label_index = self.predict_label(
                label_scores, gold_tree, left, right)
            label = argmax_label
            child_state = child_state.add_input(
                self.get_span_encoding(lstm_outputs, left, right))

            if is_train:
                parent_loss = (
                    parent_right_scores[argmax_right_index] -
                    parent_right_scores[oracle_right_index]
                    if argmax_right != oracle_right else dy.zeros(1))
                label_loss = (
                    label_scores[argmax_label_index] -
                    label_scores[oracle_label_index]
                    if argmax_label != oracle_label else dy.zeros(1))

            right_trees, right_loss, right_state = helper(
                split + 1, split + 1, right, child_state, None, None)

            childrens = left_trees + right_trees
            childrens = self.gen_nonleaf_tree(childrens, label)

            tree, loss, parent_state = helper(left, right, right_bound, child_state, childrens, (
                parent_loss + label_loss + left_loss + right_loss) if is_train else None)

            return tree, loss, parent_state

        leaf_state = self.historical_info_lstm.initial_state()
        leaf_state = leaf_state.add_input(dy.zeros(2 * self.lstm_dim))
        childrens, loss, _ = helper(
            0, 0, len(sentence) - 1, leaf_state, None, None)
        assert len(childrens) == 1
        tree = childrens[0]
        tree.propagate_sentence(sentence)

        if is_train:
            return loss, tree, 1
        else:
            return tree


class InOrderParser5(BaseParser):
    '''
        InOrder greedy parser
        historical_info_lstm:
            - with branch
            - input = [label_embedding; lstm1(l, r)]
        label prediction: [lstm(l, r); historical_info_lstm(t)]
        right boundary point prediction: lstm(l, r)
    '''

    def __init__(self, model, parameters):
        super().__init__(model, *parameters)
        self.spec = {'parameters': parameters}

        self.lstm1 = dy.BiRNNBuilder(
            self.lstm_layers,
            self.tag_embedding_dim + 2 * self.char_lstm_dim + self.word_embedding_dim,
            2 * self.lstm_dim,
            self.model,
            dy.VanillaLSTMBuilder)

        self.historical_info_lstm = dy.VanillaLSTMBuilder(
            2, self.label_embedding_dim + 2 * self.lstm_dim, self.lstm_dim, self.model)

        self.f_label = Feedforward(
            self.model, 3 * self.lstm_dim, [self.fc_hidden_dim], self.label_out)

    def parse(self, data, is_train=False):
        if is_train:
            self.lstm.set_dropout(self.dropout)
            self.lstm1.set_dropout(self.dropout)
        else:
            self.lstm.disable_dropout()
            self.lstm1.disable_dropout()

        word_indices = data['w']
        tag_indices = data['t']
        gold_tree = data['tree']
        sentence = gold_tree.sentence

        embeddings = self.get_embeddings(word_indices, tag_indices, is_train)
        lstm_outputs = self.lstm.transduce(embeddings)
        lstm_outputs1 = self.lstm1.transduce(embeddings)

        def helper(left, split, right_bound, child_state, left_trees=None, left_loss=None):
            if left == split:
                h_hat = child_state.output()
                span_encoding = dy.concatenate(
                    [self.get_span_encoding(lstm_outputs, left, split), h_hat])
                label_scores = self.f_label(span_encoding)
                if is_train:
                    oracle_label, oracle_label_index, crossing = self.get_oracle_label(
                        gold_tree, left, split)
                    label_scores = self.augment(
                        label_scores, oracle_label_index, crossing)
                argmax_label, argmax_label_index = self.predict_label(
                    label_scores, gold_tree, left, split)

                if is_train:
                    left_loss = (
                        label_scores[argmax_label_index] -
                        label_scores[oracle_label_index]
                        if argmax_label != oracle_label else dy.zeros(1))

                left_label = argmax_label
                left_trees = self.gen_leaf_tree(left, left_label)
                left_trees = [left_trees]
                child_state = child_state.add_input(
                    dy.concatenate([self.label_embeddings[argmax_label_index], self.get_span_encoding(lstm_outputs1, left, split)]))

            if split == right_bound:
                return left_trees, left_loss, child_state

            parent_right_scores = self.get_right_boundary_scores(
                lstm_outputs, left, split, right_bound)
            if is_train:
                oracle_rights = gold_tree.parent_rights(
                    left, split, right_bound)
                oracle_right = max(oracle_rights)
                oracle_right_index = oracle_right - (split + 1)
                parent_right_scores = self.augment(
                    parent_right_scores, oracle_right_index)
            parent_right_scores_np = parent_right_scores.npvalue()
            argmax_right_index = int(parent_right_scores_np.argmax())
            argmax_right = argmax_right_index + (split + 1)

            right = argmax_right

            h_hat = child_state.output()
            span_encoding = dy.concatenate(
                [self.get_span_encoding(lstm_outputs, left, right), h_hat])
            label_scores = self.f_label(span_encoding)
            if is_train:
                oracle_label, oracle_label_index, crossing = self.get_oracle_label(
                    gold_tree, left, right)
                label_scores = self.augment(
                    label_scores, oracle_label_index, crossing)
            argmax_label, argmax_label_index = self.predict_label(
                label_scores, gold_tree, left, right)
            label = argmax_label
            child_state = child_state.add_input(
                dy.concatenate([self.label_embeddings[argmax_label_index], self.get_span_encoding(lstm_outputs1, left, right)]))

            if is_train:
                parent_loss = (
                    parent_right_scores[argmax_right_index] -
                    parent_right_scores[oracle_right_index]
                    if argmax_right != oracle_right else dy.zeros(1))
                label_loss = (
                    label_scores[argmax_label_index] -
                    label_scores[oracle_label_index]
                    if argmax_label != oracle_label else dy.zeros(1))

            right_trees, right_loss, right_state = helper(
                split + 1, split + 1, right, child_state, None, None)

            childrens = left_trees + right_trees
            childrens = self.gen_nonleaf_tree(childrens, label)

            tree, loss, parent_state = helper(left, right, right_bound, child_state, childrens, (
                parent_loss + label_loss + left_loss + right_loss) if is_train else None)

            return tree, loss, parent_state

        leaf_state = self.historical_info_lstm.initial_state()
        leaf_state = leaf_state.add_input(
            dy.concatenate([self.label_embeddings[self.label_out], dy.zeros(2 * self.lstm_dim)]))
        childrens, loss, _ = helper(
            0, 0, len(sentence) - 1, leaf_state, None, None)
        assert len(childrens) == 1
        tree = childrens[0]
        tree.propagate_sentence(sentence)

        if is_train:
            return loss, tree, 1
        else:
            return tree


class InOrderParser6(BaseParser):
    '''
        InOrder greedy parser
        historical_info_lstm:
            - with branch
            - input = lstm1(l, r)
        label prediction: [lstm(l, r); historical_info_lstm(t)]
        right boundary point prediction: lstm(l, r)
    '''

    def __init__(self, model, parameters):
        super().__init__(model, *parameters)
        self.spec = {'parameters': parameters}

        self.lstm1 = dy.BiRNNBuilder(
            self.lstm_layers,
            self.tag_embedding_dim + 2 * self.char_lstm_dim + self.word_embedding_dim,
            2 * self.lstm_dim,
            self.model,
            dy.VanillaLSTMBuilder)

        self.historical_info_lstm = dy.VanillaLSTMBuilder(
            2, self.label_embedding_dim + 2 * self.lstm_dim, self.lstm_dim, self.model)

        self.f_label = Feedforward(
            self.model, 3 * self.lstm_dim, [self.fc_hidden_dim], self.label_out)

    def parse(self, data, is_train=False):
        if is_train:
            self.lstm.set_dropout(self.dropout)
            self.lstm1.set_dropout(self.dropout)
        else:
            self.lstm.disable_dropout()
            self.lstm1.disable_dropout()

        word_indices = data['w']
        tag_indices = data['t']
        gold_tree = data['tree']
        sentence = gold_tree.sentence

        embeddings = self.get_embeddings(word_indices, tag_indices, is_train)
        lstm_outputs = self.lstm.transduce(embeddings)
        lstm_outputs1 = self.lstm1.transduce(embeddings)

        def helper(left, split, right_bound, child_state, left_trees=None, left_loss=None):
            if left == split:
                h_hat = child_state.output()
                span_encoding = dy.concatenate(
                    [self.get_span_encoding(lstm_outputs, left, split), h_hat])
                label_scores = self.f_label(span_encoding)
                if is_train:
                    oracle_label, oracle_label_index, crossing = self.get_oracle_label(
                        gold_tree, left, split)
                    label_scores = self.augment(
                        label_scores, oracle_label_index, crossing)
                argmax_label, argmax_label_index = self.predict_label(
                    label_scores, gold_tree, left, split)

                if is_train:
                    left_loss = (
                        label_scores[argmax_label_index] -
                        label_scores[oracle_label_index]
                        if argmax_label != oracle_label else dy.zeros(1))

                left_label = argmax_label
                left_trees = self.gen_leaf_tree(left, left_label)
                left_trees = [left_trees]
                child_state = child_state.add_input(
                    self.get_span_encoding(lstm_outputs1, left, split))

            if split == right_bound:
                return left_trees, left_loss, child_state

            parent_right_scores = self.get_right_boundary_scores(
                lstm_outputs, left, split, right_bound)
            if is_train:
                oracle_rights = gold_tree.parent_rights(
                    left, split, right_bound)
                oracle_right = max(oracle_rights)
                oracle_right_index = oracle_right - (split + 1)
                parent_right_scores = self.augment(
                    parent_right_scores, oracle_right_index)
            parent_right_scores_np = parent_right_scores.npvalue()
            argmax_right_index = int(parent_right_scores_np.argmax())
            argmax_right = argmax_right_index + (split + 1)

            right = argmax_right

            h_hat = child_state.output()
            span_encoding = dy.concatenate(
                [self.get_span_encoding(lstm_outputs, left, right), h_hat])
            label_scores = self.f_label(span_encoding)
            if is_train:
                oracle_label, oracle_label_index, crossing = self.get_oracle_label(
                    gold_tree, left, right)
                label_scores = self.augment(
                    label_scores, oracle_label_index, crossing)
            argmax_label, argmax_label_index = self.predict_label(
                label_scores, gold_tree, left, right)
            label = argmax_label
            child_state = child_state.add_input(
                self.get_span_encoding(lstm_outputs1, left, right))

            if is_train:
                parent_loss = (
                    parent_right_scores[argmax_right_index] -
                    parent_right_scores[oracle_right_index]
                    if argmax_right != oracle_right else dy.zeros(1))
                label_loss = (
                    label_scores[argmax_label_index] -
                    label_scores[oracle_label_index]
                    if argmax_label != oracle_label else dy.zeros(1))

            right_trees, right_loss, right_state = helper(
                split + 1, split + 1, right, child_state, None, None)

            childrens = left_trees + right_trees
            childrens = self.gen_nonleaf_tree(childrens, label)

            tree, loss, parent_state = helper(left, right, right_bound, child_state, childrens, (
                parent_loss + label_loss + left_loss + right_loss) if is_train else None)

            return tree, loss, parent_state

        leaf_state = self.historical_info_lstm.initial_state()
        leaf_state = leaf_state.add_input(dy.zeros(2 * self.lstm_dim))
        childrens, loss, _ = helper(
            0, 0, len(sentence) - 1, leaf_state, None, None)
        assert len(childrens) == 1
        tree = childrens[0]
        tree.propagate_sentence(sentence)

        if is_train:
            return loss, tree, 1
        else:
            return tree


class InOrderParser7(BaseParser):
    '''
        InOrder greedy parser
        historical_info_lstm:
            - with branch
            - input = label_embedding
        label prediction: [lstm(l, r); historical_info_lstm(t)]
        right boundary point prediction: [lstm(l, r); historical_info_lstm(t)]
    '''

    def __init__(self, model, parameters):
        super().__init__(model, *parameters)
        self.spec = {'parameters': parameters}

        self.f_label = Feedforward(
            self.model, 3 * self.lstm_dim, [self.fc_hidden_dim], self.label_out)

        self.f_split = Feedforward(
            self.model, 3 * self.lstm_dim, [self.fc_hidden_dim], 1)

    def get_right_boundary_scores(self, lstm_outputs, child_state, left, split, right_bound):
        h_hat = child_state.output()
        left_encodings = []
        right_encodings = []
        for right in range(split + 1, right_bound + 1):
            left_encodings.append(dy.concatenate(
                [self.get_span_encoding(lstm_outputs, left, right), h_hat]))
            right_encodings.append(dy.concatenate(
                [self.get_span_encoding(lstm_outputs, split + 1, right), h_hat]))
        left_scores = self.f_split(dy.concatenate_to_batch(left_encodings))
        right_scores = self.f_split(
            dy.concatenate_to_batch(right_encodings))
        parent_right_scores = left_scores + right_scores
        parent_right_scores = dy.reshape(
            parent_right_scores, (len(left_encodings),))
        return parent_right_scores

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

        def helper(left, split, right_bound, child_state, left_trees=None, left_loss=None):
            if left == split:
                h_hat = child_state.output()
                span_encoding = dy.concatenate(
                    [self.get_span_encoding(lstm_outputs, left, split), h_hat])
                label_scores = self.f_label(span_encoding)
                if is_train:
                    oracle_label, oracle_label_index, crossing = self.get_oracle_label(
                        gold_tree, left, split)
                    label_scores = self.augment(
                        label_scores, oracle_label_index, crossing)
                argmax_label, argmax_label_index = self.predict_label(
                    label_scores, gold_tree, left, split)

                if is_train:
                    left_loss = (
                        label_scores[argmax_label_index] -
                        label_scores[oracle_label_index]
                        if argmax_label != oracle_label else dy.zeros(1))

                left_label = argmax_label
                left_trees = self.gen_leaf_tree(left, left_label)
                left_trees = [left_trees]
                child_state = child_state.add_input(
                    self.label_embeddings[argmax_label_index])

            if split == right_bound:
                return left_trees, left_loss, child_state

            parent_right_scores = self.get_right_boundary_scores(
                lstm_outputs, child_state, left, split, right_bound)
            if is_train:
                oracle_rights = gold_tree.parent_rights(
                    left, split, right_bound)
                oracle_right = max(oracle_rights)
                oracle_right_index = oracle_right - (split + 1)
                parent_right_scores = self.augment(
                    parent_right_scores, oracle_right_index)
            parent_right_scores_np = parent_right_scores.npvalue()
            argmax_right_index = int(parent_right_scores_np.argmax())
            argmax_right = argmax_right_index + (split + 1)

            right = argmax_right

            h_hat = child_state.output()
            span_encoding = dy.concatenate(
                [self.get_span_encoding(lstm_outputs, left, right), h_hat])
            label_scores = self.f_label(span_encoding)
            if is_train:
                oracle_label, oracle_label_index, crossing = self.get_oracle_label(
                    gold_tree, left, right)
                label_scores = self.augment(
                    label_scores, oracle_label_index, crossing)
            argmax_label, argmax_label_index = self.predict_label(
                label_scores, gold_tree, left, right)
            label = argmax_label
            child_state = child_state.add_input(
                self.label_embeddings[argmax_label_index])

            if is_train:
                parent_loss = (
                    parent_right_scores[argmax_right_index] -
                    parent_right_scores[oracle_right_index]
                    if argmax_right != oracle_right else dy.zeros(1))
                label_loss = (
                    label_scores[argmax_label_index] -
                    label_scores[oracle_label_index]
                    if argmax_label != oracle_label else dy.zeros(1))

            right_trees, right_loss, right_state = helper(
                split + 1, split + 1, right, child_state, None, None)

            childrens = left_trees + right_trees
            childrens = self.gen_nonleaf_tree(childrens, label)

            tree, loss, parent_state = helper(left, right, right_bound, child_state, childrens, (
                parent_loss + label_loss + left_loss + right_loss) if is_train else None)

            return tree, loss, parent_state

        leaf_state = self.historical_info_lstm.initial_state()
        leaf_state = leaf_state.add_input(
            self.label_embeddings[self.label_out])
        childrens, loss, _ = helper(
            0, 0, len(sentence) - 1, leaf_state, None, None)
        assert len(childrens) == 1
        tree = childrens[0]
        tree.propagate_sentence(sentence)

        if is_train:
            return loss, tree, 1
        else:
            return tree


class InOrderParser8(BaseParser):
    '''
        InOrder greedy parser
        historical_info_lstm:
            - with branch
            - input = [label_embedding; lstm(l, r)]
        label prediction: [lstm(l, r); historical_info_lstm(t)]
        right boundary point prediction: [lstm(l, r); historical_info_lstm(t)]
    '''

    def __init__(self, model, parameters):
        super().__init__(model, *parameters)
        self.spec = {'parameters': parameters}

        self.historical_info_lstm = dy.VanillaLSTMBuilder(
            2, self.label_embedding_dim + 2 * self.lstm_dim, self.lstm_dim, self.model)

        self.f_label = Feedforward(
            self.model, 3 * self.lstm_dim, [self.fc_hidden_dim], self.label_out)

        self.f_split = Feedforward(
            self.model, 3 * self.lstm_dim, [self.fc_hidden_dim], 1)

    def get_right_boundary_scores(self, lstm_outputs, child_state, left, split, right_bound):
        h_hat = child_state.output()
        left_encodings = []
        right_encodings = []
        for right in range(split + 1, right_bound + 1):
            left_encodings.append(dy.concatenate(
                [self.get_span_encoding(lstm_outputs, left, right), h_hat]))
            right_encodings.append(dy.concatenate(
                [self.get_span_encoding(lstm_outputs, split + 1, right), h_hat]))
        left_scores = self.f_split(dy.concatenate_to_batch(left_encodings))
        right_scores = self.f_split(
            dy.concatenate_to_batch(right_encodings))
        parent_right_scores = left_scores + right_scores
        parent_right_scores = dy.reshape(
            parent_right_scores, (len(left_encodings),))
        return parent_right_scores

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

        def helper(left, split, right_bound, child_state, left_trees=None, left_loss=None):
            if left == split:
                h_hat = child_state.output()
                span_encoding = dy.concatenate(
                    [self.get_span_encoding(lstm_outputs, left, split), h_hat])
                label_scores = self.f_label(span_encoding)
                if is_train:
                    oracle_label, oracle_label_index, crossing = self.get_oracle_label(
                        gold_tree, left, split)
                    label_scores = self.augment(
                        label_scores, oracle_label_index, crossing)
                argmax_label, argmax_label_index = self.predict_label(
                    label_scores, gold_tree, left, split)

                if is_train:
                    left_loss = (
                        label_scores[argmax_label_index] -
                        label_scores[oracle_label_index]
                        if argmax_label != oracle_label else dy.zeros(1))

                left_label = argmax_label
                left_trees = self.gen_leaf_tree(left, left_label)
                left_trees = [left_trees]
                child_state = child_state.add_input(
                    dy.concatenate([self.label_embeddings[argmax_label_index], self.get_span_encoding(lstm_outputs, left, split)]))

            if split == right_bound:
                return left_trees, left_loss, child_state

            parent_right_scores = self.get_right_boundary_scores(
                lstm_outputs, child_state, left, split, right_bound)
            if is_train:
                oracle_rights = gold_tree.parent_rights(
                    left, split, right_bound)
                oracle_right = max(oracle_rights)
                oracle_right_index = oracle_right - (split + 1)
                parent_right_scores = self.augment(
                    parent_right_scores, oracle_right_index)
            parent_right_scores_np = parent_right_scores.npvalue()
            argmax_right_index = int(parent_right_scores_np.argmax())
            argmax_right = argmax_right_index + (split + 1)

            right = argmax_right

            h_hat = child_state.output()
            span_encoding = dy.concatenate(
                [self.get_span_encoding(lstm_outputs, left, right), h_hat])
            label_scores = self.f_label(span_encoding)
            if is_train:
                oracle_label, oracle_label_index, crossing = self.get_oracle_label(
                    gold_tree, left, right)
                label_scores = self.augment(
                    label_scores, oracle_label_index, crossing)
            argmax_label, argmax_label_index = self.predict_label(
                label_scores, gold_tree, left, right)
            label = argmax_label
            child_state = child_state.add_input(
                dy.concatenate([self.label_embeddings[argmax_label_index], self.get_span_encoding(lstm_outputs, left, right)]))

            if is_train:
                parent_loss = (
                    parent_right_scores[argmax_right_index] -
                    parent_right_scores[oracle_right_index]
                    if argmax_right != oracle_right else dy.zeros(1))
                label_loss = (
                    label_scores[argmax_label_index] -
                    label_scores[oracle_label_index]
                    if argmax_label != oracle_label else dy.zeros(1))

            right_trees, right_loss, right_state = helper(
                split + 1, split + 1, right, child_state, None, None)

            childrens = left_trees + right_trees
            childrens = self.gen_nonleaf_tree(childrens, label)

            tree, loss, parent_state = helper(left, right, right_bound, child_state, childrens, (
                parent_loss + label_loss + left_loss + right_loss) if is_train else None)

            return tree, loss, parent_state

        leaf_state = self.historical_info_lstm.initial_state()
        leaf_state = leaf_state.add_input(
            dy.concatenate([self.label_embeddings[self.label_out], dy.zeros(2 * self.lstm_dim)]))
        childrens, loss, _ = helper(
            0, 0, len(sentence) - 1, leaf_state, None, None)
        assert len(childrens) == 1
        tree = childrens[0]
        tree.propagate_sentence(sentence)

        if is_train:
            return loss, tree, 1
        else:
            return tree
