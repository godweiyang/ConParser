import functools
import time

import _dynet as dy
import numpy as np

from lib import *

'''
InOrder + inorder-lstm + span encoding=[lstm(l, r); inorder-LSTM output]
'''


class InOrderParser13(object):
    def network_init(self):
        self.tag_embeddings = self.model.add_lookup_parameters(
            (self.tag_count, self.tag_embedding_dim),
            init='uniform',
            scale=0.01
        )
        self.char_embeddings = self.model.add_lookup_parameters(
            (self.char_count, self.char_embedding_dim),
            init='uniform',
            scale=0.01
        )
        self.word_embeddings = self.model.add_lookup_parameters(
            (self.word_count, self.word_embedding_dim),
            init='uniform',
            scale=0.01
        )
        self.label_embeddings = self.model.add_lookup_parameters(
            (self.label_out + 1, self.label_embedding_dim),
            init='uniform',
            scale=0.01
        )

        self.char_lstm = dy.BiRNNBuilder(
            self.char_lstm_layers,
            self.char_embedding_dim,
            2 * self.char_lstm_dim,
            self.model,
            dy.VanillaLSTMBuilder)

        self.lstm = dy.BiRNNBuilder(
            self.lstm_layers,
            self.tag_embedding_dim + 2 * self.char_lstm_dim + self.word_embedding_dim,
            2 * self.lstm_dim,
            self.model,
            dy.VanillaLSTMBuilder)

        self.inorder_lstm = dy.VanillaLSTMBuilder(
            2, self.label_embedding_dim, self.lstm_dim, self.model)

        self.f_label = Feedforward(
            self.model, 3 * self.lstm_dim, [self.fc_hidden_dim], self.label_out)

        self.f_split = Feedforward(
            self.model, 2 * self.lstm_dim, [self.fc_hidden_dim], 1)

    def __init__(
        self,
        model,
        vocab,
        word_embedding_dim,
        tag_embedding_dim,
        char_embedding_dim,
        label_embedding_dim,
        pos_embedding_dim,
        char_lstm_layers,
        char_lstm_dim,
        lstm_layers,
        lstm_dim,
        fc_hidden_dim,
        dropout,
        unk_param,
    ):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model
        self.vocab = vocab
        self.word_count = vocab.total_words()
        self.tag_count = vocab.total_tags()
        self.char_count = vocab.total_characters()
        self.word_embedding_dim = word_embedding_dim
        self.tag_embedding_dim = tag_embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.label_embedding_dim = label_embedding_dim
        self.char_lstm_layers = char_lstm_layers
        self.char_lstm_dim = char_lstm_dim
        self.lstm_layers = lstm_layers
        self.lstm_dim = lstm_dim
        self.fc_hidden_dim = fc_hidden_dim
        self.label_out = vocab.total_label_actions()
        self.dropout = dropout
        self.unk_param = unk_param

        self.network_init()

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    @staticmethod
    def augment(scores, oracle_index, crossing=False):
        assert isinstance(scores, dy.Expression)
        shape = scores.dim()[0]
        assert len(shape) == 1
        increment = np.ones(shape)
        increment[oracle_index] = crossing
        return scores + dy.inputVector(increment)

    def get_embeddings(self, word_inds, tag_inds, is_train=False):
        if is_train:
            self.char_lstm.set_dropout(self.dropout)
        else:
            self.char_lstm.disable_dropout()

        embeddings = []
        for w, t in zip(word_inds, tag_inds):
            if w > 2:
                count = self.vocab.word_freq_list[w]
                if not count or (is_train and np.random.rand() < self.unk_param / (self.unk_param + count)):
                    w = 0

            tag_embedding = self.tag_embeddings[t]
            chars = list(self.vocab.i2w[w]) if w > 2 else [
                self.vocab.i2w[w]]
            char_lstm_outputs = self.char_lstm.transduce(
                [self.char_embeddings[self.vocab.c2i[char]] for char in [Vocabulary.START] + chars + [Vocabulary.STOP]])
            char_embedding = dy.concatenate([
                char_lstm_outputs[-1][:self.char_lstm_dim],
                char_lstm_outputs[0][self.char_lstm_dim:]])
            word_embedding = self.word_embeddings[w]
            embeddings.append(dy.concatenate([
                tag_embedding, char_embedding, word_embedding]))

        return embeddings

    def parse(self, data):
        self.lstm.set_dropout(self.dropout)

        word_indices = data['w']
        tag_indices = data['t']
        gold_tree = data['tree']
        sentence = gold_tree.sentence

        embeddings = self.get_embeddings(
            word_indices, tag_indices, is_train=True)
        lstm_outputs = self.lstm.transduce(embeddings)

        @functools.lru_cache(maxsize=None)
        def get_span_encoding(left, right):
            forward = (
                lstm_outputs[right + 1][:self.lstm_dim] -
                lstm_outputs[left][:self.lstm_dim])
            backward = (
                lstm_outputs[left + 1][self.lstm_dim:] -
                lstm_outputs[right + 2][self.lstm_dim:])
            return dy.concatenate([forward, backward])

        @functools.lru_cache(maxsize=None)
        def get_label_scores(left, right, child_state):
            h_hat = child_state.output()
            span_encoding = dy.concatenate(
                [get_span_encoding(left, right), h_hat])
            label_scores = self.f_label(span_encoding)

            return label_scores

        @functools.lru_cache(maxsize=None)
        def predict_label(left, right, child_state):
            label_scores = get_label_scores(left, right, child_state)

            oracle_label, crossing = gold_tree.span_labels(left, right)
            oracle_label = oracle_label[::-1]
            if len(oracle_label) == 0:
                oracle_label = 'none'
            else:
                oracle_label = 'label-' + '-'.join(oracle_label)
            oracle_label_index = self.vocab.l_action_index(oracle_label)
            label_scores = InOrderParser13.augment(
                label_scores, oracle_label_index, crossing)

            label_scores_np = label_scores.npvalue()
            argmax_label_index = int(
                label_scores_np.argmax() if right - left + 1 < len(sentence) else
                label_scores_np[1:].argmax() + 1)

            if argmax_label_index == 0:
                argmax_label = 'none'
            else:
                argmax_label = 'label-' + \
                    self.vocab.i2l[argmax_label_index - 1]

            label = argmax_label
            label_loss = (
                label_scores[argmax_label_index] -
                label_scores[oracle_label_index]
                if argmax_label != oracle_label else dy.zeros(1))

            return label, label_loss, argmax_label_index

        def helper(left, split, right_bound, child_state, left_trees=None, left_loss=None):
            '''
            left_span: [left, split]
            right_span:[split+1, <=right_bound]
            '''
            if left == split or left_trees is None or left_loss is None:
                assert left == split and left_trees is None and left_loss is None

                child_state = child_state.add_input(embeddings[left])

                left_label, left_loss, argmax_label_index = predict_label(
                    left, split, child_state)
                left_trees = PhraseTree(leaf=left)
                if left_label != 'none':
                    for nt in left_label[6:].split('-'):
                        left_trees = PhraseTree(
                            symbol=nt, children=[left_trees])
                left_trees = [left_trees]
                child_state = child_state.add_input(
                    self.label_embeddings[argmax_label_index])

            if split == right_bound:
                return left_trees, left_loss, child_state

            left_encodings = []
            right_encodings = []
            for right in range(split + 1, right_bound + 1):
                left_encodings.append(get_span_encoding(left, right))
                right_encodings.append(get_span_encoding(split + 1, right))
            left_scores = self.f_split(dy.concatenate_to_batch(left_encodings))
            right_scores = self.f_split(
                dy.concatenate_to_batch(right_encodings))
            parent_right_scores = left_scores + right_scores
            parent_right_scores = dy.reshape(
                parent_right_scores, (len(left_encodings),))

            oracle_rights = gold_tree.parent_rights(left, split, right_bound)
            oracle_right = max(oracle_rights)
            oracle_right_index = oracle_right - (split + 1)
            parent_right_scores = InOrderParser13.augment(
                parent_right_scores, oracle_right_index)

            parent_right_scores_np = parent_right_scores.npvalue()
            argmax_right_index = int(parent_right_scores_np.argmax())
            argmax_right = argmax_right_index + (split + 1)

            right = argmax_right
            parent_loss = (
                parent_right_scores[argmax_right_index] -
                parent_right_scores[oracle_right_index]
                if argmax_right != oracle_right else dy.zeros(1))

            label, label_loss, argmax_label_index = predict_label(
                left, right, child_state)

            child_state = child_state.add_input(
                self.label_embeddings[argmax_label_index])

            right_trees, right_loss, right_state = helper(
                split + 1, split + 1, right, child_state, None, None)

            childrens = left_trees + right_trees

            if label != 'none':
                for nt in label[6:].split('-'):
                    childrens = [PhraseTree(
                        symbol=nt, children=childrens)]

            tree, loss, parent_state = helper(
                left, right, right_bound, right_state, childrens, parent_loss + label_loss + left_loss + right_loss)

            return tree, loss, parent_state

        leaf_state = self.inorder_lstm.initial_state()
        leaf_state = leaf_state.add_input(
            self.label_embeddings[self.label_out])
        childrens, loss, _ = helper(
            0, 0, len(sentence) - 1, leaf_state, None, None)
        assert len(childrens) == 1
        tree = childrens[0]
        tree.propagate_sentence(sentence)

        return loss, tree, 1

    def predict(self, gold_tree):
        self.lstm.disable_dropout()

        sentence = gold_tree.sentence
        w, t = self.vocab.sentence_sequences(sentence)

        embeddings = self.get_embeddings(w, t)
        lstm_outputs = self.lstm.transduce(embeddings)

        @functools.lru_cache(maxsize=None)
        def get_span_encoding(left, right):
            forward = (
                lstm_outputs[right + 1][:self.lstm_dim] -
                lstm_outputs[left][:self.lstm_dim])
            backward = (
                lstm_outputs[left + 1][self.lstm_dim:] -
                lstm_outputs[right + 2][self.lstm_dim:])
            return dy.concatenate([forward, backward])

        @functools.lru_cache(maxsize=None)
        def get_label_scores(left, right, child_state):
            h_hat = child_state.output()
            span_encoding = dy.concatenate(
                [get_span_encoding(left, right), h_hat])
            label_scores = self.f_label(span_encoding)

            return label_scores

        @functools.lru_cache(maxsize=None)
        def predict_label(left, right, child_state):
            label_scores = get_label_scores(left, right, child_state)

            label_scores_np = label_scores.npvalue()
            argmax_label_index = int(
                label_scores_np.argmax() if right - left + 1 < len(sentence) else
                label_scores_np[1:].argmax() + 1)

            if argmax_label_index == 0:
                argmax_label = 'none'
            else:
                argmax_label = 'label-' + \
                    self.vocab.i2l[argmax_label_index - 1]

            label = argmax_label
            label_loss = label_scores[argmax_label_index]

            return label, label_loss, argmax_label_index

        def helper(left, split, right_bound, child_state, left_trees=None, left_loss=None):
            '''
            left_span: [left, split]
            right_span:[split+1, <=right_bound]
            '''
            if left == split or left_trees is None or left_loss is None:
                assert left == split and left_trees is None and left_loss is None

                child_state = child_state.add_input(embeddings[left])

                left_label, left_loss, argmax_label_index = predict_label(
                    left, split, child_state)
                left_trees = PhraseTree(leaf=left)
                if left_label != 'none':
                    for nt in left_label[6:].split('-'):
                        left_trees = PhraseTree(
                            symbol=nt, children=[left_trees])
                left_trees = [left_trees]
                child_state = child_state.add_input(
                    self.label_embeddings[argmax_label_index])

            if split == right_bound:
                return left_trees, left_loss, child_state

            left_encodings = []
            right_encodings = []
            for right in range(split + 1, right_bound + 1):
                left_encodings.append(get_span_encoding(left, right))
                right_encodings.append(get_span_encoding(split + 1, right))
            left_scores = self.f_split(dy.concatenate_to_batch(left_encodings))
            right_scores = self.f_split(
                dy.concatenate_to_batch(right_encodings))
            parent_right_scores = left_scores + right_scores
            parent_right_scores = dy.reshape(
                parent_right_scores, (len(left_encodings),))

            parent_right_scores_np = parent_right_scores.npvalue()
            argmax_right_index = int(parent_right_scores_np.argmax())
            argmax_right = argmax_right_index + (split + 1)

            right = argmax_right
            parent_loss = parent_right_scores[argmax_right_index]

            label, label_loss, argmax_label_index = predict_label(
                left, right, child_state)

            child_state = child_state.add_input(
                self.label_embeddings[argmax_label_index])

            right_trees, right_loss, right_state = helper(
                split + 1, split + 1, right, child_state, None, None)

            childrens = left_trees + right_trees

            if label != 'none':
                for nt in label[6:].split('-'):
                    childrens = [PhraseTree(
                        symbol=nt, children=childrens)]

            tree, loss, parent_state = helper(
                left, right, right_bound, right_state, childrens, parent_loss + label_loss + left_loss + right_loss)

            return tree, loss, parent_state

        leaf_state = self.inorder_lstm.initial_state()
        leaf_state = leaf_state.add_input(
            self.label_embeddings[self.label_out])
        childrens, loss, _ = helper(
            0, 0, len(sentence) - 1, leaf_state, None, None)
        assert len(childrens) == 1
        tree = childrens[0]
        tree.propagate_sentence(sentence)

        return tree
