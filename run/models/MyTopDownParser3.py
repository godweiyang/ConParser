import functools
import time

import _dynet as dy
import numpy as np

from lib import *

'''
TopDown + label-LSTM + split-LSTM + headed-W + label and split parameters dynamic generate
'''


class MyTopDownParser3(object):
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
            (self.label_out, self.label_embedding_dim),
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

        self.label_lstm = dy.VanillaLSTMBuilder(
            2, self.label_embedding_dim, self.fc_hidden_dim, self.model)

        self.split_lstm = dy.VanillaLSTMBuilder(
            2, 2 * self.lstm_dim, self.fc_hidden_dim, self.model)

        self.label_lstm_W_hidden = []
        for i in range(2):
            self.label_lstm_W_hidden.append(
                self.model.add_parameters((self.label_out, self.fc_hidden_dim)))

        self.split_lstm_W_hidden = []
        for i in range(2):
            self.split_lstm_W_hidden.append(
                self.model.add_parameters((self.label_out, self.fc_hidden_dim)))

        self.f_label = MultiWeightedFeedforward(
            self.model, 2 * self.lstm_dim, [self.fc_hidden_dim], self.label_out, self.label_out)

        self.f_split = MultiWeightedFeedforward(
            self.model, 2 * self.lstm_dim, [self.fc_hidden_dim], 1, self.label_out)

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
        def get_label_scores(left, right, parent_state, direction):
            h_hat = self.label_lstm_W_hidden[direction] * parent_state.output()
            h_hat_np = h_hat.npvalue()
            argmax_weight_index = int(h_hat_np.argmax())

            label_scores = self.f_label(get_span_encoding(
                left, right), argmax_weight_index)

            return label_scores

        @functools.lru_cache(maxsize=None)
        def get_split_scores(encodings, parent_state, direction):
            h_hat = self.split_lstm_W_hidden[direction] * parent_state.output()
            h_hat_np = h_hat.npvalue()
            argmax_weight_index = int(h_hat_np.argmax())

            split_scores = self.f_split(encodings, argmax_weight_index)

            return split_scores

        def helper(left, right, label_parent_state, split_parent_state, direction):
            label_scores = get_label_scores(
                left, right, label_parent_state, direction)

            oracle_label, crossing = gold_tree.span_labels(left, right)
            oracle_label = oracle_label[::-1]
            if len(oracle_label) == 0:
                oracle_label = 'none'
            else:
                oracle_label = 'label-' + '-'.join(oracle_label)
            oracle_label_index = self.vocab.l_action_index(oracle_label)
            label_scores = MyTopDownParser3.augment(
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

            label_child_state = label_parent_state.add_input(
                self.label_embeddings[argmax_label_index])

            label = argmax_label
            label_loss = (
                label_scores[argmax_label_index] -
                label_scores[oracle_label_index]
                if argmax_label != oracle_label else dy.zeros(1))

            if left == right:
                tree = PhraseTree(leaf=left)
                if label != 'none':
                    for nt in label[6:].split('-'):
                        tree = PhraseTree(symbol=nt, children=[tree])
                return [tree], label_loss

            left_encodings = []
            right_encodings = []
            for split in range(left + 1, right + 1):
                left_encodings.append(get_span_encoding(left, split - 1))
                right_encodings.append(get_span_encoding(split, right))
            left_scores = get_split_scores(
                dy.concatenate_to_batch(left_encodings), split_parent_state, direction)
            right_scores = get_split_scores(
                dy.concatenate_to_batch(right_encodings), split_parent_state, direction)
            split_scores = left_scores + right_scores
            split_scores = dy.reshape(split_scores, (len(left_encodings),))

            oracle_splits = gold_tree.span_splits(left, right)
            oracle_split = min(oracle_splits)
            oracle_split_index = oracle_split - (left + 1)
            split_scores = MyTopDownParser3.augment(
                split_scores, oracle_split_index)

            split_scores_np = split_scores.npvalue()
            argmax_split_index = int(split_scores_np.argmax())
            argmax_split = argmax_split_index + (left + 1)

            split = argmax_split
            split_loss = (
                split_scores[argmax_split_index] -
                split_scores[oracle_split_index]
                if argmax_split != oracle_split else dy.zeros(1))

            left_split_child_state = split_parent_state.add_input(
                get_span_encoding(left, split - 1))
            right_split_child_state = split_parent_state.add_input(
                get_span_encoding(split, right))

            left_trees, left_loss = helper(
                left, split - 1, label_child_state, left_split_child_state, 0)
            right_trees, right_loss = helper(
                split, right, label_child_state, right_split_child_state, 1)

            childrens = left_trees + right_trees
            if label != 'none':
                for nt in label[6:].split('-'):
                    childrens = [PhraseTree(
                        symbol=nt, children=childrens)]

            return childrens, label_loss + split_loss + left_loss + right_loss

        label_root_state = self.label_lstm.initial_state()
        label_root_state = label_root_state.add_input(self.label_embeddings[0])
        split_root_state = self.split_lstm.initial_state()
        split_root_state = split_root_state.add_input(
            get_span_encoding(0, len(sentence) - 1))
        childrens, loss = helper(0, len(sentence) - 1,
                                 label_root_state, split_root_state, 0)
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
        def get_label_scores(left, right, parent_state, direction):
            h_hat = self.label_lstm_W_hidden[direction] * parent_state.output()
            h_hat_np = h_hat.npvalue()
            argmax_weight_index = int(h_hat_np.argmax())

            label_scores = self.f_label(get_span_encoding(
                left, right), argmax_weight_index)

            return label_scores

        @functools.lru_cache(maxsize=None)
        def get_split_scores(encodings, parent_state, direction):
            h_hat = self.split_lstm_W_hidden[direction] * parent_state.output()
            h_hat_np = h_hat.npvalue()
            argmax_weight_index = int(h_hat_np.argmax())

            split_scores = self.f_split(encodings, argmax_weight_index)

            return split_scores

        def helper(left, right, label_parent_state, split_parent_state, direction):
            label_scores = get_label_scores(
                left, right, label_parent_state, direction)

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

            label_child_state = label_parent_state.add_input(
                self.label_embeddings[argmax_label_index])

            if left == right:
                tree = PhraseTree(leaf=left)
                if label != 'none':
                    for nt in label[6:].split('-'):
                        tree = PhraseTree(symbol=nt, children=[tree])
                return [tree], label_loss

            left_encodings = []
            right_encodings = []
            for split in range(left + 1, right + 1):
                left_encodings.append(get_span_encoding(left, split - 1))
                right_encodings.append(get_span_encoding(split, right))
            left_scores = get_split_scores(
                dy.concatenate_to_batch(left_encodings), split_parent_state, direction)
            right_scores = get_split_scores(
                dy.concatenate_to_batch(right_encodings), split_parent_state, direction)
            split_scores = left_scores + right_scores
            split_scores = dy.reshape(split_scores, (len(left_encodings),))

            split_scores_np = split_scores.npvalue()
            argmax_split_index = int(split_scores_np.argmax())
            argmax_split = argmax_split_index + (left + 1)

            split = argmax_split
            split_loss = split_scores[argmax_split_index]

            left_split_child_state = split_parent_state.add_input(
                get_span_encoding(left, split - 1))
            right_split_child_state = split_parent_state.add_input(
                get_span_encoding(split, right))

            left_trees, left_loss = helper(
                left, split - 1, label_child_state, left_split_child_state, 0)
            right_trees, right_loss = helper(
                split, right, label_child_state, right_split_child_state, 1)

            childrens = left_trees + right_trees
            if label != 'none':
                for nt in label[6:].split('-'):
                    childrens = [PhraseTree(
                        symbol=nt, children=childrens)]

            return childrens, label_loss + split_loss + left_loss + right_loss

        label_root_state = self.label_lstm.initial_state()
        label_root_state = label_root_state.add_input(self.label_embeddings[0])
        split_root_state = self.split_lstm.initial_state()
        split_root_state = split_root_state.add_input(
            get_span_encoding(0, len(sentence) - 1))
        childrens, loss = helper(0, len(sentence) - 1,
                                 label_root_state, split_root_state, 0)
        assert len(childrens) == 1
        tree = childrens[0]
        tree.propagate_sentence(sentence)

        return tree
