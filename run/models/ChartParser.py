import functools
import time

import _dynet as dy
import numpy as np

from lib import *


class ChartParser(object):
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

        self.f_label = Feedforward(
            self.model, 2 * self.lstm_dim, [self.fc_hidden_dim], self.label_out - 1)

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
    def augment(scores, oracle_index):
        assert isinstance(scores, dy.Expression)
        shape = scores.dim()[0]
        assert len(shape) == 1
        increment = np.ones(shape)
        increment[oracle_index] = 0
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
        def get_label_scores(left, right):
            non_empty_label_scores = self.f_label(
                get_span_encoding(left, right))
            return dy.concatenate([dy.zeros(1), non_empty_label_scores])

        def helper(force_gold):
            chart = {}

            for length in range(1, len(sentence) + 1):
                for left in range(0, len(sentence) + 1 - length):
                    right = left + length - 1
                    label_scores = get_label_scores(left, right)
                    oracle_label, crossing = gold_tree.span_labels(left, right)
                    oracle_label = oracle_label[::-1]

                    if len(oracle_label) == 0:
                        oracle_label = 'none'
                    else:
                        oracle_label = 'label-' + '-'.join(oracle_label)
                    oracle_label_index = self.vocab.l_action_index(
                        oracle_label)

                    if force_gold:
                        label = oracle_label
                        label_score = label_scores[oracle_label_index]
                    else:
                        label_scores = ChartParser.augment(
                            label_scores, oracle_label_index)
                        label_scores_np = label_scores.npvalue()
                        argmax_label_index = int(
                            label_scores_np.argmax() if length < len(sentence) else
                            label_scores_np[1:].argmax() + 1)

                        if argmax_label_index == 0:
                            argmax_label = 'none'
                        else:
                            argmax_label = 'label-' + \
                                self.vocab.i2l[argmax_label_index - 1]

                        label = argmax_label
                        label_score = label_scores[argmax_label_index]

                    if length == 1:
                        tree = PhraseTree(leaf=left)
                        if label != 'none':
                            for nt in label[6:].split('-'):
                                tree = PhraseTree(symbol=nt, children=[tree])
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
                    if label != 'none':
                        for nt in label[6:].split('-'):
                            childrens = [PhraseTree(
                                symbol=nt, children=childrens)]

                    chart[left, right] = (
                        childrens, label_score + left_score + right_score)

            childrens, score = chart[0, len(sentence) - 1]
            assert len(childrens) == 1
            return childrens[0], score

        # t1 = time.time()
        tree, score = helper(False)
        # t2 = time.time()
        # print("\nt1-2: {:.6f}".format(t2 - t1))

        oracle_tree, oracle_score = helper(True)

        tree.propagate_sentence(sentence)
        oracle_tree.propagate_sentence(sentence)

        assert str(oracle_tree) == str(gold_tree)

        correct = (str(tree) == str(oracle_tree))
        loss = dy.zeros(1) if correct else score - oracle_score

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
        def get_label_scores(left, right):
            non_empty_label_scores = self.f_label(
                get_span_encoding(left, right))
            return dy.concatenate([dy.zeros(1), non_empty_label_scores])

        def helper():
            chart = {}
            for length in range(1, len(sentence) + 1):
                for left in range(0, len(sentence) + 1 - length):
                    right = left + length - 1

                    label_scores = get_label_scores(left, right)

                    label_scores_np = label_scores.npvalue()
                    argmax_label_index = int(
                        label_scores_np.argmax() if length < len(sentence) else
                        label_scores_np[1:].argmax() + 1)

                    if argmax_label_index == 0:
                        argmax_label = 'none'
                    else:
                        argmax_label = 'label-' + \
                            self.vocab.i2l[argmax_label_index - 1]
                    label = argmax_label
                    label_score = label_scores[argmax_label_index]

                    if length == 1:
                        tree = PhraseTree(leaf=left)
                        if label != 'none':
                            for nt in label[6:].split('-'):
                                tree = PhraseTree(symbol=nt, children=[tree])
                        chart[left, right] = [tree], label_score
                        continue

                    best_split = max(
                        range(left + 1, right + 1),
                        key=lambda split:
                            chart[left, split - 1][1].value() +
                            chart[split, right][1].value())

                    left_trees, left_score = chart[left, best_split - 1]
                    right_trees, right_score = chart[best_split, right]

                    childrens = left_trees + right_trees
                    if label != 'none':
                        for nt in label[6:].split('-'):
                            childrens = [PhraseTree(
                                symbol=nt, children=childrens)]

                    chart[left, right] = (
                        childrens, label_score + left_score + right_score)

            childrens, score = chart[0, len(sentence) - 1]
            assert len(childrens) == 1
            return childrens[0], score

        tree, score = helper()
        tree.propagate_sentence(sentence)

        return tree
