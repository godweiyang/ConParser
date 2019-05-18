import functools
import time
from collections import defaultdict

import _dynet as dy
import numpy as np

from lib import *


class BaseParser(object):
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

        self.historical_info_lstm = dy.VanillaLSTMBuilder(
            2, self.label_embedding_dim, self.lstm_dim, self.model)

        self.f_label = Feedforward(
            self.model, 2 * self.lstm_dim, [self.fc_hidden_dim], self.label_out)

        self.f_split = Feedforward(
            self.model, 2 * self.lstm_dim, [self.fc_hidden_dim], 1)

        self.W_G = self.model.add_parameters(
            (2 * self.lstm_dim, 2 * self.lstm_dim))

        self.W_GCN = self.model.add_parameters(
            (2 * self.lstm_dim, 2 * self.lstm_dim))

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

        self.span_prob_threshold = 0.5

        self.network_init()

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    @staticmethod
    def augment(scores, oracle_index, crossing=False):
        '''
            Add the hinge loss into scores.
        '''

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

    def get_span_encoding(self, lstm_outputs, left, right):
        '''
            Get the span representation using the difference of lstm_outputs of left and right.
        '''

        forward = (
            lstm_outputs[right + 1][:self.lstm_dim] -
            lstm_outputs[left][:self.lstm_dim])
        backward = (
            lstm_outputs[left + 1][self.lstm_dim:] -
            lstm_outputs[right + 2][self.lstm_dim:])
        return dy.concatenate([forward, backward])

    def get_label_scores(self, lstm_outputs, left, right):
        '''
            Get label scores and fix the score of empty label to zero.
        '''

        non_empty_label_scores = self.f_label(
            self.get_span_encoding(lstm_outputs, left, right))
        return dy.concatenate([dy.zeros(1), non_empty_label_scores])

    @staticmethod
    def tree2graph(tree):
        '''
            Return the upper triangular adjacency matrix of the tree.
        '''

        G_np = np.zeros((len(tree.sentence), len(tree.sentence)))
        for i in range(len(tree.sentence)):
            for j in range(i, len(tree.sentence)):
                if i == j:
                    G_np[i][j] = 1
                label, crossing = tree.span_labels(i, j)
                label = label[::-1]
                if (len(label) > 0):
                    G_np[i, j] = 1

        G = dy.inputTensor(G_np)
        return G

    def gen_legal_graph(self, G):
        '''
            Transfer any 0~1 matrix to legal upper triangular adjacency matrix.
        '''

        G_np = G.value()
        length = len(G_np[0])
        G_np[0][length - 1] = 1
        for i in range(length):
            G_np[i][i] = 1

        forbidden = defaultdict(int)
        for right in range(length):
            for left in range(right, -1, -1):
                if G_np[left][right] > self.span_prob_threshold and forbidden[left] == 0:
                    G_np[left][right] = 1
                    for i in range(left + 1, right + 1):
                        forbidden[i] = 1
                else:
                    G_np[left][right] = 0
                G_np[right][left] = G_np[left][right]
        return G_np

    def GCN(self, A, H):
        '''
            GCN: H_2 = RELU(D^{-0.5} * A * D^{-0.5} * H * W_GCN)
        '''

        D = np.diag(np.power(np.sum(A, axis=0), -0.5))
        C = np.dot(np.dot(D, A), D)
        H_2 = dy.rectify(dy.inputTensor(C) * H * self.W_GCN)
        return H_2

    def gen_leaf_tree(self, index, label):
        '''
            Generate subtree in the leaf.
        '''

        tree = PhraseTree(leaf=index)
        if label != 'none':
            for nt in label[6:].split('-'):
                tree = PhraseTree(
                    symbol=nt, children=[tree])
        return tree

    def gen_nonleaf_tree(self, childrens, label):
        '''
            Generate subtree which has several childrens.
        '''

        if label != 'none':
            for nt in label[6:].split('-'):
                childrens = [PhraseTree(
                    symbol=nt, children=childrens)]
        return childrens

    def get_oracle_label(self, gold_tree, left, right):
        '''
            Get oracle label of span (left, right)
        '''

        oracle_label, crossing = gold_tree.span_labels(left, right)
        oracle_label = oracle_label[::-1]

        if len(oracle_label) == 0:
            oracle_label = 'none'
        else:
            oracle_label = 'label-' + '-'.join(oracle_label)
        oracle_label_index = self.vocab.l_action_index(oracle_label)

        return oracle_label, oracle_label_index, crossing

    def predict_label(self, label_scores, gold_tree, left, right):
        '''
            Predict the best label using the label scores.
        '''

        label_scores_np = label_scores.npvalue()
        argmax_label_index = int(
            label_scores_np.argmax() if right - left < len(gold_tree.sentence) - 1 else
            label_scores_np[1:].argmax() + 1)

        if argmax_label_index == 0:
            argmax_label = 'none'
        else:
            argmax_label = 'label-' + self.vocab.i2l[argmax_label_index - 1]

        return argmax_label, argmax_label_index

    def get_right_boundary_scores(self, lstm_outputs, left, split, right_bound):
        '''
            Predict the best right boundary point.
        '''

        left_encodings = []
        right_encodings = []
        for right in range(split + 1, right_bound + 1):
            left_encodings.append(
                self.get_span_encoding(lstm_outputs, left, right))
            right_encodings.append(self.get_span_encoding(
                lstm_outputs, split + 1, right))
        left_scores = self.f_split(dy.concatenate_to_batch(left_encodings))
        right_scores = self.f_split(
            dy.concatenate_to_batch(right_encodings))
        parent_right_scores = left_scores + right_scores
        parent_right_scores = dy.reshape(
            parent_right_scores, (len(left_encodings),))
        return parent_right_scores

    def get_split_scores(self, lstm_outputs, left, right):
        '''
            Predict the best split point.
        '''

        left_encodings = []
        right_encodings = []
        for split in range(left + 1, right + 1):
            left_encodings.append(self.get_span_encoding(
                lstm_outputs, left, split - 1))
            right_encodings.append(
                self.get_span_encoding(lstm_outputs, split, right))
        left_scores = self.f_split(dy.concatenate_to_batch(left_encodings))
        right_scores = self.f_split(
            dy.concatenate_to_batch(right_encodings))
        split_scores = left_scores + right_scores
        split_scores = dy.reshape(split_scores, (len(left_encodings),))
        return split_scores

    def gen_tree_syn_dis(self, syntactic_distance, l, r):
        span_set = [(l, r)]
        if l == r:
            return span_set
        k = syntactic_distance[l:r].index(max(syntactic_distance[l:r])) + l
        span_set_left = self.gen_tree_syn_dis(syntactic_distance, l, k)
        span_set_right = self.gen_tree_syn_dis(syntactic_distance, k + 1, r)
        span_set += span_set_left + span_set_right
        return span_set
