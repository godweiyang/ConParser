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

        self.W_H = self.model.add_parameters(
            (2 * self.lstm_dim, 2 * self.lstm_dim))
        self.b1 = self.model.add_parameters((2 * self.lstm_dim, ))
        self.b2 = self.model.add_parameters((2 * self.lstm_dim, ))
        self.W_A = self.model.add_parameters(
            (2 * self.lstm_dim, 2 * self.lstm_dim))
        self.B = self.model.add_parameters(
            (2 * self.lstm_dim, 2 * self.lstm_dim))
        self.W_l = self.model.add_parameters(
            (2 * self.lstm_dim, 2 * self.lstm_dim))
        self.W_r = self.model.add_parameters(
            (2 * self.lstm_dim, 2 * self.lstm_dim))
        self.W = self.model.add_parameters(
            (self.word_count, 2 * self.lstm_dim))
        self.b = self.model.add_parameters((self.word_count, ))

    def eisner_parser(self, n, scores):

        INF = np.inf

        comp_rh = np.empty((n, n), dtype=np.float)  # C[i][j][←][0]
        comp_lh = np.empty((n, n), dtype=np.float)  # C[i][j][→][0]
        incomp_rh = np.empty((n, n), dtype=np.float)  # C[i][j][←][1]
        incomp_lh = np.empty((n, n), dtype=np.float)  # C[i][j][→][1]

        bp_comp_rh = np.empty((n, n), dtype=np.int)
        bp_comp_lh = np.empty((n, n), dtype=np.int)
        bp_incomp_rh = np.empty((n, n), dtype=np.int)
        bp_incomp_lh = np.empty((n, n), dtype=np.int)

        for i in range(0, n):
            comp_rh[i][i] = 0
            comp_lh[i][i] = 0
            incomp_rh[i][i] = 0
            incomp_lh[i][i] = 0

        for m in range(1, n):
            for i in range(0, n):
                j = i + m
                if j >= n:
                    break

                # C[i][j][←][1] : right head, incomplete
                maxx = -INF
                bp = -1
                ex = scores[i][j]
                for k in range(i, j):
                    score = comp_lh[i][k] + comp_rh[k + 1][j] + ex
                    if score > maxx:
                        maxx = score
                        bp = k
                incomp_rh[i][j] = maxx
                bp_incomp_rh[i][j] = bp

                # C[i][j][→][1] : left head, incomplete
                maxx = -INF
                bp = -1
                ex = scores[j][i]
                for k in range(i, j):
                    score = comp_lh[i][k] + comp_rh[k + 1][j] + ex
                    if score > maxx:
                        maxx = score
                        bp = k
                incomp_lh[i][j] = maxx
                bp_incomp_lh[i][j] = bp

                # C[i][j][←][0] : right head, complete
                maxx = -INF
                bp = -1
                for k in range(i, j):
                    score = comp_rh[i][k] + incomp_rh[k][j]
                    if score > maxx:
                        maxx = score
                        bp = k
                comp_rh[i][j] = maxx
                bp_comp_rh[i][j] = bp

                # C[i][j][→][0] : left head, complete
                maxx = -INF
                bp = -1
                for k in range(i + 1, j + 1):
                    score = incomp_lh[i][k] + comp_lh[k][j]
                    if score > maxx:
                        maxx = score
                        bp = k
                comp_lh[i][j] = maxx
                bp_comp_lh[i][j] = bp

        heads = [None] * n
        heads[0] = -1

        def _backtrack(i, j, lh, c):
            """
                lh: right head = 0, left head = 1
                c: complete = 0, incomplete = 1
            """
            if i == j:
                return
            elif lh == 1 and c == 0:  # comp_lh
                k = bp_comp_lh[i][j]
                heads[k] = i
                heads[j] = k
                _backtrack(i, k, 1, 1)
                _backtrack(k, j, 1, 0)
            if lh == 0 and c == 0:  # comp_rh
                k = bp_comp_rh[i][j]
                heads[k] = j
                heads[i] = k
                _backtrack(i, k, 0, 0)
                _backtrack(k, j, 0, 1)
            elif lh == 1 and c == 1:  # incomp_lh
                k = bp_incomp_lh[i][j]
                heads[j] = i
                _backtrack(i, k, 1, 0)
                _backtrack(k + 1, j, 0, 0)
            elif lh == 0 and c == 1:  # incomp_rh
                k = bp_incomp_rh[i][j]
                heads[i] = j
                _backtrack(i, k, 1, 0)
                _backtrack(k + 1, j, 0, 0)

        _backtrack(0, (n - 1), 1, 0)
        return heads

    def get_spans(self, heads, l, r):
        if l + 1 == r:
            return [(l, l)]
        for k in range(r - 1, l, -1):
            # if heads[k] == l:
            if heads[k] <= l:
                left_spans = self.get_spans(heads, l, k)
                right_spans = self.get_spans(heads, k, r)
                break

        spans = left_spans + right_spans + [(l, r - 1)]
        return spans

    def check(self, heads):
        for i in range(len(heads)):
            if heads[i] >= i:
                print(i, heads[i])
            assert heads[i] < i
            for j in range(heads[i] + 1, i):
                assert heads[j] >= heads[i]

    def parse(self, data, is_train=False):
        if is_train:
            self.lstm.set_dropout(self.dropout)
        else:
            self.lstm.disable_dropout()

        word_indices = data['w']
        tag_indices = data['t']
        gold_tree = data['tree']
        heads = data['heads']
        sentence = gold_tree.sentence

        embeddings = self.get_embeddings(word_indices, tag_indices, is_train)
        lstm_outputs = self.lstm.transduce(embeddings)

        node_representations = []
        for i in range(len(lstm_outputs) - 1):
            forward = lstm_outputs[i + 1][:self.lstm_dim] - \
                lstm_outputs[i][:self.lstm_dim]
            backward = lstm_outputs[i][self.lstm_dim:] - \
                lstm_outputs[i + 1][self.lstm_dim:]
            node_representations.append(dy.concatenate([forward, backward]))
        assert len(node_representations) == len(sentence) + 1

        mask_np = np.zeros((len(sentence) + 1, len(sentence) + 1))
        for i in range(1, len(sentence)):
            for j in range(i, len(sentence) + 1):
                mask_np[i][j] = 99999
        for j in range(1, len(sentence) + 1):
            mask_np[len(sentence)][j] = 99999
            mask_np[0][j] = 99999

        mask = dy.inputTensor(mask_np)

        H0 = dy.transpose(dy.concatenate(node_representations, d=1))

        Att1 = dy.softmax(H0 * self.W_H * dy.transpose(H0) + H0 *
                          self.b1 + dy.transpose(H0 * self.b2) - mask, d=1)
        H1 = leaky_relu(Att1 * H0 * self.W_A + H0 * self.B)

        Att2 = dy.softmax(H1 * self.W_H * dy.transpose(H1) + H1 *
                          self.b1 + dy.transpose(H1 * self.b2) - mask, d=1)
        H2 = leaky_relu(Att2 * H1 * self.W_A + H1 * self.B)

        Att3 = dy.softmax(H2 * self.W_H * dy.transpose(H2) + H2 *
                          self.b1 + dy.transpose(H2 * self.b2) - mask, d=1)

        assert Att3[0].dim()[0][0] == len(heads)

        if is_train:
            losses = []
            idx = 0
            for w_s, w_t in zip(word_indices[:-1], word_indices[1:-1]):
                h_t = H2[idx]
                scores = self.W * h_t + self.b
                loss = dy.pickneglogsoftmax(scores, int(w_t))
                losses.append(loss)
                idx += 1

            return dy.esum(losses), None, 1
        else:
            heads_predicted = self.eisner_parser(
                len(sentence) + 1, Att3.value())
            # self.check(heads_predicted)

            # spans = self.get_spans(heads_predicted, 0, len(sentence))
            spans = heads_predicted
            return spans
