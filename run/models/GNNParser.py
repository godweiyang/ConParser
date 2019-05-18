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


class GNNParser2(BaseParser):
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

    def get_span_encoding(self, H, l, r):
        return H[l] + H[r + 1]

    def get_label_loss(self, H, gold_tree):
        if len(gold_tree.children) == 1:
            return self.get_label_loss(H, gold_tree.children[0])
        losses = []
        l = gold_tree.left_span()
        r = gold_tree.right_span()
        label_scores = self.f_label(self.get_span_encoding(H, l, r))
        _, oracle_label_index, _ = self.get_oracle_label(gold_tree, l, r)
        losses.append(dy.pickneglogsoftmax(label_scores, oracle_label_index))

        if l == r:
            return losses

        for c in gold_tree.children[1:-1]:
            cl = c.left_span()
            label_scores = self.f_label(
                self.get_span_encoding(H, cl, r))
            losses.append(dy.pickneglogsoftmax(label_scores, 0))
        for c in gold_tree.children:
            losses += self.get_label_loss(H, c)

        return losses

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

    def gen_tree_by_heads(self, H, gold_tree, heads, l, r):
        label_scores = self.f_label(
            self.get_span_encoding(H, l, r - 1))
        argmax_label, _ = self.predict_label(
            label_scores, gold_tree, l, r - 1)
        if l + 1 == r:
            tree = self.gen_leaf_tree(l, argmax_label)
            return [tree]
        for k in range(r - 1, l, -1):
            if heads[k] == l:
                left_tree = self.gen_tree_by_heads(
                    H, gold_tree, heads, l, k)
                right_tree = self.gen_tree_by_heads(
                    H, gold_tree, heads, k, r)
                break

        childrens = left_tree + right_tree
        tree = self.gen_nonleaf_tree(childrens, argmax_label)
        return tree

    def check(self, heads):
        for i in range(len(heads)):
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
        # print(gold_tree)
        # print(heads)
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
        for i in range(len(sentence)):
            for j in range(i, len(sentence) + 1):
                mask_np[i][j] = 99999
        for j in range(1, len(sentence) + 1):
            mask_np[len(sentence)][j] = 99999

        mask = dy.inputTensor(mask_np)

        # (n, d)
        H0 = dy.transpose(dy.concatenate(node_representations, d=1))
        # print(H.dim(), self.b2.dim())
        Att1 = dy.softmax(H0 * self.W_H * dy.transpose(H0) + H0 *
                          self.b1 + dy.transpose(H0 * self.b2) - mask, d=1)
        H1 = leaky_relu(Att1 * H0 * self.W_A + H0 * self.B)

        Att2 = dy.softmax(H1 * self.W_H * dy.transpose(H1) + H1 *
                          self.b1 + dy.transpose(H1 * self.b2) - mask, d=1)

        # print(Att.value())
        # print(Att[0].dim()[0][0], len(heads))
        assert Att2[0].dim()[0][0] == len(heads)

        # childrens, loss = helper(0, 0, len(sentence) - 1, None, None)
        # assert len(childrens) == 1
        # tree = childrens[0]
        # tree.propagate_sentence(sentence)

        if is_train:
            losses = []
            losses1 = []
            for i in range(1, len(heads)):
                losses.append(-dy.log(Att2[i][heads[i]]))
                losses1.append(-dy.log(Att1[i][heads[i]]))

            losses += self.get_label_loss(H1, gold_tree)
            loss = dy.esum(losses) + 0.5 * dy.esum(losses1)
            # loss = dy.esum(losses)

            return loss, None, 1
        else:
            heads_predicted = self.eisner_parser(
                len(sentence) + 1, Att2.value())
            self.check(heads_predicted)
            childrens = self.gen_tree_by_heads(
                H1, gold_tree, heads_predicted, 0, len(sentence))
            tree = childrens[0]
            tree.propagate_sentence(sentence)
            return tree
