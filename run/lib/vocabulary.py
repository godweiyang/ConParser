import sys
import json
from collections import defaultdict, OrderedDict

import numpy as np

from .trees import PhraseTree
from .state import State


class Vocabulary(object):
    UNK = '<UNK>'
    START = '<START>'
    STOP = '<STOP>'

    def __init__(self, vocabfile, verbose=True):
        if vocabfile is not None:
            word_freq = defaultdict(int)
            char_freq = defaultdict(int)
            tag_freq = defaultdict(int)
            label_freq = defaultdict(int)

            trees = PhraseTree.load_treefile(vocabfile)

            for i, tree in enumerate(trees):
                for (word, tag) in tree.sentence:
                    word_freq[word] += 1
                    tag_freq[tag] += 1
                    for char in word:
                        char_freq[char] += 1

                for action in State.gold_actions(tree):
                    if action.startswith('label-'):
                        label = action[6:]
                        label_freq[label] += 1

                if verbose:
                    print('\rTree {}'.format(i), end='')
                    sys.stdout.flush()

            if verbose:
                print('\r', end='')

            i2w = [
                Vocabulary.UNK,
                Vocabulary.START,
                Vocabulary.STOP,
            ] + sorted(word_freq)
            w2i = OrderedDict((w, i) for (i, w) in enumerate(i2w))

            i2c = [
                Vocabulary.UNK,
                Vocabulary.START,
                Vocabulary.STOP,
            ] + sorted(char_freq)
            c2i = OrderedDict((c, i) for (i, c) in enumerate(i2c))

            i2t = [
                Vocabulary.UNK,
                Vocabulary.START,
                Vocabulary.STOP,
            ] + sorted(tag_freq)
            t2i = OrderedDict((t, i) for (i, t) in enumerate(i2t))

            i2l = sorted(label_freq)
            # for i in i2l:
            #     print(i)
            l2i = OrderedDict((l, i) for (i, l) in enumerate(i2l))

            if verbose:
                print('Loading vocabularies from {}'.format(vocabfile))
                print('({} words, {} characters, {} tags, {} nonterminal-chains)'.format(
                    len(w2i),
                    len(c2i),
                    len(t2i),
                    len(l2i),
                ))

            self.w2i = w2i
            self.i2w = i2w
            self.word_freq = word_freq
            self.c2i = c2i
            self.i2c = i2c
            self.t2i = t2i
            self.i2t = i2t
            self.l2i = l2i
            self.i2l = i2l

            self.word_freq_list = []
            for word in self.i2w:
                if word in self.word_freq:
                    self.word_freq_list.append(self.word_freq[word])
                else:
                    self.word_freq_list.append(0)

    def total_words(self):
        return len(self.w2i)

    def total_characters(self):
        return len(self.c2i)

    def total_tags(self):
        return len(self.t2i)

    def total_label_actions(self):
        return 1 + len(self.l2i)

    def s_action_index(self, action):
        if action == 'sh':
            return 0
        elif action == 'comb':
            return 1
        else:
            raise ValueError('Not s-action: {}'.format(action))

    def l_action_index(self, action):
        if action == 'none':
            return 0
        elif action.startswith('label-'):
            label = action[6:]
            label_index = self.l2i.get(label, None)
            if label_index is not None:
                return 1 + label_index
            else:
                return 0
        else:
            raise ValueError('Not l-action: {}'.format(action))

    def s_action(self, index):
        return ('sh', 'comb')[index]

    def l_action(self, index):
        if index == 0:
            return 'none'
        else:
            return 'label-' + self.i2l[index - 1]

    def sentence_sequences(self, sentence):
        sentence = (
            [(Vocabulary.START, Vocabulary.START)] +
            sentence +
            [(Vocabulary.STOP, Vocabulary.STOP)]
        )

        words_indices = [
            self.w2i[w]
            if w in self.w2i else self.w2i[Vocabulary.UNK]
            for (w, t) in sentence
        ]
        tags_indices = [
            self.t2i[t]
            if t in self.t2i else self.t2i[Vocabulary.UNK]
            for (w, t) in sentence
        ]

        return words_indices, tags_indices

    def get_heads(self, tree, is_right_tree=False):
        if len(tree.children) == 1:
            self.get_heads(tree.children[0], is_right_tree)
            return

        l = tree.left_span()
        r = tree.right_span()
        if not is_right_tree:
            self.heads[r + 1] = l

        if l == r:
            return

        for c in tree.children[:-1]:
            self.get_heads(c, False)

        self.get_heads(tree.children[-1], True)

    def gold_data(self, goldtree, idx):
        w, t = self.sentence_sequences(goldtree.sentence)

        (s_features, l_features) = State.training_data(goldtree)

        struct_data = {}
        for (features, action) in s_features:
            struct_data[features] = self.s_action_index(action)

        label_data = {}
        for (features, action) in l_features:
            label_data[features] = self.l_action_index(action)

        self.heads = [None] * (len(goldtree.sentence) + 1)
        self.heads[0] = -1

        self.get_heads(goldtree)

        return {
            'idx': idx,
            'tree': goldtree,
            'w': w,
            't': t,
            'struct_data': struct_data,
            'label_data': label_data,
            'heads': self.heads,
        }

    def gold_data_from_file(self, fname):
        trees = PhraseTree.load_treefile(fname)
        result = []
        for idx, tree in enumerate(trees):
            sentence_data = self.gold_data(tree, idx)
            result.append(sentence_data)
        return result

    def load_bert_embeddings(self, fname):
        sent_emb_np = []
        with open(fname) as fin:
            for line in fin:
                sent_emb = eval(line)
                sent_emb_np.append(np.array(sent_emb, dtype=np.float32))

        return sent_emb_np
