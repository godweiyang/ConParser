import _dynet as dy
import numpy as np

from lib import *


class ShiftReduceParser(object):
    def network_init(
        self,
        struct_spans=4,
        label_spans=3
    ):
        self.word_embeddings = self.model.add_lookup_parameters(
            (self.word_count, self.word_embedding_dim),
            init='uniform',
            scale=0.01
        )
        self.char_embeddings = self.model.add_lookup_parameters(
            (self.char_count, self.char_embedding_dim),
            init='uniform',
            scale=0.01
        )
        self.tag_embeddings = self.model.add_lookup_parameters(
            (self.tag_count, self.tag_embedding_dim),
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

        self.f_struct = Feedforward(
            self.model, 2 * struct_spans * self.lstm_dim, [self.fc_hidden_dim], self.struct_out)

        self.f_label = Feedforward(
            self.model, 2 * label_spans * self.lstm_dim, [self.fc_hidden_dim], self.label_out)

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
        self.char_count = vocab.total_characters()
        self.tag_count = vocab.total_tags()
        self.word_embedding_dim = word_embedding_dim
        self.tag_embedding_dim = tag_embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.char_lstm_layers = char_lstm_layers
        self.char_lstm_dim = char_lstm_dim
        self.lstm_layers = lstm_layers
        self.lstm_dim = lstm_dim
        self.fc_hidden_dim = fc_hidden_dim
        self.struct_out = 2
        self.label_out = vocab.total_label_actions()
        self.dropout = dropout
        self.unk_param = unk_param

        self.network_init()

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

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

    def evaluate_recurrent(self, word_inds, tag_inds, is_train=True):
        embeddings = self.get_embeddings(word_inds, tag_inds, is_train)

        if self.dropout > 0 and is_train:
            self.lstm.set_dropout(self.dropout)
        else:
            self.lstm.disable_dropout()

        lstm_outputs = self.lstm.transduce(embeddings)

        return lstm_outputs

    def evaluate_struct(self, lstm_outputs, lefts, rights, is_train=True):
        fwd_span_out = []
        for left_index, right_index in zip(lefts, rights):
            fwd_span_out.append(
                lstm_outputs[right_index][:self.lstm_dim] - lstm_outputs[left_index - 1][:self.lstm_dim])
        fwd_span_vec = dy.concatenate(fwd_span_out)

        back_span_out = []
        for left_index, right_index in zip(lefts, rights):
            back_span_out.append(
                lstm_outputs[left_index][self.lstm_dim:] - lstm_outputs[right_index + 1][self.lstm_dim:])
        back_span_vec = dy.concatenate(back_span_out)

        hidden_input = dy.concatenate([fwd_span_vec, back_span_vec])

        if self.dropout > 0 and is_train:
            hidden_input = dy.dropout(hidden_input, self.dropout)

        scores = self.f_struct(hidden_input)

        return scores

    def evaluate_label(self, lstm_outputs, lefts, rights, is_train=True):
        fwd_span_out = []
        for left_index, right_index in zip(lefts, rights):
            fwd_span_out.append(
                lstm_outputs[right_index][:self.lstm_dim] - lstm_outputs[left_index - 1][:self.lstm_dim])
        fwd_span_vec = dy.concatenate(fwd_span_out)

        back_span_out = []
        for left_index, right_index in zip(lefts, rights):
            back_span_out.append(
                lstm_outputs[left_index][self.lstm_dim:] - lstm_outputs[right_index + 1][self.lstm_dim:])
        back_span_vec = dy.concatenate(back_span_out)

        hidden_input = dy.concatenate([fwd_span_vec, back_span_vec])

        if self.dropout > 0 and is_train:
            hidden_input = dy.dropout(hidden_input, self.dropout)

        scores = self.f_label(hidden_input)

        return scores

    def exploration(self, data, vocab, alpha=1.0, beta=0):
        struct_data = {}
        label_data = {}

        goldtree = data['tree']
        sentence = goldtree.sentence

        n = len(sentence)
        state = State(n)

        w = data['w']
        t = data['t']
        lstm_outputs = self.evaluate_recurrent(w, t, False)

        for step in range(2 * n - 1):
            features = state.s_features()
            if not state.can_combine():
                action = 'sh'
                correct_action = 'sh'
            elif not state.can_shift():
                action = 'comb'
                correct_action = 'comb'
            else:
                correct_action = state.s_oracle(goldtree)

                r = np.random.random()
                if r < beta:
                    action = correct_action
                else:
                    left, right = features
                    scores = self.evaluate_struct(
                        lstm_outputs,
                        left,
                        right,
                        False,
                    ).npvalue()

                    exp = np.exp(scores * alpha)
                    softmax = exp / (exp.sum())
                    r = np.random.random()

                    if r <= softmax[0]:
                        action = 'sh'
                    else:
                        action = 'comb'

            struct_data[features] = vocab.s_action_index(correct_action)
            state.take_action(action)

            features = state.l_features()
            correct_action = state.l_oracle(goldtree)
            label_data[features] = vocab.l_action_index(correct_action)

            r = np.random.random()
            if r < beta:
                action = correct_action
            else:
                left, right = features
                scores = self.evaluate_label(
                    lstm_outputs,
                    left,
                    right,
                    False,
                ).npvalue()
                if step < (2 * n - 2):
                    action_index = np.argmax(scores)
                else:
                    action_index = 1 + np.argmax(scores[1:])
                action = vocab.l_action(action_index)
            state.take_action(action)

        predicted = state.stack[0][2][0]
        predicted.propagate_sentence(sentence)

        example = {
            'w': w,
            't': t,
            'struct_data': struct_data,
            'label_data': label_data,
        }

        return example, predicted

    def parse(self, data, is_train=True):
        example, predicted = self.exploration(data, self.vocab)

        lstm_outputs = self.evaluate_recurrent(
            example['w'],
            example['t'],
        )

        loss = []

        total_states = 0
        for (left, right), correct in example['struct_data'].items():
            scores = self.evaluate_struct(lstm_outputs, left, right)
            probs = dy.softmax(scores)
            loss.append(-dy.log(dy.pick(probs, correct)))
        total_states += len(example['struct_data'])

        for (left, right), correct in example['label_data'].items():
            scores = self.evaluate_label(lstm_outputs, left, right)
            probs = dy.softmax(scores)
            loss.append(-dy.log(dy.pick(probs, correct)))
        total_states += len(example['label_data'])

        return dy.esum(loss), predicted, total_states

    def predict(self, tree):
        sentence = tree.sentence

        n = len(sentence)
        state = State(n)

        w, t = self.vocab.sentence_sequences(sentence)
        lstm_outputs = self.evaluate_recurrent(w, t, is_train=False)

        for step in range(2 * n - 1):
            if not state.can_combine():
                action = 'sh'
            elif not state.can_shift():
                action = 'comb'
            else:
                left, right = state.s_features()
                scores = self.evaluate_struct(
                    lstm_outputs,
                    left,
                    right,
                    is_train=False,
                ).npvalue()

                action_index = np.argmax(scores)
                action = self.vocab.s_action(action_index)

            state.take_action(action)

            left, right = state.l_features()
            scores = self.evaluate_label(
                lstm_outputs,
                left,
                right,
                is_train=False,
            ).npvalue()
            if step < (2 * n - 2):
                action_index = np.argmax(scores)
            else:
                action_index = 1 + np.argmax(scores[1:])
            action = self.vocab.l_action(action_index)
            state.take_action(action)

        predicted = state.stack[0][2][0]
        predicted.propagate_sentence(sentence)
        return predicted
