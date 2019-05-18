import time

import numpy as np
import _dynet as dy


def leaky_relu(x):
    return dy.bmax(.1 * x, x)


class Feedforward(object):
    def __init__(self, model, input_dim, hidden_dims, output_dim):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model

        self.weights = []
        self.biases = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i, (prev_dim, next_dim) in enumerate(zip(dims, dims[1:])):
            W = self.model.add_parameters((next_dim, prev_dim))
            b = self.model.add_parameters((next_dim,), init=0)
            self.weights.append(W)
            self.biases.append(b)

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def __call__(self, x):
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = dy.affine_transform([bias, weight, x])
            if i < len(self.weights) - 1:
                x = dy.rectify(x)
        return x


class MultiWeightedFeedforward(object):
    def __init__(self, model, input_dim, hidden_dims, output_dim, weight_counts):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model

        self.weights = []
        self.biases = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i, (prev_dim, next_dim) in enumerate(zip(dims, dims[1:])):
            W = []
            for i in range(weight_counts):
                W.append(self.model.add_parameters((next_dim, prev_dim)))
            b = []
            for i in range(weight_counts):
                b.append(self.model.add_parameters((next_dim,), init=0))
            self.weights.append(W)
            self.biases.append(b)

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def __call__(self, x, weight_index):
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = dy.affine_transform(
                [bias[weight_index], weight[weight_index], x])
            if i < len(self.weights) - 1:
                x = dy.rectify(x)
        return x


class OutputLayer(object):
    def __init__(self, model, input_dim, hidden_dim, output_dim):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model

        self.H = model.add_parameters((hidden_dim, input_dim))
        self.O = model.add_parameters((output_dim, hidden_dim))

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def __call__(self, x):
        y = O * (dy.tanh(H * x))
        return y


class SingleHeadSelfAttentive(object):
    def __init__(
        self,
        model,
        d_model,
        d_k,
        d_v,
    ):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q = self.model.add_parameters((d_model, d_k))
        self.W_K = self.model.add_parameters((d_model, d_k))
        self.W_V = self.model.add_parameters((d_model, d_v))
        self.W_O = self.model.add_parameters((d_v, d_model))

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def __call__(self, X):
        Q = X * W_Q
        K = X * W_K
        V = X * W_V
        QK = Q * dy.transpose(K) / np.sqrt(self.d_k)
        QKV = dy.softmax(QK, 1) * V * W_O
        return QKV


class MultiHeadSelfAttentive(object):
    def __init__(
        self,
        model,
        d_model,
        d_k,
        d_v,
    ):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model
        self.d_model = d_model

        self.feedforward = Feedforward(
            self.model, self.d_model, [self.d_model], self.d_model)
        self.attention = []
        for i in range(8):
            self.attention.append(SingleHeadSelfAttentive(
                self.model, d_model, d_k, d_v))

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def __call__(self, X):
        d_x = X.dim()[0][0]
        d_y = X.dim()[0][1]
        g = dy.ones((d_x, d_y))
        b = dy.zeros((d_x, d_y))
        Y = []
        for attention in self.attention:
            Y.append(attention(X))
        Y = dy.esum(Y)
        Y = dy.layer_norm(X + Y, g, b)
        Y = dy.layer_norm(
            Y + dy.transpose(self.feedforward(dy.transpose(Y))), g, b)
        return Y


class AttentionEncoder(object):
    def __init__(
        self,
        model,
        d_model,
        d_k,
        d_v,
    ):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model

        self.attention = []
        for i in range(8):
            self.attention.append(MultiHeadSelfAttentive(
                self.model, d_model, d_k, d_v))

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def __call__(self, X):
        for attention in self.attention:
            X = attention(X)
        return X


def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string
