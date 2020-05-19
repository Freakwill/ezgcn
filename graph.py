#!/usr/bin/env python3
#
import numpy as np
from keras import activations, initializers, constraints
from keras import regularizers
from keras.engine import Layer
import keras.backend as K
from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam

from utils import *


class GraphConvolution(Layer):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907"""
    def __init__(self, units, support=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True

        self.support = support
        assert support >= 1

    def compute_output_shape(self, input_shapes):
        features_shape = input_shapes[0]
        output_shape = (features_shape[0], self.units)
        return output_shape  # (batch_size, output_dim)

    def build(self, input_shapes):
        features_shape = input_shapes[0]
        assert len(features_shape) == 2
        input_dim = features_shape[1]

        self.kernel = self.add_weight(shape=(input_dim * self.support,
                                             self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, mask=None):
        features = inputs[0]
        basis = inputs[1:]

        supports = [K.dot(basis[i], features) for i in range(self.support)]
        supports = K.concatenate(supports, axis=1)
        output = K.dot(supports, self.kernel)

        if self.bias is not None:
            output += self.bias
        return self.activation(output)

    def get_config(self):
        config = {'units': self.units,
                  'support': self.support,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'bias_initializer': initializers.serialize(
                      self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(
                      self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint)
        }

        c = super(GraphConvolution, self).get_config() # base config
        c.update(config)
        return c


localpool = {'sym_norm':True}
chebyshev = {'max_degree': 2, 'sym_norm':True}

class GCN(Model):
    def __init__(self, input_dim, output_dim, adj_matrix=None, filter=localpool):
        if filter == localpool:
            # Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016)
            support = 1
            G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]

        elif filter == chebyshev:
            # Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)
            support = MAX_DEGREE + 1
            G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)
                 for _ in range(support)]
        else:
            raise Exception('Invalid filter type.')

        X_in = Input(shape=(input_dim,))
        H = Dropout(rate=0.5)(X_in)
        H = GraphConvolution(16, support, activation='relu',
                             kernel_regularizer=regularizers.l2(5e-4))([H]+G)
        H = Dropout(rate=0.5)(H)
        Y = GraphConvolution(output_dim, support, activation='softmax')([H]+G)
        super(GCN, self).__init__(inputs=[X_in]+G, outputs=Y)
        self.adj_matrix = adj_matrix
        self.filter = filter


    @classmethod
    def from_data(cls, X_train, y_train, X_test=None, adj_matrix=None, similarity=None, *args, **kwargs):
        input_dim = X_train.shape[1]
        output_dim = y_train.shape[1]
        model = cls(input_dim, output_dim, *args, **kwargs)
        if X_test is not None:
            model.make_graph(X_train, X_test, adj_matrix=adj_matrix, similarity=similarity)
        return model

    def make_graph(self, X_train, X_test, adj_matrix=None, similarity=None):
        if adj_matrix is not None:
            self.adj_matrix = adj_matrix
        elif similarity is not None:
            self.adj_matrix = make_adj(X_train, X_test, similarity=cosine)
        elif not hasattr(self, 'adj_matrix'):
            raise Exception('Plz supply adj_matrix!')

        X = np.vstack((X_train, X_test))
        if self.filter == localpool:
            # Local pooling filters (see 'renormalization trick' in Kipf & Welling, arXiv 2016)
            A_ = preprocess_adj(self.adj_matrix, self.filter['sym_norm'])
            self.graph = [X, A_]

        elif self.filter == chebyshev:
            # Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)
            L = normalized_laplacian(self.adj_matrix, self.filter['sym_norm'])
            L_scaled = rescale_laplacian(L)
            T_k = chebyshev_polynomial(L_scaled, self.filter['max_degree'])
            self.graph = [X]+T_k

        self.adj_size = X.shape[0]


    def fit(self, X_train, y_train, X_test=None, adj_matrix=None, similarity=None, val_rate=0.2, *args, **kwargs):
        if not hasattr(self, 'graph'):
            if X_test is not None:
                self.make_graph(X_train, X_test, adj_matrix=adj_matrix, similarity=similarity)
            else:
                raise Exception('run make_graph method before fitting!')

        NX = X_train.shape[0]
        NX_ = self.adj_size - NX
        Ny = y_train.shape[1]
        mask = random_mask(val_rate, NX)
        val_mask = np.hstack((mask, np.zeros(NX_, dtype=np.bool)))
        train_mask = np.hstack((1-mask, np.zeros(NX_, dtype=np.bool)))
        self.batch_size = self.adj_matrix.shape[0]
        y_train = np.vstack((y_train, np.zeros((NX_, Ny), dtype=np.int32)))
        validation_data = (self.graph, y_train, val_mask)

        super(GCN, self).fit(self.graph, y_train, sample_weight=train_mask,
            validation_data=validation_data, batch_size=self.batch_size,
            *args, **kwargs)

    def ezfit(self, X, y, X_test):
        pass

    def evaluate(self, X, y):
        NX_ = X.shape[0]
        test_mask = np.hstack((np.zeros(self.adj_size-NX_), np.ones(NX_)))
        eval_results = model.evaluate(self.graph, y, sample_weight=test_mask,
                              batch_size=self.batch_size)


    def predict(self, X=None, *args, **kwargs):
        return super(GCN, self).predict(self.graph, batch_size=self.batch_size, *args, **kwargs)

