import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization, Activation, Dropout
# from keras.optimizers import SGD, Adam
from tensorflow.keras.optimizers import SGD, Adam
import os
import pickle


class QNetwork:
    def __init__(self, layers, s_formatter, a_shape, n_actions, n_mini_batch, hack=False):
        if hack:
            return
        activation = 'relu'
        initialization = 'uniform'

        self.s_formatter = s_formatter
        self.a_shape = a_shape
        self.n_actions = n_actions
        self.n_mini_batch = n_mini_batch

        self.model = Sequential()

        self.model.add(Dense(units=layers[0], input_dim=self.s_formatter.s_shape, kernel_initializer=initialization))
        self.model.add(Activation(activation))

        for layer in layers[1:]:
            self.model.add(Dense(units=layer, kernel_initializer=initialization))
            self.model.add(Activation(activation))

        self.model.add(Dense(units=self.n_actions))
        self.model.add(Activation('linear'))

        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.model.compile(loss='mse', optimizer=adam, metrics=['mse'])

    def fit_batch(self, x, y):
        self.model.train_on_batch(x, y)

    def predict(self, s):
        ss = s.copy()
        s_terms = np.any(np.isnan(ss), axis=1)
        ss[s_terms] = 0
        ys = self.model.predict(ss)
        ys[s_terms] = 0
        return ys

    def max_batch(self, ss_raw):
        ys = self.predict(ss_raw)
        maxes = ys.max(axis=1)
        return maxes

    def argmax_batch(self, ss_raw):
        qsas = self.predict(ss_raw)
        max_mask = qsas >= qsas.max(axis=1)[:, None]
        return [np.random.choice(np.flatnonzero(r)) for r in max_mask]

    def transfer_weights(self, other, polyak_rate):
        def polyak(me, you, tau):
            return tau * me + (1-tau) * you

        for l1, l2 in zip(self.model.layers, other.model.layers):
            w1 = l1.get_weights()
            w2 = l2.get_weights()
            for i in range(len(w1)):
                w1[i] = polyak(w1[i], w2[i], polyak_rate)
            l1.set_weights(w1)

    def save(self, path, base_name):
        if not os.path.exists(path):
            os.makedirs(path)

        with open(os.path.join(path, base_name + '_meta.pkl'), 'wb') as f:
            pickle.dump(self.s_formatter, f)
            pickle.dump(self.a_shape, f)
            pickle.dump(self.n_actions, f)

        self.model.save(os.path.join(path, base_name + '_model.h5'))

    def load(self, path, base_name):
        with open(os.path.join(path, base_name + '_meta.pkl'), 'rb') as f:
            self.s_formatter = pickle.load(f)
            self.a_shape = pickle.load(f)
            self.n_actions = pickle.load(f)

        self.model = load_model(os.path.join(path, base_name + '_model.h5'))
