import numpy as np

from . import dqn, qn_keras


class RandomAgent:
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def start_state(self, s0):
        return

    def next_action(self):
        return np.random.choice(self.n_actions)

    def next_reading(self, sp, r, done, train=True):
        return


class TrainedAgent:
    def __init__(self, agent_path, agent_name, n_actions, shape_observation):
        self.a_shape = 1
        self.n_actions = n_actions

        self.s_formatter = dqn.StateFormatter(shape_observation[0])
        self.s = None
        self.a = None
        self.t = 0
        self.seq_num = 0

        self.q_nns = [qn_keras.QNetwork(0, 0, 0, 0, 0, hack=True)]
        self.q_nns[0].load(agent_path, agent_name + "_target")

    def start_state(self, s0):
        self.s = self.s_formatter.convert(self.seq_num, s0)

    def next_action(self):
        qs = np.full((len(self.q_nns), self.n_actions), np.nan)
        for i, q_nn in enumerate(self.q_nns):
            qs[i] = q_nn.predict(self.s[None, :])
        qs = np.mean(qs, axis=0)
        a = np.random.choice(np.flatnonzero(qs >= qs.max()))
        self.a = a
        return a

    def next_reading(self, sp, r, done, train=True):
        self.seq_num += 1
        sp = self.s_formatter.convert(self.seq_num, sp, done)
        self.s = sp
