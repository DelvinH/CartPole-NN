import gym
import random
import numpy as np
from QNetwork import QNetwork
from ReplayBuffer import ReplayBuffer
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class DQNAgent():
    def __init__(self, env):
        self.state_dim = env.observation_space.shape
        self.action_size = env.action_space.n
        self.q_network = QNetwork(self.state_dim, self.action_size)
        self.gamma = 0.97
        self.ep = 1.0
        self.replay_buffer = ReplayBuffer(length=10000)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def get_action(self, state):
        q_state = self.q_network.get_q_state(self.sess, [state])
        if random.random() < self.ep:
            action = np.random.randint(self.action_size)
        else:
            action = np.argmax(q_state)
        return action

    def train(self, state, action, next_state, reward, done):
        self.replay_buffer.add((state, action, next_state, reward, done))
        states, actions, next_states, rewards, dones = self.replay_buffer.sample(50)
        q_next_states = self.q_network.get_q_state(self.sess, next_states)
        q_next_states[dones] = np.zeros([self.action_size])  # sets q_next_state to 0 if done
        q_targets = rewards + self.gamma * np.max(q_next_states, axis=1)

        self.q_network.update_model(self.sess, states, actions, q_targets)

        if done: self.ep = max(0.1, 0.99 * self.ep)

    def __del__(self):
        self.sess.close()