import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class QNetwork():
    def __init__(self, state_dim, action_size):
        self.state_in = tf.placeholder(tf.float32, shape=[None, *state_dim])
        self.action_in = tf.placeholder(tf.int32, shape=[None])
        self.q_target_in = tf.placeholder(tf.float32, shape=[None])

        action_one_hot = tf.one_hot(self.action_in, depth=action_size)

        self.hidden1 = tf.layers.dense(self.state_in, 100, activation=tf.nn.relu)
        self.q_state = tf.layers.dense(self.hidden1, action_size, activation=None)
        self.q_state_action = tf.reduce_sum(tf.multiply(self.q_state, action_one_hot), axis=1)

        self.loss = tf.reduce_sum(tf.square(self.q_state_action - self.q_target_in))
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.optimize = self.trainer.minimize(self.loss)

    def update_model(self, session, state, action, q_target):
        session.run(self.optimize, feed_dict={self.state_in: state, self.action_in: action, self.q_target_in: q_target})

    def get_q_state(self, session, state):
        q_state = session.run(self.q_state, feed_dict={self.state_in: state})
        return q_state
