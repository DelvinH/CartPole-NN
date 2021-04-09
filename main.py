
from PIL import Image
import numpy as np
import gym
import tensorflow as tf
import tensorflow.compat.v1 as v1
import tf_slim as slim
import PIL
from PIL import Image
import random

from matplotlib import pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
v1.disable_v2_behavior()


env = gym.make('CartPole-v0')
env.reset()











class experience_buffer():
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        #print(experience)
        self.buffer.extend(experience)
        #print(self.buffer)

    def sample(self,size):
        #print(self.buffer)
        #print(len(self.buffer))
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])

    def print(self):
        print(self.buffer)

    def get_size(self):
        return len(self.buffer)

    def get_buffer(self):
        return np.reshape(np.array(self.buffer),[len(self.buffer),5])



class Qnetwork():
    def __init__(self,h_size):
        # The network receives a frame from the game, flattened into an array.
        # It then resizes it and processes it through four convolutional layers.
        self.scalarInput = v1.placeholder(shape=[None,84672],dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput,shape=[-1,168,168,3]) # RESHAPE LINE
        self.conv1 = slim.conv2d(inputs=self.imageIn, num_outputs=32, kernel_size=[12, 12], stride=[6, 6],
                                 padding='VALID', biases_initializer=None)
        self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=64, kernel_size=[5, 5], stride=[2, 2],
                                 padding='VALID', biases_initializer=None)
        self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs=64, kernel_size=[6, 6], stride=[1, 1],
                                 padding='VALID', biases_initializer=None)
        self.conv4 = slim.conv2d(inputs=self.conv3, num_outputs=h_size, kernel_size=[7, 7], stride=[1, 1],
                                 padding='VALID', biases_initializer=None)
        # We take the output from the final convolutional layer and split it into separate advantage and value streams.
        self.streamAC,self.streamVC = tf.split(self.conv4,2, axis=3)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.initializers.GlorotUniform() # xavier_init = tf.contrib.layers.xavier_initializer() ---no contrib lib in tf 2.0
        self.AW = tf.Variable(xavier_init([h_size//2,2])) # WHY IS THIS 29 (env.actions)
        self.VW = tf.Variable(xavier_init([h_size//2,1]))
        self.Advantage = tf.matmul(self.streamA,self.AW)
        self.Value = tf.matmul(self.streamV,self.VW)

        # Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keepdims=True))
        self.predict = tf.argmax(self.Qout,1)
        self.extract_value, self.extract_index = tf.nn.top_k(self.Qout, 1, sorted=True)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = v1.placeholder(shape=[None],dtype=tf.float32)
        self.actions = v1.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,2,dtype=tf.float32) # WHY IS THIS 29

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = v1.train.AdamOptimizer(learning_rate=0.001)
        self.updateModel = self.trainer.minimize(self.loss)


def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder


def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

def processScreens(screens, dim):
    for i in range(len(screens)):
        b = np.array(Image.fromarray(screens[i][:, :, 0]).resize((dim, dim),
                                                             PIL.Image.NEAREST))  # b = scipy.misc.imresize(screen[:,:,0], [84, 84, 1], interp = 'nearest') #---imresize is depreciated
        c = np.array(Image.fromarray(screens[i][:, :, 1]).resize((dim, dim),
                                                             PIL.Image.NEAREST))  # c = scipy.misc.imresize(screen[:, :, 1], [84, 84, 1], interp = 'nearest')
        d = np.array(Image.fromarray(screens[i][:, :, 2]).resize((dim, dim),
                                                             PIL.Image.NEAREST))  # d = scipy.misc.imresize(screen[:, :, 2], [84, 84, 1], interp = 'nearest')
        screens[i] = np.reshape(np.stack([b, c, d], axis=2),[dim*dim*3])
        #plt.imshow(np.stack([b, c, d], axis=2), interpolation='nearest')
        #plt.show()
    return screens


num_episodes = 10000
max_ep_length = 1000
y = .99
h_size = 512 # what number?
ep_max = 1
ep_min = 0.1
annealing_steps = 10000
dom_period = 10000
belief = 0.5
total_steps = 0
pre_training_steps = 1000
update_freq = 8
batch_size = 32
tau = 0.001


mainQN = Qnetwork(h_size)
targetQN = Qnetwork(h_size)
init = v1.global_variables_initializer()

trainables = v1.trainable_variables()
targetOps = updateTargetGraph(trainables, tau)

episode_buffer_size = 50000
buffer = experience_buffer(episode_buffer_size)

ep_decrease = (ep_max - ep_min) / annealing_steps
ep = ep_max
belief_decrease = belief / dom_period

screen = env.render(mode='rgb_array') # 400 by 600 by 3 rgb array
[s_prev,s_prev2,s_prev3] = processScreens([screen,screen,screen],168)

with v1.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        episode_buffer = experience_buffer(episode_buffer_size) # an individual buffer for each episode
        state = env.reset() # reset at start of episode
        valid_actions = list(range(env.action_space.n)) # probably not an ideal way to do this but will work here since all actions are always valid | 0 = left, 1 = right
        total_reward = 0

        j = 0
        while j < max_ep_length: # if pole is still up, assume it can be kept up indefinitely
            j += 1
            screen = env.render(mode='rgb_array') # 400 by 600 by 3 rgb array
            [s] = processScreens([screen],168)
            np.add(np.add(np.add(np.asarray(s),np.asarray(s_prev)),np.asarray(s_prev2)),np.asarray(s_prev3))
            #plt.imshow(np.stack([b, c, d], axis=2), interpolation='nearest')
            #plt.show()
            # e-greedy policy evaluation
            if np.random.rand(1) < ep: # exploration based on e-greedy policy
                #print("exploring")
                action = env.action_space.sample()
            else:
                action_values = sess.run(mainQN.Qout, feed_dict={mainQN.scalarInput:[s]})[0] # obtain action value function from main Q network | action follows policy
                #print(action_values)
                # pick action with highest Q-value
                action = valid_actions[0]
                for _ in valid_actions:
                    if action_values[_] > action_values[action]:
                        action = _
                        #print("switching action")
                #print("on-policy: " + str(action))
            #print(ep)
            #print(action)

            state_prime, reward, done, info = env.step(action) # increment the environment by one step
            #reward = reward-abs(state_prime[2])
            screen_prime = env.render(mode='rgb_array')
            [s_prime] = processScreens([screen_prime],168)
            total_steps += 1

            if ep > ep_min: # current epsilon | lowers from ep_max to ep_min over annealing_steps steps
                ep -= ep_decrease
            if belief > belief_decrease:  # belief in teacher network | lower until entirely independent
                belief -= belief_decrease
            else:
                belief = 0

            #episode_buffer.add(np.reshape(np.array([i, j, 1, 1]), [1, 4]))
            episode_buffer.add(np.reshape(np.array([s, action, reward, s_prime, done]), [1, 5]))
            #print(np.reshape(np.array([s, action, reward, state_prime]), [1, 4]))


            # training with DQN update
            if total_steps > pre_training_steps:
                if total_steps % update_freq == 0:
                    #print("Updating network")
                    #test = buffer.get_buffer()[:,2]
                    #print(test)
                    trainBatch = buffer.sample(batch_size)

                    Q1 = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 3])})
                    Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.scalarInput: np.vstack(trainBatch[:, 3])})
                    actions_values_list = sess.run(mainQN.Qout,
                                                   feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 3])})

                    # \/ new
                    q1 = np.zeros((batch_size,), dtype=np.int)
                    aaa = valid_actions[0]
                    for batch_num in range(batch_size):
                        max_action_index = 0
                        for test in range(len(valid_actions) - 1):
                            if actions_values_list[batch_num][valid_actions[test + 1]] > \
                                    actions_values_list[batch_num][max_action_index]:
                                max_action_index = test + 1
                        q1[batch_num] = int(max_action_index)

                    doubleQ = Q2[range(batch_size), q1]

                    targetQ = trainBatch[:, 2] + (y * doubleQ) * (1 - trainBatch[:, 4])
                    # Update the network with our target values.
                    _ = sess.run(mainQN.updateModel, feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 0]),
                                                                mainQN.targetQ: targetQ,
                                                                mainQN.actions: trainBatch[:, 1]})
                    updateTarget(targetOps, sess)  # Update the target network toward the primary network.

            #if done:
                #reward -= 100
            total_reward += reward
            s_prev3 = s_prev2
            s_prev2 = s_prev
            s_prev = s
            s = s_prime

            if done: # either the cart has moved off the edge of the screen or the pole has tilted past 12 degrees | end episode
                break

        print('Episode: ', i, ', Total reward: ', total_reward,", Ep: ", ep, ", Belief: ", belief)
        buffer.add(episode_buffer.buffer)