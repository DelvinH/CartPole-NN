import tensorflow as tf
import numpy as np
import tensorflow.compat.v1 as v1

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

def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random.normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs,Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data - 0.5) + noise

xs = v1.placeholder(tf.float32, [None,1])
ys = v1.placeholder(tf.float32, [None,1])

l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
#l2 = add_layer(l1,10,10,activation_function=tf.nn.relu)
#l3 = add_layer(l2,10,10,activation_function=tf.nn.relu)
prediction = add_layer(l1,10,1,activation_function=None)

loss = tf.reduce_mean(v1.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
train_step = v1.train.GradientDescentOptimizer(0.1).minimize(loss)

init = v1.global_variables_initializer()
sess = v1.Session()
sess.run(init)

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i % 50 == 0:
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
