import csv
from types import new_class
import numpy as np
from numpy.core.fromnumeric import shape
import tensorflow as tf
from tensorflow.python.ops.string_ops import _reduce_join_reduction_dims

def loadTrainData():
    l = []
    with open('./digit_recognizer/train.csv') as f:
        lines = csv.reader(f)
        for line in lines:
            l.append(line)

    l.remove(l[0])
    l = np.array(l)
    label = np.zeros([np.shape(l)[0], 10])
    for i in range(np.shape(l)[0]):
        label[i, int(l[i, 0])] = 1
    data = l[:,1:]
    return normalizing(toInt(data)), toInt(label)

def toInt(array):
    array = np.mat(array)
    m, n = np.shape(array)
    newArray = np.zeros((m, n))
    for i in range(m): 
        for j in range(n):
            newArray[i, j] = int(array[i, j])
    return newArray

def normalizing(array):
    m, n = np.shape(array)
    for i in range(m):
        for j in range(n):
            if array[i, j] != 0:
                array[i, j] = 1
    return array

def loadTestData():
    l = []
    with open('./digit_recognizer/test.csv') as f:
        lines = csv.reader(f)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    data = np.array(l)
    return normalizing(toInt(data))

def conv(feature, W, s=1):
    return tf.nn.conv2d(feature, W, strides=[1, s, s, 1], padding='SAME')

def leakey_relu(x, alpha):
    return tf.maximum(x, alpha * x)

def weight_variale(shape, name):
    init = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(init, name=name)

def bias_variable(shape, name):
    init = tf.constant(0.01, shape=shape)
    return tf.Variable(init, name=name)

def classify():
    width = 28
    height = 28
    channel = 1
    out_class = 10
    batch_size = 10
    

    digit_ = tf.placeholder(tf.float32, shape=[batch_size, width * height * channel])
    digit = tf.reshape(digit_, shape=[batch_size, width, height, channel])
    y_ = tf.placeholder(tf.float32, shape=[batch_size, out_class])
    
    w1 = weight_variale([3, 3, 1, 32], name='w1')
    b1 = bias_variable([32], name='b1')
    c1 = leakey_relu(conv(digit, w1) + b1, 0.2)

    p1 = tf.nn.max_pool(c1, ksize=[1, 2, 2 ,1], strides=[1, 2, 2, 1], padding='SAME')

    w2 = weight_variale([3, 3, 32, 64], name='w2')
    b2 = bias_variable([64], name='b2')
    c2 = leakey_relu(conv(p1, w2) + b2, 0.2)
    
    p2 = tf.nn.max_pool(c2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    flatten = tf.reshape(p2, [-1, 7 * 7 * 64])
    

    w_fc1 = weight_variale([3136, 1024], name='fc_w1')
    b_fc1 = bias_variable([1024], name='fc_b1')
    fc1 = tf.matmul(flatten, w_fc1) + b_fc1
    fc1 = tf.nn.relu(fc1)
 

    w_fc2 = weight_variale([1024, 10], name='fc_w2')
    b_fc2 = bias_variable([10], name='fc_b2')
    fc2 = tf.matmul(fc1, w_fc2) + b_fc2
    fc2 = tf.nn.sigmoid(fc2)

    # cross_entropy = tf.reduce_mean(-tf.reduce_mean(y_ * tf.log(fc2), reduction_indices=[1]))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=fc2)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(fc2), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    epochs = 100

    train_data, label = loadTrainData()
    train_size = np.shape(train_data)[0]
    max_batch = train_size // batch_size
    # test_data = loadTestData()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            for idx in range(max_batch):
                print("step: %d, idx: %d", i, idx)
                batch_data = train_data[idx * batch_size : (idx+1) * batch_size, :]
                batch_label = label[idx * batch_size : (idx+1) * batch_size, :]
                _ = sess.run(train_step, feed_dict={digit: batch_data, y_: batch_label})


    
    

classify()
