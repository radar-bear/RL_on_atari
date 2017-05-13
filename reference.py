import tensorflow as tf
import numpy as np


class QFuncModel():
    '''
    QFuncModel defines a reference for Q-network

        s->[ batch, width, height, frames ], default [ 32, 80, 80, 4]

        conv1 use kernel 8x8 with in-channel 4 and out-channel 32
        conv1 stride [ 1, 4, 4, 1 ], padding="SAME"
        conv1 shape [batch, 20, 20, 32]
        relu!
        pool1 use 2x2 max pooling
        pool1 shape [batch, 10, 10, 32]

        conv2 use kernel 4x4 with in-channel 32 and out-channel 64
        conv2 stride [ 1, 2, 2, 1 ], padding="SAME"
        conv2 shape [batch, 5, 5, 64]
        relu!
        no pooling in this layer

        conv3 use kernel 3x3 with in-channel 64 and out-channel 64
        conv3 stride [ 1, 1, 1, 1 ], padding = "SAME"
        conv3 shape [batch, 5, 5, 64]
        relu!
        no pooling in this layer

        conv3_flat reshape the conv3 into a list of vector
        conv3_flat shape [batch, 1600]

        fc_1 fully connect with relu active function
        fc_1 shape [batch, 512]

        readout fully connect *without* active function
        readout shape [batch, action_num]

        summary:
        input       [batch, 80, 80, 4]
        conv1       [batch, 20, 20, 32]
        pool1       [batch, 10, 10, 32]
        conv2       [batch, 5,  5,  64]
        conv3       [batch, 5,  5,  64]
        conv3_flat  [batch, 1600]
        fc_1        [batch, 512]
        readout     [batch, 3]

        a is a one hot vector, where 1 marked the action
        tf.reduce_sum(tf.multiply(a, readout)) returns the Q-value-predict
        readout_action shape [batch, Q-value-predict]
        cost should calculate the mean divergence between predict and target
        which is defined as tf.reduce_mean(tf.square(y - readout_action))

        self.train_op use Adam to train this network

    Properties

        copy(sess, model2)
        Use this function to keep the original weights of Q-network.
        This function would copy all variables into model2, which can be used to calculate the Q-value-target
        Inputs:
            model2 should be an instance of class QFuncModel
            sess should be provided to run the tf.assign op


    '''
    def __init__(self, args):
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.01)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.01, shape=shape)
            return tf.Variable(initial)

        def conv2d(x, W, stride):
            return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        # input layer
        self.s = tf.placeholder("float", [None, args.resize_width, args.resize_height, args.look_forward_step])
        self.a = tf.placeholder("float", [None, args.actions])
        self.y = tf.placeholder("float", [None])

        self.W_conv1 = weight_variable([8, 8, 4, 32])
        self.b_conv1 = bias_variable([32])

        self.W_conv2 = weight_variable([4, 4, 32, 64])
        self.b_conv2 = bias_variable([64])

        self.W_conv3 = weight_variable([3, 3, 64, 64])
        self.b_conv3 = bias_variable([64])

        self.W_fc1 = weight_variable([1600, 512])
        self.b_fc1 = bias_variable([512])

        self.W_fc2 = weight_variable([512, args.actions])
        self.b_fc2 = bias_variable([args.actions])

        # hidden layers
        h_conv1 = tf.nn.relu(conv2d(self.s, self.W_conv1, 4) + self.b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(conv2d(h_pool1, self.W_conv2, 2) + self.b_conv2)
        # h_pool2 = max_pool_2x2(h_conv2)

        h_conv3 = tf.nn.relu(conv2d(h_conv2, self.W_conv3, 1) + self.b_conv3)
        # h_pool3 = max_pool_2x2(h_conv3)

        # h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
        h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

        self.h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, self.W_fc1) + self.b_fc1)

        # readout layer
        self.readout = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2

        # softmax readout
        self.softmax = tf.nn.softmax(self.readout)

        # define the cost function
        readout_action = tf.reduce_sum(tf.multiply(self.readout, self.a), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(self.y - readout_action))
        self.train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(cost)

    def variable_list(self):
        return [self.W_conv1, self.W_conv2, self.W_conv3, self.W_fc1, self.W_fc2,
                self.b_conv1, self.b_conv2, self.b_conv3, self.b_fc1, self.b_fc2 ]

    def copy(self, sess, model2):
        '''
        copy model2's varible to self
        Be careful, this function copy FROM model2 TO self!
        '''
        l1 = self.variable_list()
        l2 = model2.variable_list()
        assign_op = [tf.assign(l1[i], l2[i]) for i in range(len(l1))]
        sess.run(assign_op)

    def get_softmax(self, sess, s):
        softmax_readout = sess.run(self.softmax, feed_dict = {self.s:[s]})
        return softmax_readout[0]

    def play(self, sess, s):
        Q_value = sess.run(self.readout, feed_dict = {self.s:[s]})
        # np.argmax will flat the high dim array into a list
        # and return the index of the max element
        # Q_value has shape [1, 3] here
        action_index = np.argmax(Q_value)
        return action_index

    def maxQ(self, sess, s):
        Q_value = sess.run(self.readout, feed_dict = {self.s:[s]})
        return np.max(Q_value)