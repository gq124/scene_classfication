import os

import numpy as np
import tensorflow as tf


# images_path = "data/MNIST_train_images.gz"
# labels_path = "data/MNIST_train_labels.gz"
export_path = "saved_model/"
ckpt_path = "tmp/ckpt"




class InceptionModel():
    '''
    A base class for an Inception v4 model
    '''

    def __init__(self,input, input_shape, output_shape, keep_prob,ckpt_path='train_model/model', model_path='saved_model/model'):
        '''
        Defines the input and output variables and stores the save locations of the model
        Keyword arguments:
        input_shape -- The shape of the input tensor, indexed by [sample, row, col, ch]
        output_shape -- The shape of the output tensor, indexed by [sample, class]
        ckpt_path -- The save path for checkpoints during training
        model_path -- The save path for the forward propagation subgraph. The generated model
                      can be used to classify images without allocating space for the gradients
        '''

        self.ckpt_path = ckpt_path
        self.model_path = model_path
        self.num_class = output_shape[1]
        self.X = input
        # self.y = tf.placeholder(tf.float32, output_shape)
        self.keep_prob = keep_prob

        # self.define_model()

        return

    def define_model(self):
        '''
        Defines the Inception v4 model.
        '''

        # self.keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope('input'):
            _X = tf.image.resize_images(self.X, [299, 299])

        with tf.variable_scope('stem'):
            _stem = stem(_X)

        _inception_a = {-1: _stem}
        for i in range(4):
            with tf.variable_scope('inception_a_' + str(i)):
                _inception_a[i] = inception_a(_inception_a[i - 1])

        with tf.variable_scope('reduction_a'):
            _reduction_a = reduction_a(_inception_a[3])

        _inception_b = {-1: _reduction_a}
        for i in range(7):
            with tf.variable_scope('inception_b_' + str(i)):
                _inception_b[i] = inception_b(_inception_b[i - 1])

        with tf.variable_scope('reduction_b'):
            _reduction_b = reduction_b(_inception_b[6])

        _inception_c = {-1: _reduction_b}
        for i in range(3):
            with tf.variable_scope('inception_c_' + str(i)):
                _inception_c[i] = inception_c(_inception_c[i - 1])

        pool = tf.nn.avg_pool(_inception_c[2], [1, 8, 8, 1], [1, 1, 1, 1], padding='VALID', name='pool')
        pool_f = tf.reshape(pool, [-1, 1536])
        pool_drop = tf.nn.dropout(pool_f, self.keep_prob)
        self.fc = dense(pool_drop, 'fc', self.num_class)

        # self.y_hat = tf.nn.softmax(self.fc, name='y_hat')
        #
        # # Creates a saver object exclusively for the forward propagation subgraph
        # model_variables = tf.get_collection_ref('tf.GraphKeys.MODEL_VARIABLES')
        # self.model_saver = tf.train.Saver(model_variables)

        return self.fc




def stem(tensor):
    '''
    Generates the graph for the stem subgraph of the Inception v4 model
    '''

    conv_1 = conv(tensor, 'conv_1', [3, 3, 3, 32], [1, 2, 2, 1], padding='VALID')

    conv_2 = conv(conv_1, 'conv_2', [3, 3, 32, 32], [1, 1, 1, 1], padding='VALID')

    conv_3 = conv(conv_2, 'conv_3', [3, 3, 32, 64], [1, 1, 1, 1])

    conv_4_1 = conv(conv_3, 'conv_4_1', [3, 3, 64, 64], [1, 2, 2, 1], padding='VALID')
    conv_4_2 = conv(conv_3, 'conv_4_2', [3, 3, 64, 96], [1, 2, 2, 1], padding='VALID')

    concat_1 = tf.concat([conv_4_1, conv_4_2], axis=3, name='concat_1')

    conv_5_1_1 = conv(concat_1, 'conv_5_1_1', [1, 1, 160, 64], [1, 1, 1, 1])
    conv_5_1_2 = conv(conv_5_1_1, 'conv_5_1_2', [3, 3, 64, 96], [1, 1, 1, 1], padding='VALID')

    conv_5_2_1 = conv(concat_1, 'conv_5_2_1', [1, 1, 160, 64], [1, 1, 1, 1])
    conv_5_2_2 = conv(conv_5_2_1, 'conv_5_2_2', [7, 1, 64, 64], [1, 1, 1, 1])
    conv_5_2_3 = conv(conv_5_2_2, 'conv_5_2_3', [1, 7, 64, 64], [1, 1, 1, 1])
    conv_5_2_4 = conv(conv_5_2_3, 'conv_5_2_4', [3, 3, 64, 96], [1, 1, 1, 1], padding='VALID')

    concat_2 = tf.concat([conv_5_1_2, conv_5_2_4], axis=3, name='concat_2')

    conv_6_1 = conv(concat_2, 'conv_6_1_1', [3, 3, 192, 192], [1, 2, 2, 1], padding='VALID')
    pool_6_2 = tf.nn.max_pool(concat_2, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID', name='pool_6_2')

    concat_3 = tf.concat([conv_6_1, pool_6_2], axis=3, name='concat_3')

    return concat_3


def inception_a(tensor):
    '''
    Generates the graph for the Inception A subgraph of the Inception v4 model
    '''

    pool_1_1 = tf.nn.avg_pool(tensor, [1, 3, 3, 1], [1, 1, 1, 1], padding='SAME', name='pool_1_1')
    conv_1_1 = conv(pool_1_1, 'conv_1_1', [1, 1, 384, 96], [1, 1, 1, 1])

    conv_2_1 = conv(tensor, 'conv_2_1', [1, 1, 384, 96], [1, 1, 1, 1])

    conv_3_1 = conv(tensor, 'conv_3_1', [1, 1, 384, 64], [1, 1, 1, 1])
    conv_3_2 = conv(conv_3_1, 'conv_3_2', [3, 3, 64, 96], [1, 1, 1, 1])

    conv_4_1 = conv(tensor, 'conv_4_1', [1, 1, 384, 64], [1, 1, 1, 1])
    conv_4_2 = conv(conv_4_1, 'conv_4_2', [3, 3, 64, 96], [1, 1, 1, 1])
    conv_4_3 = conv(conv_4_2, 'conv_4_3', [3, 3, 96, 96], [1, 1, 1, 1])

    concat = tf.concat([conv_1_1, conv_2_1, conv_3_2, conv_4_3], axis=3, name='concat')

    return concat


def reduction_a(tensor):
    '''
    Generates the graph for the Reduction A subgraph of the Inception v4 model
    '''

    pool_1_1 = tf.nn.max_pool(tensor, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID', name='pool_1_1')

    conv_2_1 = conv(tensor, 'conv_2_1', [3, 3, 384, 384], [1, 2, 2, 1], padding='VALID')

    conv_3_1 = conv(tensor, 'conv_3_1', [1, 1, 384, 192], [1, 1, 1, 1])
    conv_3_2 = conv(conv_3_1, 'conv_3_2', [3, 3, 192, 224], [1, 1, 1, 1])
    conv_3_3 = conv(conv_3_2, 'conv_3_3', [3, 3, 224, 256], [1, 2, 2, 1], padding='VALID')

    concat = tf.concat([pool_1_1, conv_2_1, conv_3_3], axis=3, name='concat')

    return concat


def inception_b(tensor):
    '''
    Generates the graph for the Inception B subgraph of the Inception v4 model
    '''

    pool_1_1 = tf.nn.avg_pool(tensor, [1, 3, 3, 1], [1, 1, 1, 1], padding='SAME', name='pool_1_1')
    conv_1_2 = conv(pool_1_1, 'conv_1_2', [1, 1, 1024, 128], [1, 1, 1, 1], )

    conv_2_1 = conv(tensor, 'conv_2_1', [1, 1, 1024, 384], [1, 1, 1, 1])

    conv_3_1 = conv(tensor, 'conv_3_1', [1, 1, 1024, 192], [1, 1, 1, 1])
    conv_3_2 = conv(conv_3_1, 'conv_3_2', [1, 7, 192, 224], [1, 1, 1, 1])
    conv_3_3 = conv(conv_3_2, 'conv_3_3', [1, 7, 224, 256], [1, 1, 1, 1])

    conv_4_1 = conv(tensor, 'conv_4_1', [1, 1, 1024, 192], [1, 1, 1, 1])
    conv_4_2 = conv(conv_4_1, 'conv_4_2', [1, 7, 192, 192], [1, 1, 1, 1])
    conv_4_3 = conv(conv_4_2, 'conv_4_3', [7, 1, 192, 224], [1, 1, 1, 1])
    conv_4_4 = conv(conv_4_3, 'conv_4_4', [1, 7, 224, 224], [1, 1, 1, 1])
    conv_4_5 = conv(conv_4_4, 'conv_4_5', [7, 1, 224, 256], [1, 1, 1, 1])

    concat = tf.concat([conv_1_2, conv_2_1, conv_3_3, conv_4_5], axis=3, name='concat')

    return concat


def reduction_b(tensor):
    '''
    Generates the graph for the Reduction B subgraph of the Inception v4 model
    '''

    pool_1_1 = tf.nn.max_pool(tensor, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID', name='pool_1_1')

    conv_2_1 = conv(tensor, 'conv_2_1', [1, 1, 1024, 192], [1, 1, 1, 1])
    conv_2_2 = conv(conv_2_1, 'conv_2_2', [3, 3, 192, 192], [1, 2, 2, 1], padding='VALID')

    conv_3_1 = conv(tensor, 'conv_3_1', [1, 1, 1024, 256], [1, 1, 1, 1])
    conv_3_2 = conv(conv_3_1, 'conv_3_2', [1, 7, 256, 256], [1, 1, 1, 1])
    conv_3_3 = conv(conv_3_2, 'conv_3_3', [7, 1, 256, 320], [1, 1, 1, 1])
    conv_3_4 = conv(conv_3_3, 'conv_3_4', [3, 3, 320, 320], [1, 2, 2, 1], padding='VALID')

    concat = tf.concat([pool_1_1, conv_2_2, conv_3_4], axis=3, name='concat')

    return concat


def inception_c(tensor):
    '''
    Generates the graph for the Inception C subgraph of the Inception v4 model
    '''

    pool_1_1 = tf.nn.avg_pool(tensor, [1, 3, 3, 1], [1, 1, 1, 1], padding='SAME', name='pool_1_1')
    conv_1_2 = conv(pool_1_1, 'conv_1_2', [1, 1, 1536, 256], [1, 1, 1, 1])

    conv_2_1 = conv(tensor, 'conv_2_1', [1, 1, 1536, 256], [1, 1, 1, 1])

    conv_3_1 = conv(tensor, 'conv_3_1', [1, 1, 1536, 384], [1, 1, 1, 1])
    conv_3_2_1 = conv(conv_3_1, 'conv_3_2_1', [1, 3, 384, 256], [1, 1, 1, 1])
    conv_3_2_2 = conv(conv_3_1, 'conv_3_2_2', [3, 1, 384, 256], [1, 1, 1, 1])

    conv_4_1 = conv(tensor, 'conv_4_1', [1, 1, 1536, 384], [1, 1, 1, 1])
    conv_4_2 = conv(conv_4_1, 'conv_4_2', [1, 3, 384, 448], [1, 1, 1, 1])
    conv_4_3 = conv(conv_4_2, 'conv_4_3', [3, 1, 448, 512], [1, 1, 1, 1])
    conv_4_3_1 = conv(conv_4_3, 'conv_4_3_1', [1, 3, 512, 256], [1, 1, 1, 1])
    conv_4_3_2 = conv(conv_4_3, 'conv_4_3_2', [3, 1, 512, 256], [1, 1, 1, 1])

    concat = tf.concat([conv_1_2, conv_2_1, conv_3_2_1, conv_3_2_2, conv_4_3_1, conv_4_3_2], axis=3, name='concat')

    return concat


def conv(tensor, name, shape, strides=[1, 1, 1, 1], padding='SAME', activation=tf.nn.relu):
    '''
    Generates a convolutional layer
    Keyword arguments:
    tensor -- input tensor. Must be indexed by [sample, row, col, ch]
    name -- the name that will be given to the tensorflow Variable in the GraphDef
    shape -- the shape of the kernel. Must be indexed by [row, col, num_input_ch, num_output_ch]
    strides -- the stride of the convolution. Must be indexed by [sample, row, col, ch]
    padding -- if set to 'SAME', the output will have the same height and width as the input. If
               set to 'VALID', the output will have its size reduced by the difference between the
               tensor size and kernel size
    activation -- the activation function to use
    '''

    W = tf.get_variable(name + "_W", shape)
    b = tf.get_variable(name + "_b", shape[-1])
    tf.add_to_collection('tf.GraphKeys.MODEL_VARIABLES', W)
    tf.add_to_collection('tf.GraphKeys.MODEL_VARIABLES', b)
    z = tf.nn.conv2d(tensor, W, strides=strides, padding=padding, name=name + '_z')
    h = tf.add(z, b, name=name + '_h')
    a = activation(h, name=name + '_a')

    return a


def dense(tensor, name, num_out):
    '''
    Generates a fully connected layer. Does not apply an activation function
    Keyword arguments:
    tensor -- input tensor. Must be indexed by [sample, ch]
    name -- the name that will be given to the tensorflow Variable in the GraphDef
    num_out -- the size of the output tensor
    '''

    W_fc = tf.get_variable('W_fc', [tensor.shape[1], num_out])
    b_fc = tf.get_variable('b_fc', [num_out])
    tf.add_to_collection('tf.GraphKeys.MODEL_VARIABLES', W_fc)
    tf.add_to_collection('tf.GraphKeys.MODEL_VARIABLES', b_fc)

    z_fc = tf.matmul(tensor, W_fc, name='z_fc')
    h_fc = tf.add(z_fc, b_fc, name='h_fc')

    return h_fc


