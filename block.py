############################
# Jiachi Zhang
# zhangjiachi1007@gmail.com
############################

import tensorflow as tf



def conv1d_layer(parent, kernal_size, stride, name, if_bias = True, if_relu = False):
    '''
    input.shape = [N, sentence_length, channel_number]
    kernel_size.shape = [kernel_length, in_channel, out_channel]
    '''
    with tf.variable_scope(name):
        init_w = init_b = tf.contrib.layers.xavier_initializer(dtype = tf.float32)

        weights = tf.get_variable(name = 'weights', shape = kernal_size, dtype = 'float32', initializer = init_w)
        conv = tf.nn.conv1d(parent, weights, stride, padding = 'SAME')

        if if_bias:
            bias = tf.get_variable(name = 'bias', shape = [kernal_size[-1]], dtype = 'float32', initializer = init_b)
            conv_with_bias = tf.nn.bias_add(conv, bias)
        else:
            conv_with_bias = conv

        if if_relu:
            return tf.nn.relu(conv_with_bias)
        else:
            return conv_with_bias
