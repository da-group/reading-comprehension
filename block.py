############################
# Jiachi Zhang
# zhangjiachi1007@gmail.com
############################

import tensorflow as tf



def conv1d_layer(parent, kernal_size, stride, name, bias = True, relu = False):
    '''
    input.shape = [N, sentence_length, channel_number]
    kernel_size.shape = [kernel_length, in_channel, out_channel]
    '''
    with tf.variable_scope(name):
        init_w = init_b = tf.contrib.layers.xavier_initializer(dtype = tf.float32)

        weights = tf.get_variable(name = 'weights', shape = kernal_size, dtype = 'float32', initializer = init_w)
        conv = tf.nn.conv1d(parent, weights, stride, padding = 'SAME')

        if bias:
            bias = tf.get_variable(name = 'bias', shape = [kernal_size[-1]], dtype = 'float32', initializer = init_b)
            conv_with_bias = tf.nn.bias_add(conv, bias)
        else:
            conv_with_bias = conv

        if relu:
            return tf.nn.relu(conv_with_bias)
        else:
            return conv_with_bias

def layer_norm_compute_python(x, epsilon, scale, bias):
    """Layer norm raw computation."""
    mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * scale + bias

def layer_norm(x, filters=None, epsilon=1e-6, scope=None, reuse=None):
    """Layer normalize the tensor x, averaging over the last dimension."""
    if filters is None:
        filters = x.get_shape()[-1]
    with tf.variable_scope(scope, default_name="layer_norm", values=[x], reuse=reuse):
        regularizer = tf.contrib.layers.l2_regularizer(scale=3e-7)
        scale = tf.get_variable(
            "layer_norm_scale", [filters], regularizer = regularizer, initializer=tf.ones_initializer())
        bias = tf.get_variable(
            "layer_norm_bias", [filters], regularizer = regularizer, initializer=tf.zeros_initializer())
        result = layer_norm_compute_python(x, epsilon, scale, bias)
        return result

def depthwise_conv_layer(parent, kernel_size, num_d, num_p, stride, name, bias=True, relu=False):
    with tf.variable_scope(name):
        N, H, W, C = parent.shape.as_list()
        initializer = tf.contrib.layers.xavier_initializer(dtupe=tf.float32)
        depthwise_filter = tf.get_variable('depthwise_weights', [kernel_size[0], kernel_size[1], C, num_d], dtype=tf.float32, initializer=initializer)
        pointwise_filter = tf.get_variable('pointwise_weights', [1, 1, num_d*C, num_p], dtype=tf.float32, intializer=initializer)
        output = tf.nn.separable_conv2d(parent, depthwise_filter, pointwise_filter, stride=stride, padding='SAME')
        if bias:
            b = tf.get_variable('bias', num_p, initializer=tf.zero_initializer())
            output += b
        if relu:
            output = tf.nn.relu(output)
        return output
