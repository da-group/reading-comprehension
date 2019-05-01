############################
# Jiachi Zhang
# zhangjiachi1007@gmail.com
############################

import tensorflow as tf
import numpy as np
import math

def conv_layer(parent, kernel_size, output_channel, stride, name, bias = True, relu = False, reuse=True):
    '''
    conv1d:
    input.shape = [N, sentence_length, channel_number]
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        init_w = init_b = tf.contrib.layers.xavier_initializer(dtype = tf.float32)
        input_shape = parent.shape.as_list()
        conv_func = tf.nn.conv1d if len(input_shape)==3 else tf.nn.conv2d
        kernel_shape = [kernel_size, input_shape[-1], output_channel] if len(input_shape)==3 \
                        else [kernel_size, kernel_size, input_shape[-1], output_channel]
        # stride = [1, stride, 1] if len(input_shape)==3 else [1, stride, stride, 1]
        weights = tf.get_variable(name='weights', shape=kernel_shape, dtype='float32', initializer=init_w)
        conv = conv_func(parent, weights, stride, padding = 'SAME')
        if bias:
            bias = tf.get_variable(name = 'bias', shape = [output_channel], dtype = 'float32', initializer = init_b)
            conv_with_bias = tf.nn.bias_add(conv, bias)
        else:
            conv_with_bias = conv
        if relu:
            return tf.nn.relu(conv_with_bias)
        else:
            return conv_with_bias

def depthwise_conv_layer(parent, kernel_size, num_d, num_p, stride, name, bias=True, relu=False, reuse=True):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        N, H, W, C = parent.shape.as_list()
        kh = kernel_size if H!=1 else 1
        kw = kernel_size if W!=1 else 1
        initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
        depthwise_filter = tf.get_variable('depthwise_weights', [kh, kw, C, num_d], dtype=tf.float32, initializer=initializer)
        pointwise_filter = tf.get_variable('pointwise_weights', [1, 1, num_d*C, num_p], dtype=tf.float32, initializer=initializer)
        output = tf.nn.separable_conv2d(parent, depthwise_filter, pointwise_filter, strides=[1, stride, stride, 1], padding='SAME')
        if bias:
            b = tf.get_variable('bias', num_p, initializer=tf.zeros_initializer())
            output += b
        if relu:
            output = tf.nn.relu(output)
        return output

def depthwise_conv_block(parent, num_layers, kernel_size, stride, num_d, num_p, name, dropout=0.0, reuse=True):
    '''
    this block consist of several depthwise separable convolutional layers
    Since the depthwise separable convolutional layer does not handle 1 dimension data,
    we should use tf.expand_dims at first
    parent.shape = [N, L, C]
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        output = tf.expand_dims(parent, 2)
        for i in range(num_layers):
            bn = layer_norm(output, 'bn_'+np.str(i+1))
            relu = True if i<num_layers-1 else False
            conv = depthwise_conv_layer(bn, kernel_size, num_d, num_p, stride, 'dconv_'+np.str(i), relu=relu)
            output = layer_dropout(conv, output, dropout)
        output = tf.squeeze(output, 2)
        return output

def fc_layer(parent, output_channel, name, bias = True, relu = False, reuse=True):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        shape = parent.shape
        C = shape[-1]
        init = tf.contrib.layers.xavier_initializer_conv2d(dtype = tf.float32)
        weights = tf.get_variable(name = 'weights', shape = [C, output_channel], dtype = 'float32', initializer = init)
        fc = tf.matmul(parent, weights)
        if bias:
            bias = tf.get_variable(name = 'bias', shape = [output_channel], dtype = 'float32', initializer = init)
            fc_with_bias = tf.nn.bias_add(fc, bias)
        else:
            fc_with_bias = fc
        if relu:
            return tf.nn.relu(fc_with_bias)
        else:
            return fc_with_bias

def highway(parent, kernel_size, output_channel, num_layers=2, name="highway", dropout=0.0):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if parent.shape.as_list()[-1]!=output_channel:
            parent = conv_layer(parent, 1, output_channel, 1, 'projection')
        for i in range(num_layers):
            T = tf.sigmoid(conv_layer(parent, kernel_size, output_channel, 1, 'conv'+np.str(i), relu=False))
            H = conv_layer(parent, kernel_size, output_channel, 1, 'conv'+np.str(i), relu=True)
            H = tf.nn.dropout(H, 1.0 - dropout)
            parent = H * T + parent * (1.0 - T)
        return parent

def layer_norm_compute_python(x, epsilon, scale, bias):
    """Layer norm raw computation."""
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * scale + bias

def layer_norm(x, name, epsilon=1e-6, reuse=True):
    """Layer normalize the tensor x, averaging over the last dimension."""
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        C = x.shape.as_list()[-1]
        # regularizer = tf.contrib.layers.l2_regularizer(scale=3e-7)
        scale = tf.get_variable(
            "layer_norm_scale", [C], initializer=tf.ones_initializer())
        bias = tf.get_variable(
            "layer_norm_bias", [C], initializer=tf.zeros_initializer())
        result = layer_norm_compute_python(x, epsilon, scale, bias)
        return result

def layer_dropout(inputs, residual, dropout):
    '''
    randomly using dropout layer
    '''
    pred = tf.random_uniform([]) < dropout
    return tf.cond(pred, lambda: residual, lambda: tf.nn.dropout(inputs, 1.0 - dropout) + residual)

def mask_layer(parent, mask, mod):
    if mask==None:
        return parent
    mask = tf.cast(mask, tf.float32)
    if mod=='mul':
        return parent*mask
    elif mod=='add':
        return parent - (1-mask)*1e30

def multihead_attention(parent, num_head, size_per_head, name, mask=None, bias=True, reuse=True):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        N, L, C = parent.shape.as_list()
        q = conv_layer(parent, 1, num_head*size_per_head, 1, 'q')
        q = tf.reshape(q, (-1, L, num_head, size_per_head))
        q = tf.transpose(q, [0, 2, 1, 3])
        v = conv_layer(parent, 1, num_head*size_per_head, 1, 'v')
        v = tf.reshape(v, (-1, L, num_head, size_per_head))
        v = tf.transpose(v, [0, 2, 1, 3])
        k = conv_layer(parent, 1, num_head*size_per_head, 1, 'k')
        k = tf.reshape(k, (-1, L, num_head, size_per_head))
        k = tf.transpose(k, [0, 2, 1, 3])
        score = tf.matmul(q, k, transpose_b=True)/tf.sqrt((float(size_per_head)))
        if bias:
            b = tf.get_variable("bias", [L], initializer=tf.zeros_initializer())
            score += b
        mask = tf.reshape(mask, [mask.shape[0], 1, 1, mask.shape[-1]])
        score = mask_layer(score, mask, 'add')
        score = tf.nn.softmax(score)
        output = tf.matmul(score, v)
        output = tf.transpose(output, [0, 2, 1, 3])
        output = tf.reshape(output, [-1, L, num_head*size_per_head])
        output = conv_layer(output, 1, C, 1, 'reshape')
        return output

def self_attention_layer(parent, kernel_size, output_channel, stride, num_head, size_per_head, name, mask=None, dropout=0.0, reuse=True):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        bn_1 = layer_norm(parent, 'bn_1')
        attention = multihead_attention(bn_1, num_head, size_per_head, 'multi_attention', mask=mask, reuse=tf.AUTO_REUSE)
        residual = layer_dropout(attention, bn_1, dropout)

        bn_2 = layer_norm(residual, 'bn_2', reuse=tf.AUTO_REUSE)
        conv_1 = conv_layer(bn_2, kernel_size, output_channel, stride, 'conv_1', reuse=tf.AUTO_REUSE)
        conv_2 = conv_layer(conv_1, kernel_size, output_channel, stride, 'conv_2', reuse=tf.AUTO_REUSE)
        output = layer_dropout(conv_2, parent, dropout)
        return output

def add_timing_signal(parent, min_timescale=1.0, max_timescale=1.0e4):
    N, L, C = parent.shape.as_list()
    return parent+get_timing_signal(L, C, min_timescale, max_timescale)

def get_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = tf.to_float(tf.range(length))
    num_timescales = channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return signal

def encoder_block(parent, num_blocks, num_conv_layers,
                  kernel_size, stride, num_d, num_p, output_channel,
                  num_head, size_per_head,
                  name,
                  mask=None, dropout=0.0,
                  reuse=True):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        output = parent
        C = parent.shape.as_list()[-1]
        if (C!=output_channel):
            output = conv_layer(output, 1, output_channel, 1, 'project')
        for i in range(num_blocks):
            output = add_timing_signal(output)
            output = depthwise_conv_block(output, num_conv_layers, kernel_size, stride, num_d, num_p, 'conv_'+np.str(i))
            output = self_attention_layer(output, kernel_size, output_channel, stride, num_head, size_per_head, 'attention_'+np.str(i),
                                          mask=mask, dropout=dropout)
        return output



if __name__ == '__main__':
    t = tf.constant(np.random.rand(10, 20, 50), dtype=tf.float32)
    p = tf.placeholder(tf.float32, [None, 20, 50], 'p')
    output = encoder_block(p,
                           num_blocks=2,
                           num_conv_layers=3,
                           kernel_size=7,
                           stride=1,
                           num_d=1,
                           num_p=128,
                           output_channel=128,
                           num_head=8,
                           size_per_head=32,
                           dropout=0.1,
                           name='test')

