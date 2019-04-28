# This script is modified from https://github.com/NLPLearn/QANet/blob/master/model.py and https://github.com/NLPLearn/QANet/blob/master/layers.py

import tensorflow as tf
from tensorflow.python.ops import nn_ops
from tensorflow.keras import backend as K
import numpy as np
from block import conv_layer
# tf.enable_eager_execution()

regularizer = tf.contrib.layers.l2_regularizer(scale=3e-7)

def initializer():
    '''
    Copy from https://github.com/NLPLearn/QANet/blob/master/layers.py
    '''
    return tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True, dtype=tf.float32)

def mask_logits(inputs, mask, mask_value= -1e30):
    '''
    Copy from https://github.com/NLPLearn/QANet/blob/master/layers.py
    '''
    shapes = inputs.shape.as_list()
    mask = tf.cast(mask, tf.float32)
    return inputs + mask_value * (1 - mask)

def dot(x, y):
    """Modified from keras==2.0.6
    Multiplies 2 tensors (and/or variables) and returns a *tensor*.
    When attempting to multiply a nD tensor
    with a nD tensor, it reproduces the Theano behavior.
    (e.g. `(2, 3) * (4, 3, 5) -> (2, 4, 5)`)
    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.
    # Returns
        A tensor, dot product of `x` and `y`.
    """
    if K.ndim(x) is not None and (K.ndim(x) > 2 or K.ndim(y) > 2):
        x_shape = []
        for i, s in zip(x.get_shape().as_list(), tf.unstack(tf.shape(x))):
            if i is not None:
                x_shape.append(i)
            else:
                x_shape.append(s)
        x_shape = tuple(x_shape)
        y_shape = []
        for i, s in zip(y.get_shape().as_list(), tf.unstack(tf.shape(y))):
            if i is not None:
                y_shape.append(i)
            else:
                y_shape.append(s)
        y_shape = tuple(y_shape)
        y_permute_dim = list(range(K.ndim(y)))
        y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
        xt = tf.reshape(x, [-1, x_shape[-1]])
        yt = tf.reshape(tf.transpose(y, perm=y_permute_dim), [y_shape[-2], -1])
        return tf.reshape(tf.matmul(xt, yt),
                          x_shape[:-1] + y_shape[:-2] + y_shape[-1:])
    if isinstance(x, tf.SparseTensor):
        out = tf.sparse_tensor_dense_matmul(x, y)
    else:
        out = tf.matmul(x, y)
    return out

def batch_dot(x, y, axes=None):
    """Copy from keras==2.0.6
    Batchwise dot product.
    `batch_dot` is used to compute dot product of `x` and `y` when
    `x` and `y` are data in batch, i.e. in a shape of
    `(batch_size, :)`.
    `batch_dot` results in a tensor or variable with less dimensions
    than the input. If the number of dimensions is reduced to 1,
    we use `expand_dims` to make sure that ndim is at least 2.
    # Arguments
        x: Keras tensor or variable with `ndim >= 2`.
        y: Keras tensor or variable with `ndim >= 2`.
        axes: list of (or single) int with target dimensions.
            The lengths of `axes[0]` and `axes[1]` should be the same.
    # Returns
        A tensor with shape equal to the concatenation of `x`'s shape
        (less the dimension that was summed over) and `y`'s shape
        (less the batch dimension and the dimension that was summed over).
        If the final rank is 1, we reshape it to `(batch_size, 1)`.
    """
    if isinstance(axes, int):
        axes = (axes, axes)
    x_ndim = K.ndim(x)
    y_ndim = K.ndim(y)
    if x_ndim > y_ndim:
        diff = x_ndim - y_ndim
        y = tf.reshape(y, tf.concat([tf.shape(y), [1] * (diff)], axis=0))
    elif y_ndim > x_ndim:
        diff = y_ndim - x_ndim
        x = tf.reshape(x, tf.concat([tf.shape(x), [1] * (diff)], axis=0))
    else:
        diff = 0
    if K.ndim(x) == 2 and K.ndim(y) == 2:
        if axes[0] == axes[1]:
            out = tf.reduce_sum(tf.multiply(x, y), axes[0])
        else:
            out = tf.reduce_sum(tf.multiply(
                tf.transpose(x, [1, 0]), y), axes[1])
    else:
        if axes is not None:
            adj_x = None if axes[0] == K.ndim(x) - 1 else True
            adj_y = True if axes[1] == K.ndim(y) - 1 else None
        else:
            adj_x = None
            adj_y = None
        out = tf.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)
    if diff:
        if x_ndim > y_ndim:
            idx = x_ndim + y_ndim - 3
        else:
            idx = x_ndim - 1
        out = tf.squeeze(out, list(range(idx, idx + diff)))
    if K.ndim(out) == 1:
        out = tf.expand_dims(out, 1)
    return out

def optimized_trilinear_for_attention(args, c_maxlen, q_maxlen, input_keep_prob=1.0,
                                      scope='efficient_trilinear',
                                      bias_initializer=tf.zeros_initializer(),
                                      kernel_initializer=initializer()):
    '''
    Copy from https://github.com/NLPLearn/QANet/blob/master/layers.py
    '''
    assert len(args) == 2, "just use for computing attention with two input"
    arg0_shape = args[0].get_shape().as_list()
    arg1_shape = args[1].get_shape().as_list()
    if len(arg0_shape) != 3 or len(arg1_shape) != 3:
        raise ValueError("`args` must be 3 dims (batch_size, len, dimension)")
    if arg0_shape[2] != arg1_shape[2]:
        raise ValueError("the last dimension of `args` must equal")
    arg_size = arg0_shape[2]
    dtype = args[0].dtype
    droped_args = [tf.nn.dropout(arg, input_keep_prob) for arg in args]
    with tf.variable_scope(scope):
        weights4arg0 = tf.get_variable(
            "linear_kernel4arg0", [arg_size, 1],
            dtype=dtype,
            regularizer=regularizer,
            initializer=kernel_initializer)
        weights4arg1 = tf.get_variable(
            "linear_kernel4arg1", [arg_size, 1],
            dtype=dtype,
            regularizer=regularizer,
            initializer=kernel_initializer)
        weights4mlu = tf.get_variable(
            "linear_kernel4mul", [1, 1, arg_size],
            dtype=dtype,
            regularizer=regularizer,
            initializer=kernel_initializer)
        biases = tf.get_variable(
            "linear_bias", [1],
            dtype=dtype,
            regularizer=regularizer,
            initializer=bias_initializer)
        subres0 = tf.tile(dot(droped_args[0], weights4arg0), [1, 1, q_maxlen])
        subres1 = tf.tile(tf.transpose(
            dot(droped_args[1], weights4arg1), perm=(0, 2, 1)), [1, c_maxlen, 1])
        subres2 = batch_dot(
            droped_args[0] * weights4mlu, tf.transpose(droped_args[1], perm=(0, 2, 1)))
        res = subres0 + subres1 + subres2
        # nn_ops.bias_add(res, biases)
        return res


def CQ_attention_layer(c, q, N, c_maxlen, q_maxlen, scope, dropout=0.0):
    '''
    Modified from https://github.com/NLPLearn/QANet/blob/master/model.py
    :param c: context, shape = (batch_size, context_max_sentence_length, vector_length)  e.g.(32, 80, 50)
    :param q: question, shape = (batch_size, question_max_sentence_length, vector_length)  e.g.(32, 40, 50)
    :param N: int, batch_size
    :param c_maxlen: int, max_sentence_length of context
    :param q_maxlen: int, max_sentence_length of question
    :return: attention tensor, shape = (batch_size, context_max_sentence_length, 4*vector_length)   e.g.(32, 80, 200)
    '''
    with tf.variable_scope(scope):
        # c_mask = tf.cast(c, tf.bool)
        # c_mask = tf.slice(c_mask, [0, 0], [N, c_maxlen])
        # mask_c = tf.expand_dims(c_mask, 2)

        # q_mask = tf.cast(q, tf.bool)
        # q_mask = tf.slice(q_mask, [0, 0], [N, q_maxlen])
        # mask_q = tf.expand_dims(q_mask, 1)

        S = optimized_trilinear_for_attention([c, q], c_maxlen, q_maxlen, input_keep_prob=1.0 - dropout)

        # S_ = tf.nn.softmax(mask_logits(S, mask=mask_q))
        # S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask=mask_c), dim=1), (0, 2, 1))

        S_ = tf.nn.softmax(S)
        S_T = tf.transpose(tf.nn.softmax(S, dim=1), (0, 2, 1))

        c2q = tf.matmul(S_, q)
        q2c = tf.matmul(tf.matmul(S_, S_T), c)
        # attention_outputs = [c, c2q, c * c2q, c * q2c]
        return c2q, c * c2q, c * q2c




# Compute the attention between Context, Question and Answer
def CQA_attention(c, q, a, N, output_channel, c_maxlen, q_maxlen, a_maxlen, dropout=0.0):
    '''
    : param c: context, shape = (batch_size, context_max_sentence_length, vector_length)  e.g.(32, 80, 50)
    : param q: question, shape = (batch_size, question_max_sentence_length, vector_length)  e.g.(32, 40, 50)
    : param a: answer, shape = (batch_size, answer_max_sentence_length, vector_length)  e.g.(32, 20, 50)
    : param N: int, batch_size
    : param c_maxlen: int, max_sentence_length of context
    : param q_maxlen: int, max_sentence_length of question
    : param a_maxlen: int, max_sentence_length of answer
    : return: attention tensor, shape = (batch_size, context_max_sentence_length, 4*vector_length)   e.g.(32, 80, 200)
    '''
    # cq_atten_outputs = CQ_attention_layer(c, q, N, c_maxlen, q_maxlen, scope = 'CQ_attention', dropout=0.0)
    # # e.g. cq_atten_outputs.shape = (32, 80, 200)
    # # use conv_layer to transform above shape(32, 80, 200) to shape(32, 80, 50) as following input
    # cq_outputs = conv_layer(cq_atten_outputs, 1, c.shape[2], 1, 'cq_transform')
    # cqa_atten_outputs = CQ_attention_layer(cq_outputs, a, N, c_maxlen, a_maxlen, scope='CQA_attention', dropout=0.0)
    # outputs = conv_layer(cqa_atten_outputs, 1, output_channel, 1, 'cqa_output')
    # return outputs

    c2q, c_c2q, c_q2c = CQ_attention_layer(c, q, N, c_maxlen, q_maxlen, scope='CQ_attention', dropout=dropout)
    # e.g. cq_atten_outputs.shape = (32, 80, 200)
    # use conv_layer to transform above shape(32, 80, 200) to shape(32, 80, 50) as following input
    c2a, c_c2a, c_a2c = CQ_attention_layer(c, a, N, c_maxlen, a_maxlen, scope='CA_attention', dropout=dropout)
    attention_outputs = [c, c2q, c2a, c_c2q, c_q2c, c_c2a, c_a2c]
    attention_outputs = tf.concat(attention_outputs, axis=-1)
    outputs = conv_layer(attention_outputs, 1, output_channel, 1, 'cqa_output')
    return outputs


def CQA_attention_v2(c, q, a, N, output_channel, c_maxlen, q_maxlen, a_maxlen, dropout=0.0):
    c2q, c_c2q, c_q2c = CQ_attention_layer(c, q, N, c_maxlen, q_maxlen, scope='CQ_attention', dropout=dropout)
    a2c, a_a2c, a_c2a = CQ_attention_layer(a, c, N, a_maxlen, c_maxlen, scope='AC_attention', dropout=dropout)
    attention_output = [a, a2c, a_a2c, a_c2a]
    attention_output = tf.concat(attention_output, axis=-1)
    output = conv_layer(attention_output, 1, output_channel, 1, 'cqa_output')
    return output

#=======================TEST=============================
if __name__ == '__main__':
    with tf.Session():
        c = tf.placeholder(dtype=tf.float32, shape=[32, 400, 96], name='c')
        q = tf.placeholder(dtype=tf.float32, shape=[32, 50, 96], name='c')
        a = tf.placeholder(dtype=tf.float32, shape=[32, 20, 96], name='c')
        # c = tf.constant(np.random.rand(32, 400, 96), dtype=tf.float32)
        # q = tf.constant(np.random.rand(32, 50, 96), dtype=tf.float32)
        # a = tf.constant(np.random.rand(32, 20, 96), dtype=tf.float32)
        c_maxlen = 400
        q_maxlen = 50
        a_maxlen = 20
        N = 32

        CQA_attention_v2(c, q, a, N, 50, c_maxlen, q_maxlen, a_maxlen, 0.3)


