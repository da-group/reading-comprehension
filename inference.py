import tensorflow as tf
import sklearn
import numpy as np
import json

import CQA_attention
import gen_vectors
import block
import matplotlib.pyplot as plt

from encoder_layer import encoder_layer, embedding_layer

import warnings
warnings.filterwarnings('ignore')

batch_size = 64
max_step = 20
c_len = 200
q_len = 50
a_len = 90
train_file = 'splitv2/train_w.json'
test_file = 'splitv2/dev_w.json'
model_name = 'model/best_model'

# def confusion_matrix(prob, label):
#     labels = tf.argmax(label, axis = -1)
#     probs = tf.argmax(prob, axis = -1)
#     return tf.confusion_matrix(labels, probs, num_classes =2)


if __name__ == '__main__':
    # context_train, _ = gen_vectors.trans_sentences(c_len, 50, 'contexts')
    # question_train, _ = gen_vectors.trans_sentences(q_len, 50, 'questions')
    # answer_train, labels = gen_vectors.trans_sentences(a_len, 50, 'answers')

    input_context = tf.placeholder(tf.int32, [batch_size, c_len])
    input_question = tf.placeholder(tf.int32, [batch_size, q_len])
    input_answer = tf.placeholder(tf.int32, [batch_size, a_len])
    y_hat = tf.placeholder(tf.float32, [batch_size, 2])

    context_mask = tf.cast(input_context, tf.bool)
    question_mask = tf.cast(input_question, tf.bool)
    answer_mask = tf.cast(input_answer, tf.bool)

    # embedding
    context = embedding_layer(input_context, 18129, 50, 'word_embedding')
    question = embedding_layer(input_question, 18129, 50, 'word_embedding')
    answer = embedding_layer(input_answer, 18129, 50, 'word_embedding')

    print(input_context.shape)
    print(input_question.shape)
    print(input_answer.shape)
    # input_context = tf.reshape(tf.nn.embedding_lookup(np.array(context_train), input_context), [batch_size*c_len, 50, 64])
    # input_question = tf.reshape(tf.nn.embedding_lookup(np.array(question_train), input_question), [batch_size*q_len, 50, 64])
    # input_answer = tf.reshape(tf.nn.embedding_lookup(np.array(answer_train), input_answer), [batch_size*a_len, 50, 64])

    encoder_context = block.encoder_block(context, num_blocks=2,
                           num_conv_layers=2,
                           kernel_size=3,
                           stride=1,
                           num_d=1,
                           num_p=50,
                           output_channel=50,
                           num_head=8,
                           size_per_head=32,
                           dropout=0.3,
                           name='context',
                           mask = context_mask)

    encoder_question =block.encoder_block(question, num_blocks=2,
                           num_conv_layers=2,
                           kernel_size=3,
                           stride=1,
                           num_d=1,
                           num_p=50,
                           output_channel=50,
                           num_head=8,
                           size_per_head=32,
                           dropout=0.3,
                           name='question',
                           mask = question_mask)

    encoder_answer = block.encoder_block(answer, num_blocks=2,
                           num_conv_layers=2,
                           kernel_size=3,
                           stride=1,
                           num_d=1,
                           num_p=50,
                           output_channel=50,
                           num_head=8,
                           size_per_head=32,
                           dropout=0.3,
                           name='answer',
                           mask = answer_mask)

    output = CQA_attention.CQA_attention_v4(encoder_context, encoder_question, encoder_answer, batch_size,
                                         c_maxlen=c_len, q_maxlen=q_len, a_maxlen=a_len,
                                         c_mask=context_mask, q_mask=question_mask, a_mask=answer_mask,
                                         output_channel=50)
    flatted = tf.layers.flatten(output)

    res = block.fc_layer(flatted, 512, 'fc1', relu=True)
    # res = block.fc_layer(res, 512, 'fc2', relu=True)
    out_layer = tf.layers.dense(res, 2)
    prob = tf.nn.softmax(out_layer)
    pred = tf.argmax(prob, 1)
    labels_bool = tf.argmax(y_hat, 1)

    correct_prediction = tf.equal(labels_bool, pred)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # precision = tf.metrics.precision(labels_bool, pred)[0]
    # recall = tf.metrics.recall(labels_bool, pred)[0]
    # f1 = 2 * precision*recall/(precision+recall)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_hat, logits=out_layer)
    loss = tf.reduce_mean(loss)
    opt = tf.train.AdamOptimizer(0.001).minimize(loss)

    with tf.Session() as sess:
        saver = tf.Saver()
        saver.restore(sess, model_name)
        sess.run(tf.global_variables_initializer())
        train = json.load(open(test_file, 'r'))
        labels = []

        for res in train['results']:
            if not res:
                labels.append([1.0, 0.0])
            else:
                labels.append([0.0, 1.0])
        labels = np.array(labels, dtype=float)

        l = len(labels)//batch_size
        x = []
        y_loss = []
        y_acc = []
        y_f1 = []

        steps = 0

        context_train = train['contexts']
        question_train = train['questions']
        answer_train = train['answers']

        # labels = np.concatenate(labels, axis=0)
        a_mean = 0.0
        f1_mean = 0.0
        for i in range(l):
            acc, pred_value, labels_value = sess.run([accuracy, pred, labels_bool], feed_dict={input_context: context_train[i*batch_size:(i+1)*batch_size],
                                                input_question: question_train[i*batch_size: (i+1)*batch_size],
                                                input_answer: answer_train[i*batch_size: (i+1)*batch_size],
                                                y_hat: labels[i*batch_size:(i + 1)*batch_size]})
            f1 = sklearn.metrics.f1_score(labels_value, pred_value)
            a_mean += acc
            f1_mean += f1
            #if i % 10 == 0:
        print('Accuracy: ', a_mean/(l), '  F1 score: ', f1_mean/l)
