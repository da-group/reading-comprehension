import tensorflow as tf
import numpy as np
import json

import CQA_attention
import gen_vectors
import block
import matplotlib.pyplot as plt

from encoding_layer import embedding_layer, encoder_layer

import warnings
warnings.filterwarnings('ignore')

batch_size = 64
max_step = 20
c_len = 200
a_len = 90
q_len = 50
ch_len = 20


if __name__ == '__main__':
    context_train, _ = gen_vectors.trans_sentences(c_len, 50, 'contexts')
    question_train, _ = gen_vectors.trans_sentences(q_len, 50, 'questions')
    answer_train, labels = gen_vectors.trans_sentences(a_len, 50, 'answers')

    train_c = json.load(open('./splitv2/train_c.json', 'r'))
    context_c = train_c['contexts']
    question_c = train_c['questions']
    answer_c = train_c['answers']


    input_context = tf.placeholder(tf.float32, [batch_size, c_len, 50])
    input_question = tf.placeholder(tf.float32, [batch_size, q_len, 50])
    input_answer = tf.placeholder(tf.float32, [batch_size, a_len, 50])
    y_hat = tf.placeholder(tf.float32, [None, 2])

    context_mask = tf.cast(tf.reduce_sum(input_context, -1), tf.bool)
    question_mask = tf.cast(tf.reduce_sum(input_question, -1), tf.bool)
    answer_mask = tf.cast(tf.reduce_sum(input_answer, -1), tf.bool)

    # character operation
    context_ch = tf.placeholder(tf.float32, [batch_size, c_len, ch_len])
    question_ch = tf.placeholder(tf.float32, [batch_size, q_len, ch_len])
    answer_ch = tf.placeholder(tf.float32, [batch_size, a_len, ch_len])

    context_ch = tf.reshape(embedding_layer(context_ch, 95, 50, 'character_embedding'), [batch_size*c_len, ch_len, 50])
    question_ch = tf.reshape(embedding_layer(question_ch, 95, 50, 'character_embedding'), [batch_size*q_len, ch_len, 50])
    answer_ch = tf.reshape(embedding_layer(answer_ch, 95, 50, 'character_embedding'), [batch_size*a_len, ch_len, 50])

    context_ch = tf.reduce_max(block.conv_layer(context_ch, 5, 50, 1, 'ch_conv', relu=True), axis=1)
    question_ch = tf.reduce_max(block.conv_layer(question_ch, 5, 50, 1, 'ch_conv', relu=True), axis=1)
    answer_ch = tf.reduce_max(block.conv_layer(answer_ch, 5, 50, 1, 'ch_conv', relu=True), axis=1)

    context_ch = tf.reshape(context_ch, [batch_size, c_len, 50])
    question_ch = tf.reshape(question_ch, [batch_size, q_len, 50])
    answer_ch = tf.reshape(answer_ch, [batch_size, a_len, 50])


    encoder_context = block.encoder_block(input_context, num_blocks=2,
                           num_conv_layers=2,
                           kernel_size=7,
                           stride=1,
                           num_d=1,
                           num_p=50,
                           output_channel=50,
                           num_head=8,
                           size_per_head=32,
                           dropout=0.3,
                           name='context',
                           mask = context_mask)

    encoder_question =block.encoder_block(input_question, num_blocks=2,
                           num_conv_layers=2,
                           kernel_size=7,
                           stride=1,
                           num_d=1,
                           num_p=50,
                           output_channel=50,
                           num_head=8,
                           size_per_head=32,
                           dropout=0.3,
                           name='question',
                           mask = question_mask)

    encoder_answer = block.encoder_block(input_answer, num_blocks=2,
                           num_conv_layers=2,
                           kernel_size=7,
                           stride=1,
                           num_d=1,
                           num_p=50,
                           output_channel=50,
                           num_head=8,
                           size_per_head=32,
                           dropout=0.3,
                           name='answer',
                           mask = answer_mask)

    output = CQA_attention.CQA_attention_v2(encoder_context, encoder_question, encoder_answer, batch_size,
                                         c_maxlen=c_len, q_maxlen=q_len, a_maxlen=a_len, c_mask=context_mask, q_mask=question_mask, a_mask=answer_mask, output_channel=50)


    output = block.encoder_block(output, num_blocks=2,
                           num_conv_layers=2,
                           kernel_size=7,
                           stride=1,
                           num_d=1,
                           num_p=50,
                           output_channel=50,
                           num_head=8,
                           size_per_head=32,
                           dropout=0.3,
                           name='answer',
                           mask = answer_mask)

    flatted = tf.layers.flatten(output)

    res = block.fc_layer(flatted, 512, 'fc1', relu=True)
    # res = block.fc_layer(res, 512, 'fc2', relu=True)
    out_layer = tf.layers.dense(res, 2)
    pred = tf.nn.softmax(out_layer)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_hat, logits=out_layer)
    loss = tf.reduce_mean(loss)
    opt = tf.train.AdamOptimizer(0.001).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    with tf.Session() as sess:
        f = open('result_text.txt', 'w')
        sess.run(tf.global_variables_initializer())

        l = len(labels)//batch_size
        x = []
        y_loss = []
        y_acc = []

        steps = 0

        context_train = np.array(context_train)
        question_train = np.array(question_train)
        answer_train = np.array(answer_train)
        labels = np.array(labels)

        for j in range(100):
            a_mean = 0
            L = np.arange(len(labels))
            np.random.shuffle(L)
            print(L)
            print(type(context_train))
            for i in range(l):
                steps += 1
                loss_value, _, acc, pred_value = sess.run([loss, opt, accuracy, pred], feed_dict={input_context: context_train[L[i*batch_size:(i+1)*batch_size]],
                                                 input_question: question_train[L[i*batch_size: (i+1)*batch_size]],
                                                 input_answer: answer_train[L[i*batch_size: (i+1)*batch_size]],
                                                 y_hat: labels[L[i*batch_size: (i+1)*batch_size]]})
                a_mean += acc
                if steps % 1 == 0:
                    print('Step: ', steps, ' Loss: ', loss_value, '  Accuracy: ', a_mean/(i+1))
                    x.append(steps)
                    y_acc.append(a_mean/l)
                    y_loss.append(loss_value)
                    f.write(str(a_mean/l)+' '+str(loss_value)+'\n')

        plt.plot(x, y_acc)
        plt.xlabel('step')
        plt.ylabel('accuracy')
        plt.savefig('acc.png')
        plt.plot(x, y_loss)
        plt.xlabel('step')
        plt.ylabel('loss')
        plt.savefig('loss.png')
        f.close()

