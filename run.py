import tensorflow as tf
import numpy as np

import CQA_attention
import gen_vectors
import block
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

batch_size = 64
max_step = 20
c_len = 192
a_len = 48
q_len = 89

if __name__ == '__main__':
    context_train, _ = gen_vectors.trans_sentences(c_len, 50, 'contexts')
    question_train, _ = gen_vectors.trans_sentences(q_len, 50, 'questions')
    answer_train, labels = gen_vectors.trans_sentences(a_len, 50, 'answers')

    input_context = tf.placeholder(tf.float32, [batch_size, c_len, 50])
    input_question = tf.placeholder(tf.float32, [batch_size, q_len, 50])
    input_answer = tf.placeholder(tf.float32, [batch_size, a_len, 50])
    y_hat = tf.placeholder(tf.float32, [None, 2])

    encoder_context = block.encoder_block(input_context, num_blocks=2,
                           num_conv_layers=3,
                           kernel_size=3,
                           stride=1,
                           num_d=128,
                           num_p=50,
                           output_channel=50,
                           num_head=8,
                           size_per_head=32,
                           dropout=0.3,
                           name='context')

    encoder_question =block.encoder_block(input_question, num_blocks=2,
                           num_conv_layers=3,
                           kernel_size=3,
                           stride=1,
                           num_d=128,
                           num_p=50,
                           output_channel=50,
                           num_head=8,
                           size_per_head=32,
                           dropout=0.3,
                           name='question')

    encoder_answer = block.encoder_block(input_answer, num_blocks=2,
                           num_conv_layers=3,
                           kernel_size=3,
                           stride=1,
                           num_d=128,
                           num_p=50,
                           output_channel=50,
                           num_head=8,
                           size_per_head=32,
                           dropout=0.3,
                           name='answer')

    output = CQA_attention.CQA_attention(encoder_context, encoder_question, encoder_answer, batch_size,
                                         c_maxlen=c_len, q_maxlen=q_len, a_maxlen=a_len, output_channel=50)
    flatted = tf.layers.flatten(output)

    res = block.fc_layer(flatted, 50, 'fc1', relu=True)
    out_layer = tf.layers.dense(res, 2)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_hat, logits=out_layer)
    loss = tf.reduce_mean(loss)
    opt = tf.train.AdamOptimizer(0.01).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(out_layer, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    with tf.Session() as sess:
        f = open('result_text.txt', 'w')
        sess.run(tf.global_variables_initializer())

        l = len(labels)//batch_size
        x = []
        y_loss = []
        y_acc = []

        for j in range(1):
            a_mean = 0
            for i in range(l):
                loss_value, _, acc = sess.run([loss, opt, accuracy], feed_dict={input_context: context_train[i*batch_size:(i+1)*batch_size],
                                                 input_question: question_train[i*batch_size: (i+1)*batch_size],
                                                 input_answer: answer_train[i*batch_size: (i+1)*batch_size],
                                                 y_hat: labels[i*batch_size: (i+1)*batch_size]})
                a_mean += acc
                if i % 20 == 0:
                    print('Step: ', j, ' Loss: ', loss_value, '  Accuracy: ', a_mean/l)
                    x.append(j)
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

