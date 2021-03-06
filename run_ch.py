import tensorflow as tf
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
ch_len = 20
train_file = 'splitv2/train_w.json'

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

    # character operation
    context_c = tf.placeholder(tf.int32, [batch_size, c_len, ch_len])
    question_c = tf.placeholder(tf.int32, [batch_size, q_len, ch_len])
    answer_c = tf.placeholder(tf.int32, [batch_size, a_len, ch_len])

    context_ch = tf.reshape(embedding_layer(context_c, 95, 50, 'character_embedding'), [batch_size*c_len, ch_len, 50])
    question_ch = tf.reshape(embedding_layer(question_c, 95, 50, 'character_embedding'), [batch_size*q_len, ch_len, 50])
    answer_ch = tf.reshape(embedding_layer(answer_c, 95, 50, 'character_embedding'), [batch_size*a_len, ch_len, 50])

    context_ch = tf.reduce_max(block.conv_layer(context_ch, 5, 50, 1, 'ch_conv', relu=True), axis=1)
    question_ch = tf.reduce_max(block.conv_layer(question_ch, 5, 50, 1, 'ch_conv', relu=True), axis=1)
    answer_ch = tf.reduce_max(block.conv_layer(answer_ch, 5, 50, 1, 'ch_conv', relu=True), axis=1)

    context_ch = tf.reshape(context_ch, [batch_size, c_len, 50])
    question_ch = tf.reshape(question_ch, [batch_size, q_len, 50])
    answer_ch = tf.reshape(answer_ch, [batch_size, a_len, 50])

    context = tf.concat([context, context_ch], axis=-1)
    question = tf.concat([question, question_ch], axis=-1)
    answer = tf.concat([answer, answer_ch], axis=-1)

    context = block.highway(context, 1, 50, name='highway', dropout=0.3)
    question = block.highway(question, 1, 50, name='highway', dropout=0.3)
    answer = block.highway(answer, 1, 50, name='highway', dropout=0.3)

    print(context.shape)
    print(question.shape)
    print(answer.shape)
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

    output = CQA_attention.CQA_attention(encoder_context, encoder_question, encoder_answer, batch_size,
                                         c_maxlen=c_len, q_maxlen=q_len, a_maxlen=a_len,
                                         c_mask=context_mask, q_mask=question_mask, a_mask=answer_mask,
                                         output_channel=50)
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
        train = json.load(open(train_file, 'r'))
        train_c = json.load(open('./splitv2/train_c.json', 'r'))
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

        steps = 0

        context_train = train['contexts']
        question_train = train['questions']
        answer_train = train['answers']
        context_train_ch = train_c['contexts']
        question_train_ch = train_c['questions']
        answer_train_ch = train_c['answers']

        # labels = np.concatenate(labels, axis=0)

        for j in range(30):
            a_mean = 0
            L = np.arange(len(labels))
            np.random.shuffle(L)
            # print(L)
            for i in range(l):
                steps += 1
                loss_value, _, acc, pred_value = sess.run([loss, opt, accuracy, pred], feed_dict={input_context: context_train[i*batch_size:(i+1)*batch_size],
                                                 input_question: question_train[i*batch_size: (i+1)*batch_size],
                                                 input_answer: answer_train[i*batch_size: (i+1)*batch_size],
                                                 y_hat: labels[i*batch_size: (i+1)*batch_size],
                                                 context_c: context_train_ch[i*batch_size: (i+1)*batch_size],
                                                 question_c: question_train_ch[i*batch_size: (i+1)*batch_size],
                                                 answer_c: answer_train_ch[i*batch_size: (i+1)*batch_size]})
                a_mean += acc
                if i % 1 == 0:
                    print('Step: ', steps, ' Loss: ', loss_value, '  Accuracy: ', acc)
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

