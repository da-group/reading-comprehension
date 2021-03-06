import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np



def encoder_layer(input_sents,pretrained_vector,dropout=0.0):
    '''
        input_sents.shape = (batch_size, sentence_length_limitation) dtype=int32
        pretrained_vector.shape = (corpus_size, vector_length) type = tfe.Variable
        return.shape = (batch_size, sentence_length_limitation, vector_length)
    '''
    vector_len = pretrained_vector.shape[1]
    num_sents = input_sents.shape[0]
    len_sents = input_sents.shape[1]
    output = tf.nn.dropout(tf.nn.embedding_lookup(
        pretrained_vector, input_sents,[num_sents, len_sents, vector_len]), 1.0-dropout)
    return output

def embedding_layer(parent, vocab_size, embedding_size, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        word_embeddings = tf.get_variable("word_embeddings", [vocab_size, embedding_size], tf.float32)
        output = tf.nn.embedding_lookup(word_embeddings, parent)
        return output


if __name__ == '__main__':
    input_sents = np.array([[0, 1], [2, 3]])
    pretrained_vector = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    print(encoder_layer(input_sents, pretrained_vector, dropout=0.0))
