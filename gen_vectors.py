import json
import numpy as np

train_file = 'splitv2/train.json'
embedding_file = 'embedding_'
dict_file = 'dict_'

def trans_sentences(length, vec_size, part):
    emb = embedding_file + part + '.json'
    d = dict_file + part + '.json'
    train = json.load(open(train_file, 'r'))
    embedding = json.load(open(emb, 'r'))
    dic = json.load(open(d, 'r'))

    matrix = []
    sentences = []
    labels = []
    slen = []

    for sen in train[part]:
        for word in sen:
            word = word.lower()
            if word in dic.keys():
                matrix.append(embedding[dic[word]])
        slen.append(len(matrix))
        if len(matrix) < length:
            for i in range(length-len(matrix)):
                matrix.append([0.0]*vec_size)
        else:
            matrix = matrix[:length]
        sentences.append(np.array(matrix)) # length*vec_size
        matrix = []
    print(max(slen))

    for res in train['results']:
        if not res:
            labels.append([1.0, 0.0])
        else:
            labels.append([0.0, 1.0])
    labels = np.array(labels)

    return sentences, labels




