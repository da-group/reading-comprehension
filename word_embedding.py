from collections import Counter
import json
import numpy as np

train_file = 'splitv2/train.json'
fname = 'embedding.json'
e_file = 'glove.6B.50d.txt'

def get_embedding(word_file, emb_file, vec_size, part):
    words = json.load(open(word_file, 'r'))
    embedding_dict = {}
    word_counter = Counter()

    for sentence in words[part]:
        word_counter += Counter(sentence)

    elements = [k for k in word_counter.keys()]
    with open(emb_file, 'r', encoding='utf-8') as ef:
        for line in ef:
            array = line.split()
            word = "".join(array[0: -vec_size])
            vector = list(map(float, array[-vec_size:]))
            if word in word_counter.keys():
                embedding_dict[word] = vector
            else:
                embedding_dict[word] = [np.random.normal(scale=0.1) for _ in range(vec_size)]

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token] for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict

def save_file(name, d, part):
    name = name + '_' + part
    with open(name, 'w') as wf:
        json.dump(d, wf)

if __name__ == '__main__' :
    parts = ['contexts', 'questions', 'answers']
    for part in parts:
        mat, _ = get_embedding(train_file, e_file, 50, part)
        save_file(fname, mat, part)