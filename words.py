#############################
# Jiachi Zhang
# zhangjiachi1007@gmail.com
#############################

import json
import numpy as np

C_MAX = 200
Q_MAX = 50
A_MAX = 90

def convert(file_name, save_name):
    d = json.load(open(file_name, 'r'))
    # v = open(emb_name, 'r')
    contexts = d['contexts']
    questions = d['questions']
    answers = d['answers']
    results = d['results']

    words = {}
    words['NULL'] = 0
    words['oov'] = 1

    lcs = []
    c_maxlen = 0
    for context in contexts:
        lc = []
        for word in context:
            word = word.lower()
            if word not in words.keys():
                words[word] = len(words)
            lc.append(words[word])
        if len(lc)>c_maxlen:
            c_maxlen = len(lc)
        lcs.append(lc)
    print(c_maxlen)

    for i in range(len(lcs)):
        for j in range(len(lcs[i]), C_MAX):
            lcs[i].append(0)
    C = np.array(lcs)[:, :200]
    print(C.shape)

    lqs = []
    q_maxlen = 0
    for context in questions:
        lq = []
        for word in context:
            word = word.lower()
            if word not in words.keys():
                words[word] = len(words)
            lq.append(words[word])
        if len(lq)>q_maxlen:
            q_maxlen = len(lq)
        lqs.append(lq)
    print(q_maxlen)

    for i in range(len(lqs)):
        for j in range(len(lqs[i]), Q_MAX):
            lqs[i].append(0)
    Q = np.array(lqs)[:, :50]
    print(Q.shape)

    las = []
    a_maxlen = 0
    for context in answers:
        la = []
        for word in context:
            word = word.lower()
            if word not in words.keys():
                words[word] = len(words)
            la.append(words[word])
        if len(la)>a_maxlen:
            a_maxlen = len(la)
        las.append(la)
    print(a_maxlen)

    for i in range(len(las)):
        for j in range(len(las[i]), A_MAX):
            las[i].append(0)
    A = np.array(las)[:, :90]
    print(A.shape)

    new_d = {'contexts': C.tolist(), 'questions': Q.tolist(), 'answers': A.tolist(), 'results': results}
    json.dump(new_d, open(save_name, 'w'))
    json.dump(words, open('./splitv2/words.json', 'w'))

    print(len(words))


def convertWithDict(file_name, save_name, dict_name):
    words = json.load(open(dict_name, 'r'))
    d = json.load(open(file_name, 'r'))
    contexts = d['contexts']
    questions = d['questions']
    answers = d['answers']
    results = d['results']

    oov = []

    lcs = []
    c_maxlen = 0
    for context in contexts:
        lc = []
        for word in context:
            if word not in words.keys():
                oov.append(word)
                word = 'oov'
            lc.append(words[word])
        if len(lc)>c_maxlen:
            c_maxlen = len(lc)
        lcs.append(lc)
    print(c_maxlen)

    for i in range(len(lcs)):
        for j in range(len(lcs[i]), C_MAX):
            lcs[i].append(0)
    print(np.array(lcs).shape)
    C = np.array(lcs)[:, :200]

    lqs = []
    q_maxlen = 0
    for context in questions:
        lq = []
        for word in context:
            if word not in words.keys():
                oov.append(word)
                word = 'oov'
            lq.append(words[word])
        if len(lq)>q_maxlen:
            q_maxlen = len(lq)
        lqs.append(lq)
    print(q_maxlen)

    for i in range(len(lqs)):
        for j in range(len(lqs[i]), Q_MAX):
            lqs[i].append(0)
    print(np.array(lqs).shape)
    Q = np.array(lqs)[:, :50]

    las = []
    a_maxlen = 0
    for context in answers:
        la = []
        for word in context:
            if word not in words.keys():
                oov.append(word)
                word = 'oov'
            la.append(words[word])
        if len(la)>a_maxlen:
            a_maxlen = len(la)
        las.append(la)
    print(a_maxlen)

    for i in range(len(las)):
        for j in range(len(las[i]), A_MAX):
            las[i].append(0)
    print(np.array(las).shape)
    A = np.array(las)[:, :90]

    new_d = {'contexts': C.tolist(), 'questions': Q.tolist(), 'answers': A.tolist(), 'results': results}
    json.dump(new_d, open(save_name, 'w'))

    print(oov)


if __name__ == "__main__":
    convert('./splitv2/train_clean.json', './splitv2/train_clean_w.json')
    convertWithDict('./splitv2/dev_clean.json', './splitv2/dev_clean_w.json', './splitv2/words.json')
