#############################
# Jiachi Zhang
# zhangjiachi1007@gmail.com
#############################

import json
import numpy as np

def convert(file_name, save_name):
    d = json.load(open(file_name, 'r'))
    # v = open(emb_name, 'r')
    contexts = d['contexts']
    questions = d['questions']
    answers = d['answers']
    results = d['results']

    words = {}
    words['NULL'] = 0
    words['OOV'] = 1

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
        for j in range(len(lcs[i]), c_maxlen):
            lcs[i].append(0)
    print(np.array(lcs).shape)
    C = np.array(lcs)[:, :200]

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
        for j in range(len(lqs[i]), q_maxlen):
            lqs[i].append(0)
    print(np.array(lqs).shape)
    Q = np.array(lqs)[:, :50]

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
        for j in range(len(las[i]), a_maxlen):
            las[i].append(0)
    print(np.array(las).shape)
    A = np.array(las)[:, :90]

    new_d = {'contexts': C.tolist(), 'questions': Q.tolist(), 'answers': A.tolist(), 'results': results}
    json.dump(new_d, open(save_name, 'w'))

    print(len(words))


if __name__ == "__main__":
    convert('./splitv2/train.json', './''./splitv2/train.json')
    convert('./splitv2/dev.json', './''./splitv2/dev.json')
