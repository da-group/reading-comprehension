#############################
# Jiachi Zhang
# zhangjiachi1007@gmail.com
#############################

import json
import numpy as np

def convert(file_name, save_name):
    d = json.load(open(file_name, 'r'))
    contexts = d['contexts']
    questions = d['questions']
    answers = d['answers']

    characters = {}
    characters['NULL'] = 0


    lcs = []
    c_maxlen = 0
    ch_maxlen = 0
    for context in contexts:
        lc = []
        for word in context:
            l = []
            word = word.lower()
            for c in word:
                if c not in characters.keys():
                    characters[c] = len(characters.keys())
                l.append(characters[c])
            if len(l)>ch_maxlen:
                ch_maxlen = len(l)
            lc.append(l)
        if len(lc)>c_maxlen:
            c_maxlen = len(lc)
        lcs.append(lc)
    print(c_maxlen)

    for i in range(len(lcs)):
        for j in range(c_maxlen):
            if j<len(lcs[i]):
                for k in range(len(lcs[i][j]), ch_maxlen):
                    lcs[i][j].append(0)
            else:
                lcs[i].append([0 for _ in range(ch_maxlen)])
    C = np.array(lcs)[:, :200, :20]


    lqs = []
    q_maxlen = 0
    qh_maxlen = 0
    for context in questions:
        lq = []
        for word in context:
            l = []
            word = word.lower()
            for c in word:
                if c not in characters.keys():
                    characters[c] = len(characters.keys())
                l.append(characters[c])
            if len(l)>qh_maxlen:
                qh_maxlen = len(l)
            lq.append(l)
        if len(lq)>q_maxlen:
            q_maxlen = len(lq)
        lqs.append(lq)
    print(q_maxlen)

    for i in range(len(lqs)):
        for j in range(q_maxlen):
            if j<len(lqs[i]):
                for k in range(len(lqs[i][j]), qh_maxlen):
                    lqs[i][j].append(0)
            else:
                lqs[i].append([0 for _ in range(qh_maxlen)])
    Q = np.array(lqs)[:, :50, :20]



    las = []
    a_maxlen = 0
    ah_maxlen = 0
    for context in answers:
        la = []
        for word in context:
            l = []
            word = word.lower()
            for c in word:
                if c not in characters.keys():
                    characters[c] = len(characters.keys())
                l.append(characters[c])
            if len(l)>ah_maxlen:
                print(word)
                ah_maxlen = len(l)
            la.append(l)
        if len(la)>a_maxlen:
            a_maxlen = len(la)
        las.append(la)
    print(a_maxlen)
    print(ah_maxlen)

    for i in range(len(las)):
        for j in range(a_maxlen):
            if j<len(las[i]):
                for k in range(len(las[i][j]), ah_maxlen):
                    las[i][j].append(0)
            else:
                las[i].append([0 for _ in range(ah_maxlen)])
    A = np.array(las)[:, :90, :20]


    new_d = {'contexts': C.tolist(), 'questions': Q.tolist(), 'answers': A.tolist()}
    json.dump(new_d, open(save_name, 'w'))


if __name__ == "__main__":
    convert('./splitv2/train.json', './splitv2/train_c.json')
