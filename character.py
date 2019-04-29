#############################
# Jiachi Zhang
# zhangjiachi1007@gmail.com
#############################

import json

def convert(file_name, save_name):
    d = json.load(open(file_name, 'r'))
    contexts = d['contexts']
    questions = d['questions']
    answers = d['answers']

    characters = {}
    characters[' '] = 0

    lcs = []
    c_maxlen = 0
    for context in contexts:
        lc = []
        for word in context:
            for c in word:
                if c not in characters.keys():
                    characters[c] = len(characters.keys())
                lc.append(characters[c])
            lc.append(characters[' '])
            if len(lc)>c_maxlen:
                c_maxlen = len(lc)-1
        lcs.append(lc[:-1])
    print(c_maxlen)

    lqs = []
    q_maxlen = 0
    for question in questions:
        lq = []
        for word in question:
            for c in word:
                if c not in characters.keys():
                    characters[c] = len(characters.keys())
                lq.append(characters[c])
            lq.append(characters[' '])
            if len(lq)>q_maxlen:
                q_maxlen = len(lq)-1
        lqs.append(lq[:-1])
    print(q_maxlen)

    las = []
    a_maxlen = 0
    for answer in answers:
        la = []
        for word in context:
            for c in word:
                if c not in characters.keys():
                    characters[c] = len(characters.keys())
                la.append(characters[c])
            la.append(characters[' '])
            if len(la)>a_maxlen:
                a_maxlen = len(la)-1
        las.append(la[:-1])
    print(a_maxlen)

    new_d = {'contexts': lcs, 'questions': lqs, 'answers': las}
    json.dump(new_d, open(save_name, 'w'))


if __name__ == "__main__":
    convert('./splitv2/train.json', './splitv2/train_c.json')
