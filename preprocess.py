#############################
# Jiachi Zhang
# zhangjiachi1007@gmail.com
#############################

import json
import string

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def splitPunc(sentence):
    re = []
    for word in sentence:
        if word in string.punctuation:
            re.append(word)
            continue
        if word[0] in string.punctuation:
            re.append(word[0])
            word = word[1:]
        if word[-1] in string.punctuation:
            re.append(word[:-1])
            re.append(word[-1])
        else:
            re.append(word)
    return re

def clean(sentence, d):
    '''
    sentence: a string
    '''
    sentence = sentence.split()
    res = []
    for word in sentence:
        word = word.lower()
        word = word.strip(string.punctuation)
        if word.endswith("n't"):
            w1 = word[:-3]
            w2 = 'not'
            res.append(w1)
            res.append(w2)
            continue
        if "'" in word:
            word = word.split("'")[0]
        res.append(word)
    for i in range(len(res)):
        res[i] = lemmatizer.lemmatize(res[i])
    re = []
    for ele in res:
        if ele in d:
            re.append(ele)
    return re

def listToString(sentence):
    re = ''
    for word in sentence:
        re += word
        re += ' '
    return re.strip()


def convert(file_name, save_name, d):
    f = open(save_name, 'w')
    dic = json.load(open(file_name, 'r'))
    data = dic['data']
    Contexts = []
    Questions = []
    Answers = []
    Results = []
    for material in data:
        para = material['paragraph']
        text = para['text']
        sentences = text.split('<br>')
        sentences = [sentence.split('</b>')[-1] for sentence in sentences][:-1]
        sentences = [clean(sentence, d) for sentence in sentences]
        questions = para['questions']
        for question in questions:
            # get contex
            index = question['sentences_used']
            c = []
            for i in index:
                c += sentences[i-1]
            # get question
            q = question['question']
            q = clean(q, d)
            # get answers
            answers = question['answers']
            for answer in answers:
                a = clean(answer['text'], d)
                r = answer['isAnswer']
                Contexts.append(c)
                Questions.append(q)
                Answers.append(a)
                Results.append(r)
    d = {'contexts': Contexts, 'questions': Questions, 'answers': Answers, 'results': Results}
    json.dump(d, f)
    f.close()


if __name__ =='__main__':
    f = open('./splitv2/glove.6B.50d.txt', 'r')
    d = []
    for line in f.readlines():
        line = line.strip()
        d.append(line.split(' ')[0].lower())
    convert('./splitv2/dev_83-fixedIds.json', './splitv2/dev_clean.json', d)
    convert('./splitv2/train_456-fixedIds.json', './splitv2/train_clean.json', d)







