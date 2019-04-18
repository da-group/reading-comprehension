#############################
# Jiachi Zhang
# zhangjiachi1007@gmail.com
#############################

import json
import string


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


def listToString(sentence):
    re = ''
    for word in sentence:
        re += word
        re += ' '
    return re.strip()


def convert(file_name, save_name):
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
        sentences = [splitPunc(sentence.split(' ')) for sentence in sentences]
        questions = para['questions']
        for question in questions:
            # get context
            index = question['sentences_used']
            c = []
            for i in index:
                c += sentences[i-1]
            # get question
            q = question['question']
            q = splitPunc(q.split(' '))
            # get answers
            answers = question['answers']
            for answer in answers:
                a = splitPunc(answer['text'].split(' '))
                r = answer['isAnswer']
                Contexts.append(c)
                Questions.append(q)
                Answers.append(a)
                Results.append(r)
    d = {'contexts': Contexts, 'questions': Questions, 'answers': Answers, 'results': Results}
    json.dump(d, f)
    f.close()


if __name__ =='__main__':
    convert('./splitv2/dev_83-fixedIds.json', './splitv2/dev.json')
    convert('./splitv2/train_456-fixedIds.json', './splitv2/train.json')







