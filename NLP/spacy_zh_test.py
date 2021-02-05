# Chinese NLP test via spaCy

import spacy
from collections import Counter
#from tabulate import tabulate
from pathlib import Path

p = Path('.')   # current Path

# 飞雪连天射白鹿，笑书神侠倚碧鸳

feihu = list(p.glob('novels/jinyong/feihuwaizhuan/*.txt'))
feihu.sort()

xueshan = list(p.glob('novels/jinyong/xueshanfeihu/*.txt'))
xueshan.sort()

liancheng = list(p.glob('novels/jinyong/lianchengjue/*.txt'))
liancheng.sort()

tianlong = list(p.glob('novels/jinyong/tianlongbabu/*.txt'))
tianlong.sort()

shediao = list(p.glob('novels/jinyong/shediaoyingxiongzhuan/*.txt'))
shediao.sort()

baima = list(p.glob('novels/jinyong/baimaxiaoxifeng/*.txt'))
baima.sort()

luding = list(p.glob('novels/jinyong/ludingji/*.txt'))
luding.sort()

xiaoao = list(p.glob('novels/jinyong/xiaoaojianghu/*.txt'))
xiaoao.sort()

shujian = list(p.glob('novels/jinyong/shujianenchoulu/*.txt'))
shujian.sort()

shendiao = list(p.glob('novels/jinyong/shendiaoxialv/*.txt'))
shendiao.sort()

xiakexing = list(p.glob('novels/jinyong/xiakexing/*.txt'))
xiakexing.sort()

yitian = list(p.glob('novels/jinyong/yitiantulongji/*.txt'))
yitian.sort()

bixuejian = list(p.glob('novels/jinyong/bixuejian/*.txt'))
bixuejian.sort()

yuanyang = list(p.glob('novels/jinyong/yuanyangdao/*.txt'))
yuanyang.sort()

person_names = []
with open('novels/jinyong/person_list.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        person_names.append(line.rstrip('\n'))

# spacy.prefer_gpu()  # Using GPU to run programm

nlp = spacy.load("zh_core_web_lg")

def read_chapters(book):
    text = ''
    for chapter in book:
        with open(chapter, 'r') as file:
            text += file.read()
    return text

#shendiao_text = read_chapters(shendiao)
shediao_text = read_chapters(shediao)
#tianlong_text =read_chapters(tianlong)

#shendiao_doc = nlp(shendiao_text)
shediao_doc = nlp(shediao_text)
#ianlong_doc = nlp(tianlong_text)

def token_is_subject_with_action(token):
    nsubj = token.dep_ == 'nsubj'
    head_verb = token.head.pos_ == 'VERB'
    person = token.ent_type_ == 'PERSON'
    in_person = token.text in person_names
    return nsubj and head_verb and person and in_person

def token_is_object(token):
    dobj = token.dep_ == 'dobj'
    return dobj

def check_action(token):
    pass

def check_battle_words(token):  # 暂时将含有「杀」、「伤」、「胜」、「败」的动词统计进来
    is_verb = token.pos_ == 'VERB'
    has_keywords = ('杀' in token.text) or ('伤' in token.text) or ('胜' in token.text) or ('败' in token.text)
    return is_verb and has_keywords

def count_ents(doc):
    entities = {}
    for ent in doc.ents:
        if ent.text in entities:
            entities[ent.text] += 1
        else:
            entities[ent.text] = 1
    print("top entities{}".format(sorted(entities.items(), key=lambda kv: kv[1], reverse=True)[:30]))

def walk_through(doc):
    assert doc.has_annotation("SENT_START")
    nsubj = '路人甲'
    dobj = '飞刀'
    action = '躲开了'
    for sent in doc.sents:
        for token in sent:
            if (token_is_subject_with_action(token) == True):
                nsubj = token.text
            elif token_is_object(token) == True:
                dobj = token.text
        action = sent.root
        if (check_battle_words(action) == True):
            print(sent.text)

def calc_similarity(doc1, doc2):
    return doc1.similarity(doc2)

def time_line(person, doc):
    assert doc.has_annotation("SENT_START")
    person_track = []
    for sent in doc.sents:
        has_person = False
        for token in sent:
            if token.text == person:    # Find person here.
                has_person = True
        if has_person:
            person_track.append(sent)
                
    return person_track

def abstract(sent):
    abs = ''
    for token in sent:
        if token.dep_ == 'nsubj':
            abs += token.text
        elif token.dep_ == 'ROOT':
            abs += token.text
        elif token.dep_ == 'dobj':
            abs += token.text
    return abs

def count_labels(doc):
    labels = {}
    for ent in doc.ents:
        if ent.label_ in labels:
            labels[ent.label_] += 1
        else:
            labels[ent.label_] = 1
    print(sorted(labels.items(), key=lambda kv: kv[1], reverse=True))

def life_of_yours(name, doc):
    events = []
    lifeline = time_line(name, doc)
    for spark in lifeline:
        events.append(abstract(spark))
    return events

#print("Similarity between 神雕侠侣 and 射雕英雄传 docs is{}".format(calc_similarity(shediao_doc, shendiao_doc)))
#walk_through(shediao_doc)

# Labels Counting results in 「射雕英雄传」
# [('PERSON', 19555), ('CARDINAL', 7091), ('GPE', 1758), ('DATE', 1655), ('LOC', 665), ('ORG', 585), ('WORK_OF_ART', 416), 
# ('TIME', 362), ('ORDINAL', 341), ('FAC', 334), ('NORP', 216), ('EVENT', 87), ('PERCENT', 78), ('QUANTITY', 62), ('PRODUCT', 43), ('LANGUAGE', 30), ('MONEY', 19), ('LAW', 1)]

def check_events(doc):
    events = {}
    assert doc.has_annotation("SENT_START")
    for sent in doc.sents:
        for ent in sent.ents:
            if ent.label_ == 'EVENT':
                events[ent.text] = sent.text
    return events

shediao_events = check_events(shediao_doc)
print(sorted(shediao_events.items(), key=lambda kv: kv[1], reverse=True))
