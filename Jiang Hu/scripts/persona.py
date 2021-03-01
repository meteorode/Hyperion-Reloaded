# persona.py

# Initilization

import spacy
from collections import Counter
#from tabulate import tabulate
from pathlib import Path
import time

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

# 杯雪
beixue = list(p.glob('books/beixue/*.txt'))
beixue.sort()

spacy.prefer_gpu()  # Using GPU to run programm

#nlp = spacy.load("zh_core_web_lg")
nlp = spacy.load('zh_core_web_trf') # spacy 3.0 stable model.

def read_chapters(book):
    txts = []
    for chapter in book:
        with open(chapter, 'r') as file:
            txts.append(file.read())
    return txts

# Basic Statistics methods

def count_attrs(doc, attr_type):
    entities = {}
    if attr_type == 'ENT':
        for ent in doc.ents:
            if ent.text in entities:
                entities[ent.text] += 1
            else:
                entities[ent.text] = 1
    elif attr_type == 'LABEL':
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_] += 1
            else:
                entities[ent.label_] = 1
    #print("top entities{}".format(sorted(entities.items(), key=lambda kv: kv[1], reverse=True)[:30]))
    return entities

def count_words_with_specs(doc, spec_type, spec_name):
    result = {}
    if spec_type == 'LABEL':
        for ent in doc.ents:
            if ent.label_ == spec_name:
                if ent.text in result:
                    result[ent.text] += 1
                else:
                    result[ent.text] = 1
    
    return result

def word_cloud(name, doc, pos_types, dep_types): # First count subtrees of a token, then _TO_BE_CONTINUED_
    cloud = {}
    for token in doc:
        if name in token.text:
            for leaf in token.sent:
                if leaf.pos_ in pos_types or leaf.dep_ in dep_types:
                    if leaf.text in cloud:
                        cloud[leaf.text] += 1
                    else:
                        cloud[leaf.text] = 1
    return cloud

def thou_life(name, doc):
    bio = []
    sents = list(doc.sents)
    for sent in sents:
        word = ''
        for token in sent:
            if name in token.text and token.pos_ == 'PROPN':
                for t in sent:
                    if (t.dep_ == 'nsubj' or t.dep_ == 'pobj' or t.dep_ == 'dobj' or t.text == token.text) and (t.text not in word):
                        word += t.text + ' '
                    elif (t.head.text not in word):
                        word += t.head.text + ' '
                bio.append(word + '\n')
    return bio

# Part I: Sentiment Analysis to determine A char's 「心」traits

# Part II: Skill analysis

# Part III: Body measurement traits analysis

# Part IV: Intelligence analysis

# Test units here.

def test(): 
    book_txts = read_chapters(shediao)
    result = {}
    context = []
    name = '黄蓉'
    for bt in book_txts:
        current_doc = nlp(bt)
        temp_bio = thou_life(name, current_doc)
        context += temp_bio
        #temp_result = word_cloud(name, current_doc, ['ADJ', 'NOUN', 'VERB'], ['amod', 'dobj', 'pobj'])
        #for tr in temp_result:
        #    if tr in result:
        #        result[tr] += temp_result[tr]
        #    else:
        #        result[tr] = temp_result[tr]
    with open('test_result.txt', 'r+') as file: 
        for bio in context:
            for info_set in bio:
                file.write(info_set)
    #print(name + " 's Top labels: {}".format(sorted(result.items(), key=lambda kv: kv[1], reverse=True)[:100]))

test()