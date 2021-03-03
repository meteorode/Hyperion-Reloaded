# persona.py

# Initilization

import spacy
from collections import Counter
#from tabulate import tabulate
from pathlib import Path
import time
from senticnet.senticnet import SenticNet
from senticnet.babelsenticnet import BabelSenticNet

spacy.prefer_gpu()  # Using GPU to run programm

#nlp = spacy.load("zh_core_web_lg")
nlp = spacy.load('zh_core_web_trf') # spacy 3.0 stable model.
en_nlp = spacy.load('en_core_web_trf')

sn = SenticNet()
cn_sn = BabelSenticNet('cn')    # Use SenticNet to analysis.

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

# 小李飞刀成绝响，人间不见楚留香

dashamo = ['novels/gulong/chuliuxiang-dashamo.txt']

# 杯雪
beixue = list(p.glob('books/beixue/*.txt'))
beixue.sort()

jinyong_names = []
with open('novels/jinyong/person_list.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        jinyong_names.append(line.rstrip('\n'))

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

def calc_distance(token, leaf): # Assert leaf is another token in the same sentence with token, parse the tree then calc the distance of [token, leaf]
    dis = 0.2 # 3 produces everything!
    sent = token.sent
    assert (leaf in sent) == True
    if (leaf == token):
        dis = 0.0
    elif (leaf in token.children or leaf == token.head):
        dis = 3.0
    elif (leaf in token.subtree or leaf in token.head.children):
        dis = 1.0

    return dis

def word_cloud(name, doc, pos_types, dep_types): # Calc polarity_value with token's related words.
    cloud = {}
    for token in doc:
        if name in token.text:
            for leaf in token.sent:
                #print (leaf.text, calc_distance(token, leaf))
                if leaf.pos_ in pos_types or leaf.dep_ in dep_types:
                    if leaf.text in cloud:
                        try:
                            cloud[leaf.text] += calc_distance(token, leaf) * cn_sn.polarity_value(leaf.text)
                        except:
                            cloud[leaf.text] += calc_distance(token, leaf) * 0.001 # If no senticnet data, multipled by a minor 
                    else:
                        try:
                            cloud[leaf.text] = calc_distance(token, leaf) * cn_sn.polarity_value(leaf.text)
                        except:
                            cloud[leaf.text] = calc_distance(token, leaf) * 0.001 # If no senticnet data, multipled by a minor 
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
    #context = []
    name = '黄蓉'
    name_en = 'huangrong'
    for bt in book_txts:
        current_doc = nlp(bt)
        #temp_bio = thou_life(name, current_doc)
        #context += temp_bio
        temp_result = word_cloud(name, current_doc, ['ADJ', 'NOUN', 'VERB'], ['amod', 'dobj', 'pobj'])
        for tr in temp_result:
            if tr in result:
                result[tr] += temp_result[tr]
            else:
                result[tr] = temp_result[tr]
    new_result = sorted(result.items(), key=lambda kv: abs(kv[1]), reverse=True)
    with open('%s_result.txt' %(name_en), 'w+') as file:
        file.write('%s 的关联词如下：\n' %(name))
        for item in new_result:
            try:
                file.write(item[0] + ' ' + cn_sn.polarity_label(item[0]) + ' %.2f ' %(item[1]))
            except:
                file.write(item[0] + ' none ' + ' %.3f ' %(item[1]))
            if new_result.index(item) % 6 == 5:
                file.write('\n')
    #print(name + " 's Word Clouds: {}".format(new_result[:200]))

def test2():
    zh_doc = nlp("郭靖顺着各人眼光望去，只见黄沙蔽天之中，一队人马急驰而来，队中高高举起一根长杆，杆上挂着几丛白毛。")
    en_doc = en_nlp('Once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, "and what is the use of a book," thought Alice, "without pictures or conversations?" ')
    for token in zh_doc:
        try:
            print(token.text, token.pos_, token.tag_, token.dep_, cn_sn.concept(token.text))
        except:
            print(token.text, token.pos_, token.tag_, token.dep_, 'No SenticNet Concept')
    for token in en_doc:
        try:
            print(token.text, token.pos_, token.tag_, token.dep_, sn.concept(token.lemma_))
        except:
            print(token.text, token.pos_, token.tag_, token.dep_, 'No SenticNet Concept')

test()
#test2()