# persona.py

# Initilization

import spacy
from collections import Counter
#from tabulate import tabulate
from pathlib import Path
import time
from senticnet.senticnet import SenticNet
from senticnet.babelsenticnet import BabelSenticNet
import inspect
import math

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

# Methods to get var name as string.

def retrieve_name(var):
        """
        Gets the name of var. Does it from the out most frame inner-wards.
        :param var: variable to get name from.
        :return: string
        """
        for fi in reversed(inspect.stack()):
            names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
            if len(names) > 0:
                return names[0]

# Basic Statistics methods

def count_big_names(names, docs, count_num): # Count most common names from docs(not raw texts)
    name_freq = {}
    for doc in docs:
        for name in names:
            for token in doc:
                if name in token.text:
                    if name in name_freq:
                        name_freq[name] += 1
                    else:
                        name_freq[name] = 1
    name_tuple = sorted(name_freq.items(), key=lambda kv: kv[1], reverse=True)
    if len(name_tuple) >= count_num:
        return dict(name_tuple[:count_num])
    else:
        return dict(name_tuple)

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

def word_cloud(name, docs, pos_types, dep_types): # Calc polarity_value with token's related words.
    cloud = {}
    hourglass = {'pleasantness': 0, 'attention': 0, 'sensitivity': 0, 'aptitude': 0}
    total = 1
    for doc in docs:
        for token in doc:
            if name in token.text:
                total += 1
                for leaf in token.sent:
                    #print (leaf.text, calc_distance(token, leaf))
                    if leaf.pos_ in pos_types or leaf.dep_ in dep_types:
                        if leaf.text in cloud:
                            try:
                                cloud[leaf.text] += calc_distance(token, leaf) * cn_sn.polarity_value(leaf.text)
                                sentics = cn_sn.sentics(leaf.text)
                                for s in sentics:
                                    hourglass[s] += calc_distance(token, leaf) * sentics[s]
                            except:
                                cloud[leaf.text] += calc_distance(token, leaf) * 0.01 # If no senticnet data, multipled by a minor 
                        else:
                            try:
                                cloud[leaf.text] = calc_distance(token, leaf) * cn_sn.polarity_value(leaf.text)
                                sentics = cn_sn.sentics(leaf.text)
                                for s in sentics:
                                    hourglass[s] += calc_distance(token, leaf) * sentics[s]
                            except:
                                cloud[leaf.text] = calc_distance(token, leaf) * 0.01 # If no senticnet data, multipled by a minor
    return [cloud, hourglass, total]

# Part I: Sentiment Analysis to determine A char's 「心」traits

def translate(cn_words):    # _TO_BE_UPDATED_
    en_words = cn_words
    return en_words

def hourglass_analysis(book_name, docs, names):   # docs shoule be the nlp parsing result of read_chapters(book)
    name_en = translate(names)
    hourglass_with_names = {}
    for name in names:
        result = {'pleasantness': 0, 'attention': 0, 'sensitivity': 0, 'aptitude': 0}
        wc_with_names = word_cloud(name, docs, ['ADJ', 'NOUN', 'VERB'], ['amod', 'dobj', 'pobj'])
        result = wc_with_names[1]
        sent_count = wc_with_names[2]
        for r in result:
            result[r] = result[r] / sent_count
        hourglass_with_names[name] = result
    with open('%s_char_emotion.txt' %(book_name), 'w+') as file:
        file.write('Name Pleasantness Attention Sensitivity Aptitude\n')
        for name in names:
            file.write("%s " %(name))
            for key in list(hourglass_with_names[name]):
                file.write('%.4f ' %(hourglass_with_names[name][key]) + ' ')
            file.write('\n')

# Words' Sentic similarity calculation
def calc_sentic_similarity(word1, word2):   # Using sn.sentics to calc
    default = {'pleasantness': 0.000001, 'attention': 0.000001, 'sensitivity': 0.000001, 'aptitude': 0.000001}
    try:
        sentic1 = cn_sn.sentics(word1)
    except:
        try:
            sentic1 = sn.sentics(word1)
            for key in sentic1:
                sentic1[key] = float(sentic1[key])
        except:
            sentic1 = default
    try:
        sentic2 = cn_sn.sentics(word2)
    except:
        try:
            sentic2 = sn.sentics(word2)
            for key in sentic2:
                sentic2[key] = float(sentic2[key])
        except:
            sentic2 = default
    s1_values = list(sentic1.values()) 
    s2_values = list(sentic2.values())  # en sentic form would be {'introspection': '0.897', 'temper': '0', 'attitude': '0', 'sensitivity': '0.762'}, so needed update.
    quotients = []
    for i in range(len(s1_values)):
        quotients.append(abs(s1_values[i]/(s2_values[i]+0.00001)))
    norm = math.fsum(quotients) / len(quotients) 
    for value in s2_values:
        value *= norm
    dis = math.dist(s1_values, s2_values) / 4
    return (1 - dis)

# Part II: Skill analysis

# Part III: Body measurement traits analysis

# Part IV: Intelligence analysis

# Test units here.

def test(): 
    #with open('%s_result.txt' %(name_en), 'w+') as file:
    #    file.write('%s 的关联词如下：\n' %(name))
    #    for item in new_result:
    #        try:
    #            file.write(item[0] + ' ' + cn_sn.polarity_label(item[0]) + ' %.2f ' %(item[1]))
    #        except:
    #            file.write(item[0] + ' none ' + ' %.2f ' %(item[1]))
    #        if new_result.index(item) % 6 == 5:
    #            file.write('\n')
    txts = read_chapters(tianlong)
    docs = []
    for txt in txts:
        docs.append(nlp(txt))
    names = list(count_big_names(jinyong_names, docs, 20))
    hourglass_analysis("tianlong", docs, names)

test()
