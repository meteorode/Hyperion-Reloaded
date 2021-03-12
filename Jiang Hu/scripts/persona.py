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
import numpy as np

spacy.prefer_gpu()  # Using GPU to run programm

#nlp = spacy.load("zh_core_web_lg")
nlp = spacy.load('zh_core_web_trf') # spacy 3.0 stable model.
#en_nlp = spacy.load('en_core_web_trf')

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

def dict_modify(dic, key, value_modifier=1, value_base=1):  # A general method to modify dict(d) with key
    if key in dic:
        dic[key] += value_modifier
    else:
        dic[key] = value_base

# Basic Statistics methods

def count_big_names(names, docs, count_num): # Count most common names from docs(not raw texts)
    name_freq = {}
    for doc in docs:
        for name in names:
            for token in doc:
                if name in token.text:
                    dict_modify(name_freq, name)
    name_tuple = sorted(name_freq.items(), key=lambda kv: kv[1], reverse=True)
    if len(name_tuple) >= count_num:
        return dict(name_tuple[:count_num])
    else:
        return dict(name_tuple)

def count_attrs(doc, attr_type):
    entities = {}
    if attr_type == 'ENT':
        for ent in doc.ents:
            dict_modify(entities, ent.text)
    elif attr_type == 'LABEL':
        for ent in doc.ents:
            dict_modify(entities, ent.label_)
    #print("top entities{}".format(sorted(entities.items(), key=lambda kv: kv[1], reverse=True)[:30]))
    return entities

def sent_clustering(sent, doc, neighbor_num):   # return {neighbor_num} sents before and after in doc
    assert(sent in doc.sents) == True
    sents = list(doc.sents)
    sent_index = sents.index(sent)
    lower = max(0, sent_index-neighbor_num)
    upper = min(len(sents)-1, sent_index+neighbor_num)
    result = ''
    for i in range(lower, upper):
        result += sents[i].text
    return result

def find_sents_with_specs(docs, spec_names):
    sents = {}
    doc_slice = {}
    for doc in docs:
        for token in doc:
            if token.ent_type_ in spec_names:
                token_slice = sent_clustering(token.sent, doc, 5)
                if token.sent.text not in doc_slice:
                    doc_slice[token.sent.text] = token_slice
                token_description = token.text + ': ' + token.ent_type_
                dict_modify(sents, token.sent.text, token_description, token_description)
    return [sents, doc_slice]

def count_words_with_specs(doc, spec_type, spec_name):
    result = {}
    if spec_type == 'LABEL':
        for ent in doc.ents:
            if ent.label_ == spec_name:
                dict_modify(result, ent.text)
    
    return result

def unify_name(name, name_set): # set name to real name in (name_set)
    for n_s in name_set:
        if (name in n_s) or (n_s in name):
            name = n_s
    return name  

def calc_distance(token, leaf): # Assert leaf is another token in the same sentence with token, parse the tree then calc the distance of [token, leaf]
    dis = 0.3 # 3 produces everything!
    sent = token.sent
    assert (leaf in sent) == True
    if (leaf == token):
        dis = 0.0
    elif (leaf in token.children or leaf == token.head):
        dis = 1.0
    elif (leaf in token.subtree or leaf in token.head.children):
        dis = 0.7

    return dis

# Words' Sentic Value/Similarity calculation
def has_cn_sentic(word):   # Check whether in SenticNet
    try:
        who_cares = cn_sn.concept(word)
        return True
    except:
        return False

def has_en_sentic(word): # En version
    try:
        who_cares = sn.concept(word)
        return True
    except:
        return False

def word_vec_similarity(vec1, vec2):    # vec1 and vec2 should be a dict
    v1_values = list(vec1.values())
    v2_values = list(vec2.values())
    v1_len = len(v1_values)
    v2_len = len(v2_values)
    assert (v1_len == v2_len)
    max = [1.0] * v1_len
    min = [-1.0] * v1_len
    dis = math.dist(v1_values, v2_values) / math.dist(max, min)
    numerator = np.dot(v1_values, v2_values)
    v1_np = np.array(v1_values)
    v2_np = np.array(v2_values)
    denominator = math.sqrt(v1_np.dot(v1_np)) * math.sqrt(v2_np.dot(v2_np))
    if (denominator > 0):
        cos_similarity = numerator / denominator
    else:
        cos_similarity = 1.0
    similarity = (1 - dis + cos_similarity) / 2
    return similarity

def en_word_transformer(word): # like cn version, but sentic differs
    result = {'polarity_value': 0.0, 'introspection': 0.0, 'temper': 0.0, 'attitude': 0.0, 'sensitivity': 0.0}
    moodtags_weight = 0.3
    semantics_weight = 0.1
    if has_en_sentic(word):
        result['polarity_value'] = float(sn.polarity_value(word))
        w_sentics = sn.sentics(word)
        for key in w_sentics:
            result[key] += float(w_sentics[key])
        oldtags = sn.moodtags(word)
        moodtags = []
        for ot in oldtags:
            moodtags.append(ot.lstrip('#'))
        semantics = sn.semantics(word)
        for mt in moodtags:
            if (has_en_sentic(mt)):
                result['polarity_value'] += moodtags_weight * float(sn.polarity_value(mt))
                mt_sentics = sn.sentics(mt)
                for key in mt_sentics:
                    result[key] += float(mt_sentics[key]) * moodtags_weight
        for sm in semantics:
            if (has_en_sentic(sm)):
                result['polarity_value'] += semantics_weight * float(sn.polarity_value(sm))
                sm_sentics = sn.sentics(sm)
                for key in sm_sentics:
                    result[key] += float(sm_sentics[key]) * semantics_weight
        total_len = 1 + len(moodtags) * moodtags_weight + len(semantics) * semantics_weight
        for key in result:
            result[key] = result[key] / total_len
    return result

def word_transformer(word):  # return a {'polarity_value': x1, 'pleasantness': x2, ...}
    result = {'polarity_value': 0, 'pleasantness': 0, 'attention': 0, 'sensitivity': 0, 'aptitude': 0}
    moodtags_weight = 0.3
    semantics_weight = 0.1
    try:
        result['polarity_value'] = cn_sn.polarity_value(word)
        w_sentics = cn_sn.sentics(word)
        for key in w_sentics:
            result[key] += w_sentics[key]
        oldtags = cn_sn.moodtags(word)
        moodtags = []
        for ot in oldtags:
            moodtags.append(ot.lstrip('#'))
        semantics = cn_sn.semantics(word)
        for mt in moodtags:
            result['polarity_value'] += moodtags_weight * cn_sn.polarity_value(mt)
            mt_sentics = cn_sn.sentics(mt)
            for key in mt_sentics:
                result[key] += mt_sentics[key] * moodtags_weight
        for sm in semantics:
            result['polarity_value'] += semantics_weight * cn_sn.polarity_value(sm)
            sm_sentics = cn_sn.sentics(sm)
            for key in sm_sentics:
                result[key] += sm_sentics[key] * semantics_weight
        total_len = 1 + len(moodtags) * moodtags_weight + len(semantics) * semantics_weight
        for key in result:
            result[key] = result[key] / total_len
    except:
        return result
    return result

# Part I: Sentiment Analysis to determine A char's 「心」traits

Openness = ['想象力', '审美', '情感', '创造力', '开放', '哲学', '价值']
Conscientiousness = ['能力', '公正', '逻辑', '责任', '成就', '自律', '谨慎', '克制']
Extraversion = ['热情', '社交', '果断', '活跃', '冒险', '乐观']
Agreeableness = ['信任', '奉献', '直率', '服从', '谦虚', '移情']
Neuroticism = ['焦虑', '敌对', '神经质', '自我', '冲动', '脆弱']    # big five model words list

def calc_persona_score(word, wordsets): # transfer all words to a vec then calc similiarties of each other.
    sets_cap = len(wordsets)
    word_vec = word_transformer(word)
    score = 0
    for w in wordsets:
        w_vec = word_transformer(w)
        score += word_vec_similarity(word_vec, w_vec)
    score = score / sets_cap
    return score

def hourglass_light(word): # Raw Hourglass data calculating from word.
    hourglass = {'pleasantness': 0, 'attention': 0, 'sensitivity': 0, 'aptitude': 0}
    try:
        for key in hourglass:
            hourglass[key] = cn_sn.sentics(word)[key]
    except:
        return hourglass
    return hourglass

def ocean_horn(word): # Big Five Personalities Model, Aka O.C.E.A.N(Openness, Conscientiousness, Extraversion, 
    #Agreebleness, Neuroticism), calculating from word.
    big_five = {'Openness': 0, 'Consientiousness': 0, 'Extraversion': 0, 'Agreebleness': 0, 'Neuroticism': 0}   # Should drop all zero datas.
    try:
        big_five['Openness'] = calc_persona_score(word, Openness)
        big_five['Consientiousness'] = calc_persona_score(word, Conscientiousness)
        big_five['Extraversion'] = calc_persona_score(word, Extraversion)
        big_five['Agreebleness'] = calc_persona_score(word, Agreeableness)
        big_five['Neuroticism'] = calc_persona_score(word, Neuroticism)
    except:
        return big_five
    return big_five

def translate(cn_words):    # _TO_BE_UPDATED_
    en_words = cn_words
    return en_words

def word_cloud(word, docs, pos_types, dep_types): # return a dict like this {'related_word_1': weight, ...}
    cloud = {}
    for doc in docs:
        for token in doc:
            if word in token.text:
                for leaf in token.sent:
                    dis = calc_distance(token, leaf)
                    if leaf.pos_ in pos_types or leaf.dep_ in dep_types:
                        dict_modify(cloud, leaf.text, dis, dis)
    return cloud

def Est_Sularus_oth_Mithas(cloud, model_type): # return a normalized dict by model_type
    big_five = {'Openness': 0, 'Consientiousness': 0, 'Extraversion': 0, 'Agreebleness': 0, 'Neuroticism': 0}
    hourglass = {'pleasantness': 0, 'attention': 0, 'sensitivity': 0, 'aptitude': 0}
    total_weights = 0 
    if model_type == 'big_five':
        for key in cloud:   # assert key is a word and cloud is a dict
            temp_result = ocean_horn(key)
            if has_cn_sentic(key) == True:
                total_weights += cloud[key]
                for factor in big_five:
                    big_five[factor] += cloud[key] * temp_result[factor]
        total_weights = max(total_weights, 1)
        for factor in big_five:
            big_five[factor] = big_five[factor] / total_weights
        return big_five
    elif model_type == 'hourglass':
        for key in cloud:
            temp_result = hourglass_light(key)
            if has_cn_sentic(key) == True:
                for factor in hourglass:
                    total_weights += cloud[key]
                    hourglass[factor] += cloud[key] * temp_result[factor]
        total_weights = max(total_weights, 1)
        for factor in hourglass:
            hourglass[factor] = hourglass[factor] / total_weights
        return hourglass

def personality_traits_analysis(book_name, docs, names, model_type):   # docs shoule be the nlp 
    # parsing result of read_chapters(book)
    name_en = translate(names)
    persoanlity_traits_with_names = {}
    for name in names:
        wc_with_names = word_cloud(name, docs, ['ADJ', 'NOUN', 'VERB'], ['amod', 'dobj', 'pobj'])
        result = Est_Sularus_oth_Mithas(wc_with_names, model_type)
        persoanlity_traits_with_names[name] = result
    with open('%s_char_emotion.txt' %(book_name), 'w+') as file:
        if model_type == 'hourglass':
            file.write('Name Pleasantness Attention Sensitivity Aptitude\n')
        elif model_type == 'big_five':
            file.write('Name Openness Consientiousness Extraversion Agreebleness Neuroticism\n')
        for name in names:
            file.write("%s " %(name))
            for key in list(persoanlity_traits_with_names[name]):
                file.write('%.4f ' %(persoanlity_traits_with_names[name][key]) + ' ')
            file.write('\n')  

# Part II: Events and Choices slicing

def dixit(name, docs): # Etymology Borrowed from Latin ipse dīxit (“he himself said it”), calque of Ancient Greek αὐτὸς ἔφα (autòs épha). 
                      # Originally used by the followers of Pythagoreanism, who claimed this or that proposition to be uttered by Pythagoras himself.
    dixit_words = {}
    return dixit_words

# spaCy supports the following entity types:
# PERSON, NORP (nationalities, religious and political groups), FAC (buildings, airports etc.), 
# ORG (organizations), GPE (countries, cities etc.), LOC (mountain ranges, water bodies etc.), PRODUCT (products), 
# EVENT (event names), WORK_OF_ART (books, song titles), LAW (legal document titles), LANGUAGE (named languages), DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL and CARDINAL

def eye_tracking(doc, name_set):  # return series like 'PERSON'(supposed to be nsubj) MOVE TO 'LOC'/'GPE' or 'PERSON' ATTEND 'EVENT'
    scripts = []
    doc_milestone = find_sents_with_specs(doc, ['LOC', 'GPE', 'EVENT'])[1]
    for sent in doc_milestone:
        nsubj = '舞台中心'
        action = 'MOVETO'
        destination = '华山'
        for token in sent:
            if ((token.ent_type_ == 'PERSON' or token.pos_ == 'PROPN') and token.dep_ == 'nsubj'): 
                nsubj = unify_name(token.text, name_set)
            elif (token.ent_type_ == 'LOC' or token.ent_type_ == 'GPE'):
                destination = token.text
            elif (token.ent_type_ == 'EVENT'):
                action = 'ATTEND'
                destination = token.text
        script = nsubj + ' ' + action + ' ' + destination
        scripts.append(script)
    return scripts  

# Test units here.

def test(): 
    txts = read_chapters(shediao)
    docs = []
    for txt in txts:
        docs.append(nlp(txt))
    names = list(count_big_names(jinyong_names, docs, 20))
    personality_traits_analysis('shediao', docs, names, 'hourglass')

#test()