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

def find_sents_with_specs(doc, spec_names):
    sents = {}
    doc_slice = []
    for token in doc:
        if token.ent_type_ in spec_names:
            if token.sent not in doc_slice:
                doc_slice.append(token.sent)
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

# Words' Sentic Value/Similarity calculation
def word_trasformer(word):
    result = 0
    moodtags_weight = 0.3
    semantics_weight = 0.1
    try:
        polarirty_value = cn_sn.polarity_value(word)
        sentics_value = list(cn_sn.sentics(word).values())
        oldtags = cn_sn.moodtags(word)
        moodtags = []
        for ot in oldtags:
            moodtags.append(ot.lstrip('#'))
        semantics = cn_sn.semantics(word)
        moodtag_modifier = # Tired, to be continue
        for mt in moodtags:
            polarirty_value += moodtags_weight * cn_sn.polarity_value(mt)
    except:
        return result

def simple_sentic_similarity(sentic1, sentic2):
    s1_values = list(sentic1.values())
    s2_values = list(sentic2.values())
    dis = math.dist(s1_values, s2_values) / 4
    return (1 - dis)

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
    result = simple_sentic_similarity(sentic1, sentic2)
    return result

def word_cloud(name, docs, pos_types, dep_types): # Calc polarity_value with token's related words.
    cloud = {}
    hourglass = {'pleasantness': 0, 'attention': 0, 'sensitivity': 0, 'aptitude': 0}
    total = 1
    moodtags = {}
    for doc in docs:
        for token in doc:
            if name in token.text:
                total += 1
                for leaf in token.sent:
                    #print (leaf.text, calc_distance(token, leaf))
                    if leaf.pos_ in pos_types or leaf.dep_ in dep_types:
                        if leaf.text in cloud:
                            dis = calc_distance(token, leaf)
                            try:
                                mt = cn_sn.moodtags(leaf.text)
                                pv = cn_sn.polarity_value(leaf.text)
                                sentics = cn_sn.sentics(leaf.text)
                                nmt = []
                                for m in mt:
                                    nmt.append(m.lstrip('#'))
                                for nm in nmt:
                                    dict_modify(moodtags, nm, dis, dis)
                                cloud[leaf.text] += dis * pv
                                for s in sentics:
                                    hourglass[s] += dis * sentics[s]
                            except:
                                cloud[leaf.text] += dis * 0.01 # If no senticnet data, multipled by a minor 
                        else:
                            dis = calc_distance(token, leaf)
                            try:
                                mt = cn_sn.moodtags(leaf.text)
                                pv = cn_sn.polarity_value(leaf.text)
                                sentics = cn_sn.sentics(leaf.text)
                                nmt = []
                                for m in mt:
                                    nmt.append(m.lstrip('#'))
                                for nm in nmt:
                                    dict_modify(moodtags, nm, dis, dis)
                                cloud[leaf.text] += dis * pv
                                for s in sentics:
                                    hourglass[s] += dis * sentics[s]
                            except:
                                cloud[leaf.text] = dis * 0.01 # If no senticnet data, multipled by a minor
    return [cloud, hourglass, total, moodtags]

# Part I: Sentiment Analysis to determine A char's 「心」traits

Openness = ['想象力', '审美', '情感', '创造力', '开放', '哲学', '价值']
Conscientiousness = ['能力', '公正', '逻辑', '责任', '成就', '自律', '谨慎', '克制']
Extraversion = ['热情', '社交', '果断', '活跃', '冒险', '乐观']
Agreeableness = ['信任', '奉献', '直率', '服从', '谦虚', '移情']
Neuroticism = ['焦虑', '敌对', '神经质', '自我', '冲动', '脆弱']    # big five model words list

def calc_persona_score(word, polarity_value, sentics, moodtags, semantics, wordsets):
    # polarity_value, sentics, moodtags, semantics came form SenticNet and wordsets defined above
    polarity_weight = 3
    moodtags_weight = 0.3
    semantics_weight = 0.1
    total_num = len(wordsets) * (1 + polarity_weight + len(moodtags) * moodtags_weight + len(semantics) * semantics_weight) 
    raw_score = 0
    for w in wordsets:
        word_sentics = cn_sn.sentics(w)
        word_polarity = cn_sn.polarity_value(w)
        rp_sim = (1 - (polarity_value - word_polarity)/2)
        raw_score += rp_sim * polarity_weight
        raw_score += simple_sentic_similarity(sentics, word_sentics)
        for mt in moodtags:
            mt_sentics = cn_sn.sentics(mt)
            raw_score += simple_sentic_similarity(mt_sentics, word_sentics) * moodtags_weight
        for sm in semantics:
            sm_sentics = cn_sn.sentics(sm)
            raw_score += simple_sentic_similarity(sm_sentics, word_sentics) * semantics_weight
    score = raw_score / total_num
    return score

def to_big_five(word): # Word would be assigned [sentics, moodtags, semanticwords] etc attr, then parse this attr to 
    # big five personal traits
    try:
        sentics = cn_sn.sentics(word)
        oldtags = cn_sn.moodtags(word)
        moodtags = []
        for ot in oldtags:
            moodtags.append(ot.lstrip('#'))
    except:
        sentics = {'pleasantness': 0, 'attention': 0, 'sensitivity': 0, 'aptitude': 0}
        moodtags = []
    #_TO_BE_CONTINUE

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

def mood_analysis(book_name, docs, names):
    mood_with_names = {}
    for name in names:
        mood_result = {'pleasantness': 0, 'attention': 0, 'sensitivity': 0, 'aptitude': 0}
        wc_with_name = word_cloud(name, docs, ['ADJ', 'NOUN', 'VERB'], ['amod', 'dobj', 'pobj'])
        name_with_moodtags = wc_with_name[3]
        total_sent = wc_with_name[2]
        for nwm in name_with_moodtags:
            nwm_hourglass = cn_sn.sentics(nwm)
            for nh in nwm_hourglass:
                nwm_hourglass[nh] *= name_with_moodtags[nwm]
            for mr in mood_result:
                mood_result[mr] += nwm_hourglass[mr]
        for mr in mood_result:
            mood_result[mr] = mood_result[mr] / total_sent
        mood_with_names[name] = mood_result
    with open('%s_mood_analysis_result.txt' %(book_name), 'w+') as file:
        file.write('Name Pleasantness Attention Sensitivity Aptitude\n')
        for name in names:
            file.write("%s " %(name))
            for key in list(mood_with_names[name]):
                file.write('%.4f ' %(mood_with_names[name][key]) + ' ')
            file.write('\n')   

# Part II: Events and Choices slicing

def dixit(name, docs): # Etymology Borrowed from Latin ipse dīxit (“he himself said it”), calque of Ancient Greek αὐτὸς ἔφα (autòs épha). 
                      # Originally used by the followers of Pythagoreanism, who claimed this or that proposition to be uttered by Pythagoras himself.
    dixit_words = {}
    return dixit_words

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
    #txts = read_chapters(shediao)
    #docs = []
    #for txt in txts:
    #    docs.append(nlp(txt))
    #names = list(count_big_names(jinyong_names, docs, 20))
    #mood_analysis('shediao', docs, names)
    word = '脆弱'
    polarity_value = -0.531
    sentics = {'pleasantness': 0, 'attention': 0, 'sensitivity': 0.771, 'aptitude': -0.822}
    moodtags = ['生气', '恶心']
    semantics = ['弱', '容易被破解', '易碎', '微妙', '身体虚弱']
    result = calc_persona_score(word, polarity_value, sentics, moodtags, semantics, Openness)
    result2 = calc_persona_score(word, polarity_value, sentics, moodtags, semantics, Extraversion)
    result3 = calc_persona_score(word, polarity_value, sentics, moodtags, semantics, Neuroticism)
    print(result, result2, result3)

test()
