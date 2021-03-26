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
import texts
from sentence_transformers import SentenceTransformer, util
import torch
#import nlp

spacy.prefer_gpu()  # Using GPU to run programm

#nlp = spacy.load("zh_core_web_lg")
nlp = spacy.load('zh_core_web_trf') # spacy 3.0 stable model.
#en_nlp = spacy.load('en_core_web_trf')

sn = SenticNet()
cn_sn = BabelSenticNet('cn')    # Use SenticNet to analysis.

p = Path('.')   # current Path

#embedder = SentenceTransformer('./models/distiluse-base-multilingual-cased')    # Trying use SentenceTransformer to re-calculate word_similarity
embedder = SentenceTransformer('./models/stsb-xlm-r-multilingual') # Optimized for Semantic Textual Similarity

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

def bayesian_average(raw_average, dataset_len, C=666, m=0.333):  # Calc Bayesian average of a distribution [len, avg], with formula: ba = C*m + sum(dic.values())/ (C + len(dic))
    return (C * m + raw_average * dataset_len) / (C + dataset_len)

def count_big_names(names, docs, count_num, sent_cap=1024): # Count most common names from docs(not raw texts), return sents of them
    name_freq = {}
    sents = {}
    for name in names:
        sents[name] = []
    for doc in docs:
        for name in names:
            for token in doc:
                if name in token.text:
                    sents[name].append(token.sent.text)
                    dict_modify(name_freq, name)
    name_tuple = sorted(name_freq.items(), key=lambda kv: kv[1], reverse=True)
    for name in names:
        sents_with_name = sents[name][:min(sent_cap, len(sents[name]))]
        sents[name] = sents_with_name
    if len(name_tuple) >= count_num:
        big_names = dict(name_tuple[:count_num])
    else:
        big_names = dict(name_tuple)
    name_with_result = {}
    for name in big_names:
        name_with_result[name] = sents[name]
    return name_with_result # a dict like {name: [sent1, ... sent2]}

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

def word_similarity(word1, word2):  # Use model.encode() and pytorch_cos_sim() to calc
    emb1 = embedder.encode(word1)
    emb2 = embedder.encode(word2)
    
    cos_sim = util.pytorch_cos_sim(emb1, emb2).item()   # convert an 1 dimensional tensor to float
    return cos_sim

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

models = texts.models

def calc_sample_bias(model_name='ridiculousJiangHu'):
    model = models[model_name]
    all_sents = []
    data = {}
    for key in model:
        data[key] = 0
        for sent in model[key]:
            all_sents.append(sent)
    for key in model:
        for sent in model[key]:
            for other_sent in all_sents:
                data[key] += word_similarity(sent, other_sent) / len(all_sents)
    return data

rJ_data_bias = calc_sample_bias()   # ridiculous data bias
#wuxia_data_bias = calc_sample_bias(model_name='wuxia')

def init_model_samples():
    samples = {}
    for model_name in models:
        tmp_result = {}
        model = models[model_name]
        for key in model:
            tmp_result[key] = 0
        samples[model_name] = tmp_result
    return samples

def sentence_modelling(sent, model_name, is_dualistic=False, bar=0.67):  # Using sentece to compare instead of a single word, if is_dualistic == True, the model woul
    # have pros and cons.
    try:
        all_model_samples = init_model_samples()
        this_model = models[model_name]
        this_sample = all_model_samples[model_name]
    except:
        return {}
    if (is_dualistic == False):
        for key in this_model:
            s_len = len(this_model[key])
            sims = 0.0
            for s in this_model[key]:
                sims += word_similarity(sent, s) / s_len
                this_sample[key] = sims / calc_sample_bias(model_name)
    else:
        for key in this_model:
            pro_cons = list(this_model[key].items())
            assert (len(pro_cons) == 2) == True
            pros = pro_cons[0][1]   # list of pros
            cons = pro_cons[1][1]   # list of cons
            sims = 0.0
            p_len = len(pros)
            c_len = len(cons)
            for p in pros:
                sims += word_similarity(sent, p) / p_len
            for c in cons:
                sims -= word_similarity(sent, c) / c_len
            this_sample[key] = sims

    return this_sample

def general_modelling(word, model_type, bar=0.1): # General modelling using word_similarity()
    wuxia_hex = init_model_samples()['wuxia']
    wuxia_opposite = ['怯懦', '邪恶', '冷漠', '愚蠢', '卑鄙', '拘谨', '灵活', '坚强', '喧闹']
    if (model_type == 'wuxia'):
        for key in wuxia_hex:
            sims = word_similarity(key, word)
            index = list(wuxia_hex.keys()).index(key)
            opps = word_similarity(wuxia_opposite[index], word)
            meaning = sims - opps # [-1, 1] theoretically, (-0.33, 0.33) mostly
            if (meaning >= bar):    # Likely to be in hex.
                wuxia_hex[key] = sims
            elif (meaning <= (0-bar)):  # Likely to be the opposite
                wuxia_hex[key] = -opps/2
            else:   # In the midele
                wuxia_hex[key] = (meaning + bar)/ (2 * bar)
    return wuxia_hex

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

def word_cloud(words, docs, pos_types, dep_types, cap=1024): # return a nested dict like this {'word1': {'related_word_1': weight, ...}, ... }
    clouds = {}
    for word in words:
        clouds[word] = {}
    for doc in docs:
        for token in doc:
            for word in words:
                if word in token.text:
                    for leaf in token.sent:
                        dis = calc_distance(token, leaf)
                        if leaf.pos_ in pos_types or leaf.dep_ in dep_types:
                            dict_modify(clouds[word], leaf.text, dis, dis)
    for cloud in clouds:
        new_cap = max(int(len(clouds[cloud])/10), cap)
        cloud_tuple = sorted(clouds[cloud].items(), key=lambda kv: kv[1], reverse=True)
        new_cloud = dict(cloud_tuple[:min(len(cloud_tuple), new_cap)])
        clouds[cloud] = new_cloud
    return clouds

def Est_Sularus_oth_Mithas(cloud, model_type, sents=[], mining_type='cloud', is_dualistic=False): # return a normalized dict by model_type
    big_five = {'Openness': 0, 'Consientiousness': 0, 'Extraversion': 0, 'Agreebleness': 0, 'Neuroticism': 0}
    hourglass = {'pleasantness': 0, 'attention': 0, 'sensitivity': 0, 'aptitude': 0}
    sents_hex = init_model_samples()[model_type]
    total_weights = 0
    if mining_type == 'sents':  # Comparing with sents.
        sent_len = len(sents)
        for sent in sents:
            tmp_result = sentence_modelling(sent, model_type, is_dualistic)
            for key in tmp_result:
                sents_hex[key] += tmp_result[key] / sent_len
        return sents_hex
    elif mining_type == 'cloud':
        if model_type == 'big_five':
            for key in cloud:   # assert key is a word and cloud is a dict
                temp_result = ocean_horn(key)
                if has_cn_sentic(key) == True:
                    total_weights += cloud[key]  # Shoule be meaningful while = 1, means equitity of each word
                    for factor in big_five:
                        big_five[factor] += cloud[key] * temp_result[factor]
            for factor in big_five:
                big_five[factor] = big_five[factor] / total_weights
                big_five[factor] = bayesian_average(big_five[factor], total_weights, m=0.42)
            return big_five
        elif model_type == 'hourglass':
            for key in cloud:
                temp_result = hourglass_light(key)
                if has_cn_sentic(key) == True:
                    total_weights += cloud[key]
                    for factor in hourglass:
                        hourglass[factor] += cloud[key] * temp_result[factor]
            for factor in hourglass:
                hourglass[factor] = hourglass[factor] / total_weights
                hourglass[factor] = bayesian_average(hourglass[factor], total_weights, m=0.1)
            return hourglass

def personality_traits_analysis(book_name, docs, names, model_type, sents_dict={}, mining_type='cloud', is_dualistic=False):   # docs shoule be the nlp 
    # parsing result of read_chapters(book)
    name_en = translate(names)
    persoanlity_traits_with_names = {}
    wc_with_names = word_cloud(names, docs, ['ADJ', 'VERB', 'ROOT', 'VERB|ROOT'], ['amod', 'dobj', 'pobj'])
    print("===Names with Clouds Generated===")
    for name in names:
        if (mining_type != 'sents'):
            wc_with_name = wc_with_names[name]
            print('===%s Word Cloud Created==='%(name))
            result = Est_Sularus_oth_Mithas(wc_with_name, model_type)
            print('===%s Personal Traits Calculated=='%(name))
            print(result)
            persoanlity_traits_with_names[name] = result
        else:
            print('===%s Sents Loaded==='%(name))
            result = Est_Sularus_oth_Mithas({}, model_type, sents_dict[name], 'sents', is_dualistic)
            print('===%s Personal Traits Calculated=='%(name))
            print(result)
            persoanlity_traits_with_names[name] = result
    
    return persoanlity_traits_with_names

def write_parsing_result(book_name, result_as_dict, model_type):
    with open('%s_personal_traits.txt' %(book_name), 'w+') as file:
        if model_type == 'hourglass':
            file.write('Name Pleasantness Attention Sensitivity Aptitude\n')
        elif model_type == 'big_five':
            file.write('Name Openness Consientiousness Extraversion Agreebleness Neuroticism\n')
        elif model_type == 'wuxia':
            file.write('Name 勇敢 善良 深情 聪明 侠义 脆弱\n')
        elif model_type == 'ridiculousJiangHu':
            file.write('Name 侠客 路人 红颜 备胎 反派 喽啰 名门弟子 世外高人')
        for name in result_as_dict:
            file.write(name + ' ')
            pr = result_as_dict[name]
            for key in pr:
                file.write('%.6f ' %(pr[key]) + ' ')
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

def test2():
    big_five = models['big_five']
    for key in big_five:
        print(key, sentence_modelling(key, 'big_five', is_dualistic=True))

def test(): 
    txts = texts.read_chapters(texts.shediao)
    print('===Finished books reading!===')
    docs = []
    for txt in txts:
        docs.append(nlp(txt))
        print('===Chapter %d spacy NLP done!==='%(txts.index(txt)+1))
    #beixue_names = ['骆寒', '易敛', '荆三娘', '沈放', '袁老大', '萧如', '文翰林']
    names_with_sents = count_big_names(texts.jinyong_names, docs, 20)
    names = list(names_with_sents.keys())
    #names = ['杨过', '丘处机']
    #result1 = personality_traits_analysis('shendiao', docs, names, 'wuxia')
    result2 = personality_traits_analysis('shediao', docs, names, 'big_five', sents_dict=names_with_sents, mining_type='sents', is_dualistic=True)
    write_parsing_result('shediao', result2, 'big_five')

#test()
#test2()
