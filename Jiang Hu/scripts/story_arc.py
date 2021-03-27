# story_arc

# Template from Wuxia novels, Greek mythology, Shakespeare, etc
# Most of them would be defined like:
#   choices = [c1, c2, ..., cn]
#   if f(hero.persona) ==/!=/>/< sth:
#       hero.choice = c(h)
#   action_lists = execute(hero.choice)

import spacy
from pathlib import Path
from senticnet.senticnet import SenticNet
from senticnet.babelsenticnet import BabelSenticNet
import inspect
import math
import numpy as np
#import persona
import texts
from sentence_transformers import SentenceTransformer, util
import torch
#import nlp

# spaCy init

spacy.prefer_gpu()  # Using GPU to run programm

nlp = spacy.load('zh_core_web_trf') # spacy 3.0 stable model.
en_nlp = spacy.load('en_core_web_trf')

sn = SenticNet()
cn_sn = BabelSenticNet('cn')    # Use SenticNet to analysis.

p = Path('.')   # current Path

embedder = SentenceTransformer('./models/distiluse-base-multilingual-cased')
sts_embedder = SentenceTransformer('./models/stsb-xlm-r-multilingual') # Optimized for Semantic Textual Similarity

# Fundamental modules

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

def sort_dict(dict): # return a sorted dictionary by values
    sorted_tuples = sorted(dict.items(), key=lambda item:item[1], reverse=True)   # return a sorted tuple by lambda function
    sorted_dict = {k: v for k, v in sorted_tuples}
    return sorted_dict

def word_similarity(w1, w2, model_name='default'):    # Use model.encode() and pytorch_cos_sim() to calc
    if (model_name != 'sts'):
        emb1 = embedder.encode(w1)
        emb2 = embedder.encode(w2)
    else:
        emb1 = sts_embedder.encode(w1)
        emb2 = sts_embedder.encode(w2)
    
    cos_sim = util.pytorch_cos_sim(emb1, emb2).item()   # convert an 1 dimensional tensor to float
    return cos_sim

# Define heroes

class hero:
    def __init__(self, name ='吴茗', persona_ocean = {'Openness': 0.5, 'Consientiousness': 0.5, 'Extraversion': 0.5, 
    'Agreebleness': 0.5, 'Neuroticsm': 0.5}, persona_hourglass = {'pleasantness': 0, 'attention': 0, 
    'sensitivity': 0, 'aptitude': 0}):
        self.name = name
        self.persona_ocean = persona_ocean
        self.persona_hourglass = persona_hourglass

# ridiculousJiangHu and other models 

models = texts.models # read from json file.
propp_model = models['propp']

#persona.sent_clustering(sent, doc, neighbor_num=5)

def text_classification(contents, model_name='propp', bar=0.3, top_k=3, is_dualistic=False): # Classify a list of sent by model[model_name]
    try:
        classify_model = models[model_name]
    except:
        classify_model = models['propp']
    result = {}
    if (is_dualistic == False):
        for key in classify_model:
            result[key] = 0
            s_len = len(classify_model[key])
            for sent in classify_model[key]:
                for c in contents:
                    if (word_similarity(c, sent) >= bar):
                        result[key] += word_similarity(c, sent, model_name='sts') / s_len
        sorted_result = sort_dict(result)
        topk_items = list(sorted_result.items())[:top_k]
        final_result = {}
        for item in topk_items:
            final_result[item[0]] = item[1]
        return final_result

def trim_conversation(words):   # trim “” and ‘’
    thou_say = ''
    quota_marks = {'“': '”', '‘': '’'}
    for left in quota_marks:
        if left in words:
            temp_words = words.partition(left)[2]
            if quota_marks[left] in temp_words:
                thou_say = temp_words.rpartition(quota_marks[left])[0]
            else:
                thou_say = temp_words
            return thou_say
    return thou_say

def summarization(doc): # Extract summary from a doc.
    pass

def behvaior_analysis(name, doc):   # Analysis character with {name} from doc
    pass

# JiangHu II script abstract
# Conditions are clear, Actions would be like this:
# <nsubj>[PERSON] {VERB}S='TALK TO' <dobj>[PERSON] (<> for dep_, {} for pos_, S for Semantics and [] for ent_type_)
# <nsubj>[PERSON] {VERB}S='SAY'
# <nsubj>[PERSON] {VERB}S='GAIN' <dobj>[PRODUCT/MONEY/WORK_OF_ART]
# <nsubj>[PERSON] {VERB}S='FIGHT' <dobj>[PERSON]
# <nsubj>[PERSON] {VERB}S='BEAT DOWN' <dobj>[PERSON]
# <nsubj>[PERSON] {VERB}S='KILL' <dobj>[PERSON]
# <nsubj>[PERSON] {VERB}S='MOVE TO' <dobj>[LOC/GPE]
# <nsubj>[PERSON] {VERB}S='ATTEND' <dobj>[EVENT]
# __OTHER_PROPP_MODEL_LIKE_SCRIPTS_WILL_BE_ADDED_

JiangHuActions = ['talk', 'say', 'gain', 'fight', 'defeat', 'kill', 'move', 'attend']

def jh_action_classify(word, bar=0.6):   # Suppose a word similarity bar to judge
    sims = {}
    for action in JiangHuActions:
        sims[action] = word_similarity(word, action)
    sorted_sims = sort_dict(sims)
    if (list(sorted_sims.values())[0] >= bar):
        return list(sorted_sims.keys())[0]
    else:
        return 'none'

def slice_doc_by_sparkle(doc, sparkles=['GPE', 'LOC', 'PRODUCT', 'WORK_OF_ART', 'EVENT', 'MONEY'], sent_dis=3):
    # Slice the doc by ent_type_ in spakles, with this algorithm:
    # if dis(Sn,Sn+1) < sent_dis, Sn.rear = (Sn, ..Sn+1)
    sents = list(doc.sents)
    slices = []
    sent_index_with_sparkles = []
    for sent in sents:
        sent_index = sents.index(sent)
        for token in doc:
            if token.ent_type_ in sparkles:
                sent_index_with_sparkles.append(sent_index)
    sparkle_nums = len(sent_index_with_sparkles)
    if sparkle_nums == 0:   # No sparkle
        return []
    else:
        for i in range(sparkle_nums):
            if i == 0:
                front_dis = sent_index_with_sparkles[i]
                rear_dis = sent_index_with_sparkles[i+1] - sent_index_with_sparkles[i]
            elif i == sparkle_nums-1:
                front_dis = sent_index_with_sparkles[i] - sent_index_with_sparkles[i-1]
                rear_dis = len(sents) - sent_index_with_sparkles[i]
            else:
                front_dis = sent_index_with_sparkles[i] - sent_index_with_sparkles[i-1]
                rear_dis = sent_index_with_sparkles[i+1] - sent_index_with_sparkles[i]
            if (front_dis >= sent_dis):
                slice_start = sent_index_with_sparkles[i] - sent_dis
            else:
                slice_start = sent_index_with_sparkles[i-1]+1
            if (rear_dis >= sent_dis):
                slice_end = sent_index_with_sparkles[i] + sent_dis
            elif i < sparkle_nums-1:
                slice_end = sent_index_with_sparkles[i+1]-1
            else:
                slice_end = len(sents) - 1
            slices.append(sents[slice_start:slice_end])
        return slices

def script_extractor(doc, model_name='JiangHu', bar=0.25): # extract scripts-like information from doc
    scripts = []
    nsubj = dobj = script = ''
    sents = doc.sents   # We'll use sentence for basic units
    ent_table = {'GPE': ' MOVE TO: ', 'LOC': ' MOVE TO: ', 'PRODUCT': ' GAIN: ', 'MONEY': ' GAIN: ', 'WORK_OF_ART': ' GAIN: ', 'EVENT': ' ATTEND: '}
    action_table = {'TALK': ' TALK TO: ', 'DEFEAT': ' DEFEAT: ', 'FIGHT': ' FIGHT WITH: ', 'KILL': ' KILL: '}
    for sent in sents:
        sent_in_JiangHu = text_classification([sent.text], model_name='JiangHu Script')
        if max(list(sent_in_JiangHu.values())) >= bar: # meanigful sent?
            sent_type = list(sent_in_JiangHu.keys())[0] # suppose sent_type is determined by text classification.
            for token in sent:
                if token.pos_ == 'VERB' and sent_type == 'SAY':   # SAY
                    found_speaker = False
                    for child in token.children:
                        if child.dep_ == 'nsubj':
                            nsubj = child.text
                            found_speaker = True
                    if (found_speaker == False):
                        nsubj = '作者'
                    dixit = '' # Etymology Borrowed from Latin ipse dīxit (“he himself said it”), calque of Ancient Greek αὐτὸς ἔφα (autòs épha). 
                        # Originally used by the followers of Pythagoreanism, who claimed this or that proposition to be uttered by Pythagoras himself.
                    dixit = trim_conversation(sent.text)
                    script = nsubj + ' SAY: ' + dixit
                    print(script)
                    scripts.append(script)
                    break
                elif token.ent_type_ in ['GPE', 'LOC', 'PRODUCT', 'MONEY', 'WORK_OF_ART', 'EVENT'] and sent_type in ['MOVE', 'GAIN', 'ATTEND']: # MOVE TO/GAIN/ATTEND
                    ent = token.ent_type_
                    dobj = token.text
                    found_sb = False
                    for nbor in token.head.children:  # suppose the depency tree is (token: [GPE])(token.head: {VERB})(some child: <nsubj>)
                        if nbor.dep_ == 'nsubj':
                            nsubj = nbor.text
                            found_sb = True
                    if (found_sb == False):
                        nsubj = '宋兵乙'
                    script = nsubj + ent_table[ent] + dobj
                    print(script)
                    scripts.append(script)
                    break
                elif token.ent_type_ == 'PERSON' and token.dep_ == 'nsubj':
                    if (sent_type in ['TALK', 'KILL', 'FIGHT', 'DEFEAT']):
                        found_erdos = False
                        for nbor in token.head.children:
                            if nbor.dep_ == 'dobj':
                                dobj = nbor.text
                                found_erdos = True
                        if (found_erdos == False):
                            dobj = '路人甲'
                        script = nsubj + action_table[sent_type] + dobj
                        print(script)
                        scripts.append(script)
                        break
    return scripts

def write_script(book_name, book_prefix, slice_length, doc_type):  # Write scipts to files, slice the docs to increase performance
    txts = texts.read_chapters(book_name)
    print('===Finish Reading===\n')
    chapters_num = len(txts)
    if chapters_num >= slice_length:
        parts_num = int(chapters_num/slice_length) + 1
    else:
        parts_num = 1
    for i in range(parts_num):
        all_cmds = []
        for p in range(min(slice_length, chapters_num)):
            txt = txts[min(i*3+p, chapters_num-1)]
            if doc_type == 'cn':
                doc = nlp(txt)
            elif doc_type == 'en':
                doc = en_nlp(txt)
            print('===spaCy NLP done!===\n')
            cmds = script_extractor(doc)
            print('===Chapter %d parsed!===\n' %(txts.index(txt)+1))
            all_cmds.append(cmds)
        with open('%s_timeline_part%d.txt' %(book_prefix, i), 'w+') as file:
            for cmds in all_cmds:
                for cmd in cmds:
                    file.write(cmd + '\n')
                print('==File part%d write finished\n'%(i+1))

# Test Unit
def test():
    #txt = ['岂知杨康极是乖觉，只恐有变，对遗命一节绝口不提，直到在大会之中方始宣示。净衣派三老明知自己无份，也不失望，只消鲁有脚不任帮主，便遂心愿，又想杨康年轻，必可诱他就范。何况他衣着华丽，食求精美，决不会偏向污衣派。当下三人对望了一眼，各自点了点头。简长老道：“这位杨相公所持的，确是本帮圣物。众兄弟如有疑惑，请上前检视。”鲁有脚侧目斜睨杨康，心道：“凭你这小子也配作本帮帮主，统率天下各路丐帮？”伸手接过竹杖，见那杖碧绿晶莹，果是本帮帮主世代相传之物，心想，“必是洪帮主感念相救之德，是以传他']
    #print(text_classification(txt))    
    #write_script(texts.shediao, 'shediao', 3, 'cn')
    #docs.append(nlp(txt))
    #doc_milestone = list(persona.find_sents_with_specs(docs, ['PERSON', 'LOC', 'GPE', 'EVENT'])[1].values())
    #queries = list(propp_models.values())
    #complex_queries = list(read_model_details('./propp.txt').values())
    #semantic_search(doc_milestone, complex_queries, 10)
    test_txt = texts.read_chapters(texts.shediao)[0]
    test_doc = nlp(test_txt)
    slices = slice_doc_by_sparkle(test_doc)
    for s in slices:
        result = ''
        for sent in s:
            result += sent.text
        print(result)

test()