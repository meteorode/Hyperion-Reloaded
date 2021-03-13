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
import persona
from sentence_transformers import SentenceTransformer, util
import torch

# spaCy init

spacy.prefer_gpu()  # Using GPU to run programm

nlp = spacy.load('zh_core_web_trf') # spacy 3.0 stable model.
#en_nlp = spacy.load('en_core_web_trf')

sn = SenticNet()
cn_sn = BabelSenticNet('cn')    # Use SenticNet to analysis.

p = Path('.')   # current Path

# Fundamental modules

def sort_dict(dict): # return a sorted dictionary by values
    sorted_tuples = sorted(dict.items(), key=lambda item:item[1], reverse=True)   # return a sorted tuple by lambda function
    sorted_dict = {k: v for k, v in sorted_tuples}
    return sorted_dict

embedder = SentenceTransformer('./models/distiluse-base-multilingual-cased')

def word_similarity(w1, w2):    # Use model.encode() and pytorch_cos_sim() to calc
    emb1 = embedder.encode(w1)
    emb2 = embedder.encode(w2)
    
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

# Propp models
propp_models = {'Absentation': 'Someone goes missing', 'Interdiction': 'Hero is warned', 'Violation': 'Violation of interdiction', 'Reconnaissance': 'Villain seeks something', 
                'Delivery': 'The villain gains information', 'Trickery': 'Villain attempts to deceive victim', 'Complicity': 'Unwitting helping of the enemy',
                'Villainy and lack': 'The need is identified', 'Mediation': 'Hero discovers the lack', 'Counteraction': 'Hero chooses positive action', 'Departure': 'Hero leave on mission',
                'Testing': 'Hero is challenged to prove heroic qualities', 'Reaction': 'Hero responds to test', 'Acquisition': 'Hero gains magical item', 'Guidance': 'Hero reaches destination',
                'Struggle': 'Hero and villain do battle', 'Branding': 'Hero is branded', 'Victory': 'Villain is defeated', 'Resolution': 'Initial misfortune or lack is resolved',
                'Return': 'Hero sets out for home', 'Pursuit': 'Hero is chased', 'Rescue': 'pursuit ends', 'Arrival': 'Hero arrives unrecognized', 'Claim': 'False hero makes unfounded claims',
                'Task': 'Difficult task proposed to the hero', 'Solution': 'Task is resolved', 'Recognition': 'Hero is recognised', 'Exposure': 'False hero is exposed',
                'Transfiguration': 'Hero is given a new appearance', 'Punishment': 'Villain is punished', 'Wedding': 'Hero marries and ascends the throne'}

def read_model_details(filename):   # read detaied description from file
    details = {}
    with open(filename, 'r') as file:
        for line in file:
            if line != '':
                dict_pairs = line.split('|')
                assert (len(dict_pairs) == 2) == True
                key = dict_pairs[0]
                value = dict_pairs[1]
                details[key] = value
    return details

# ridiculousJiangHu models

ridiculousJiangHu_roles = {'侠客': '与反派敌对', '反派': '与侠客敌对', }

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

def action_classify(word, bar=0.6):   # Suppose a word similarity bar to judge
    sims = {}
    for action in JiangHuActions:
        sims[action] = word_similarity(word, action)
    sorted_sims = sort_dict(sims)
    if (list(sorted_sims.values())[0] >= bar):
        return list(sorted_sims.keys())[0]
    else:
        return 'none'
        
def sent_type_judge(sent):	# judge a sent obj's type
    sent_type = ''
    nsubj = ''
    dobj = ''
    for token in sent:
        if token.pos_ == 'VERB' and action_classify(token.text) == 'say':
            sent_type = 'say'
            found_speaker = False
            for c in token.children:
                if c.dep_ == 'nsubj' and c.pos_ == 'NOUN':
                    nsubj = c.text
                    found_speaker = True
            if (found_speaker == False):
                nsubj = '作者'
            break
        elif token.ent_type_ in ['GPE', 'LOC']:
            sent_type = 'move'
            dobj = token.text
            break
        elif token.ent_type_ in ['PRODUCT', 'MONEY', 'WORK_OF_ART']:
            sent_type = 'gain'
            dobj = token.text
            break
        elif token.ent_type_ in ['EVENT']:
            sent_type = 'attend'
            dobj = token.text
            break
        elif token.ent_type_ == 'PERSON' and token.dep_ == 'nsubj':
            action_related = action_classify(token.head.text)
            if (action_related in ['talk', 'kill', 'fight', 'defeat']):
                sent_type = action_related
            nsubj = token.text
            break
    return [sent_type, nsubj, dobj]

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

def script_auto_complete(sent, sent_type_lists):    # sent_type_lists should be [sent_type, nsubj, dobj] calculated by judger.
    assert (len(sent_type_lists) == 3) == True
    sent_type = sent_type_lists[0]
    nsubj = sent_type_lists[1]
    dobj = sent_type_lists[2]
    script = ''
    action_table = {'move': ' MOVE TO: ', 'gain': ' GAIN: ', 'attend': ' ATTEND: ', 'talk': ' TALK TO: ', 
    'defeat': ' DEFEAT: ', 'say': ' SAY: ', 'fight': ' FIGHT WITH: ', 'kill': ' KILL: '}
    if sent_type in ['move', 'gain', 'attend']:
        for token in sent:
            if token.dep_ == 'nsubj' and token.pos_ == 'NOUN':
                nsubj = token.text
        if (dobj != ''):
            script = nsubj + action_table[sent_type] + dobj
            print(script)
    elif sent_type in ['talk', 'defeat', 'kill', 'fight']:
        for token in sent:
            if token.dep_ == 'dobj' and token.pos_ == 'NOUN':
                dobj = token.text
        if (dobj != ''):
            script = nsubj + action_table[sent_type] + dobj
            print(script)
    elif sent_type == 'say':
        dixit = '' # Etymology Borrowed from Latin ipse dīxit (“he himself said it”), calque of Ancient Greek αὐτὸς ἔφα (autòs épha). 
            # Originally used by the followers of Pythagoreanism, who claimed this or that proposition to be uttered by Pythagoras himself.
        dixit = trim_conversation(sent.text)
        if dixit != '':
            script = nsubj + ' SAY: ' + dixit
            print(script)
    return script

def script_extractor(doc): # extract scripts like information from docs
    scripts_list = []
    script =  ''
    sents = doc.sents # We'll use sentence for basic units
    for sent in sents:
        sent_type_lists = sent_type_judge(sent)
        script = script_auto_complete(sent, sent_type_lists)
        if script != '':
            scripts_list.append(script)
    return scripts_list

# Semantic Search based on sentence transformer

def semantic_search(corpus, queries, result_num): # # Find the closest {result_num} sentences of the corpus for each query sentence based on cosine similarity
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor = True)
    top_k = min(result_num, len(corpus))
    for query in queries:
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        # We use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        print("\n\n======================\n\n")
        print("Query:", query)
        print("\nTop %d most similar sentences in corpus:" %(top_k))

        for score, idx in zip(top_results[0], top_results[1]):
            print(corpus[idx], "(Score: {:.4f})".format(score))

# Test Unit
def test():
    txts = persona.read_chapters(persona.shediao)
    print('===Finish Reading===\n')
    book_name = 'shediao'
    all_cmds = []
    for txt in txts:
        doc = nlp(txt)
        print('==spaCy NLP done!==\n')
        cmds = script_extractor(doc)
        print('==Chapter %d parsed.==\n' %(txts.index(txt)) )
        all_cmds.append(cmds)
    with open('%s_timeline.txt' %(book_name), 'w+') as file:
        for cmds in all_cmds:
            for cmd in cmds:
                file.write(cmd + '\n')
            print('==Some part write finished\n')
    #docs.append(nlp(txt))
    #doc_milestone = list(persona.find_sents_with_specs(docs, ['PERSON', 'LOC', 'GPE', 'EVENT'])[1].values())
    #queries = list(propp_models.values())
    #complex_queries = list(read_model_details('./propp.txt').values())
    #semantic_search(doc_milestone, complex_queries, 10)

test()
