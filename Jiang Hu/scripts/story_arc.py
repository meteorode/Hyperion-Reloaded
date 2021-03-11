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
en_nlp = spacy.load('en_core_web_trf')

sn = SenticNet()
cn_sn = BabelSenticNet('cn')    # Use SenticNet to analysis.

p = Path('.')   # current Path

embedder = SentenceTransformer('./models/distiluse-base-multilingual-cased')

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

# def script_extractor(doc_slice, )

def word_similarity(w1, w2):    # Use model.encode() and pytorch_cos_sim() to calc
    emb1 = embedder.encode(w1)
    emb2 = embedder.encode(w2)
    
    cos_sim = util.pytorch_cos_sim(emb1, emb2).item()   # convert an 1 dimensional tensor to float
    return cos_sim

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
    #txts = persona.read_chapters(persona.shediao)
    #docs = []
    #for txt in txts:
    #    docs.append(nlp(txt))
    #doc_milestone = list(persona.find_sents_with_specs(docs, ['PERSON', 'LOC', 'GPE', 'EVENT'])[1].values())
    #queries = list(propp_models.values())
    #complex_queries = list(read_model_details('./propp.txt').values())
    #semantic_search(doc_milestone, complex_queries, 10)
    print(word_similarity('talk', 'talk'))
    print(word_similarity('talk', '说'))
    print(word_similarity('爱', '说'))

test()
