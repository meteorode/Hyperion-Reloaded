# nlp.py

# Move nlp foundamental methods here.

import spacy
from pathlib import Path
from senticnet.senticnet import SenticNet
from senticnet.babelsenticnet import BabelSenticNet
import math
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import texts

spacy.prefer_gpu()  # Using GPU to run programm

nlp = spacy.load('zh_core_web_trf') # spacy 3.0 stable model.
#en_nlp = spacy.load('en_core_web_trf')

# spaCy tags, ner and dep_ definition:

# dep_: ROOT, acl, advcl:loc, advmod, advmod:dvp, advmod:loc, advmod:rcomp, amod, amod:ordmod, appos, aux:asp, aux:ba, aux:modal, aux:prtmod, auxpass, 
# case, cc, ccomp, compound:nn, compound:vc, conj, cop, dep, det, discourse, dobj, etc, mark, mark:clf, name, neg, nmod, nmod:assmod, nmod:poss, nmod:prep, 
# nmod:range, nmod:tmod, nmod:topic, nsubj, nsubj:xsubj, nsubjpass, nummod, parataxis:prnmod, punct, xcomp

# ner(ent_type_): CARDINAL, DATE, EVENT, FAC, GPE, LANGUAGE, LAW, LOC, MONEY, NORP, ORDINAL, ORG, PERCENT, PERSON, PRODUCT, QUANTITY, TIME, WORK_OF_ART

sn = SenticNet()
cn_sn = BabelSenticNet('cn')    # Use SenticNet to analysis.

p = Path('.')   # current Path

embedder = SentenceTransformer('./models/distiluse-base-multilingual-cased')    # Trying use SentenceTransformer to re-calculate word_similarity
sts_embedder = SentenceTransformer('./models/stsb-xlm-r-multilingual') # Optimized for Semantic Textual Similarity

def word_similarity(w1, w2, model_name='default'):    # Use model.encode() and pytorch_cos_sim() to calc
    if (model_name != 'sts'):
        emb1 = embedder.encode(w1)
        emb2 = embedder.encode(w2)
    else:
        emb1 = sts_embedder.encode(w1)
        emb2 = sts_embedder.encode(w2)
    
    cos_sim = util.pytorch_cos_sim(emb1, emb2).item()   # convert an 1 dimensional tensor to float
    return cos_sim

def count_names_with_attrs(names, words_related, docs, name_attrs={'dep': ['nsubj']}, attrs_related={'pos': ['VERB', 'ROOT', 'ADJ', 'VERB|ROOT']}, count_num=20, result_cap=1024):    
    # count names with {name_attrs} and words_related with {attr_related} in docs, return a {'some key': sent} dict
    pass

# Semantic Search based on sentence transformer

def semantic_search(corpus, queries, result_num): # # Find the closest {result_num} sentences of the corpus for each query sentence based on cosine similarity
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor = True)
    top_k = min(result_num, len(corpus))
    query_results = {}
    for query in queries:
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        # We use cosine-similarity and torch.topk to find the highest top_k scores
        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)
        query_results[query] = top_results
    return query_results

models = texts.models

en_sentiment = models['SenticNet En']
cn_sentiment = models['SenticNet Cn']

corpus = list(en_sentiment.keys())
queries = ['happy']
print (semantic_search(corpus, queries, 3))

def calc_polarity_value(word, lang='cn'):  # If sn/cn_sn(word) then score += polar_value, else find whether word is in semantics_union, finally += similarity * score
    semantics = {}
    if lang == 'en':
        for key in en_sentiment:
            sms = sn.semantics(key)
            semantics[key] = []
            for sm in sms:
                if sm not in semantics[key]:
                    semantics[key].append(sm)
        if word in en_sentiment:
            score = en_sentiment[word]

def sentiment_analysis(doc, lang='cn'):    # Calc a score using calc_polarity_value
    score = 0.0
    if lang == 'en':    # English docs.
        sentic_words = list(en_sentiment.keys())
        for token in doc:
            if token.text in sentic_words:
                score += en_sentiment[token.text]
    elif lang == 'cn':  # Chinese docs
        pass