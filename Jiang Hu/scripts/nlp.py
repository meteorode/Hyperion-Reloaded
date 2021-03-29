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

def count_names_with_attrs(names, words_related, docs, name_attrs={'dep': ['nsubj']}, attrs_related={'pos': ['VERB', 'ROOT', 'ADJ', 'VERB|ROOT']}, count_num=20, result_cap=1024):    
    # count names with {name_attrs} and words_related with {attr_related} in docs, return a {'some key': sent} dict
    pass

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
