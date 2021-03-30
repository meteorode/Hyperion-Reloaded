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

def sort_dict(dict): # return a sorted dictionary by values
    sorted_tuples = sorted(dict.items(), key=lambda item:item[1], reverse=True)   # return a sorted tuple by lambda function
    sorted_dict = {k: v for k, v in sorted_tuples}
    return sorted_dict

def count_names_with_attrs(names, words_related, docs, name_attrs={'dep': ['nsubj']}, attrs_related={'pos': ['VERB', 'ROOT', 'ADJ', 'VERB|ROOT']}, count_num=20, result_cap=1024):    
    # count names with {name_attrs} and words_related with {attr_related} in docs, return a {'some key': sent} dict
    pass

def summarization(doc): # Extract summary from a doc.
    pass

def behvaior_analysis(name, doc):   # Analysis character with {name} from doc
    pass

# Semantic Search based on sentence transformer

def semantic_search(corpus, queries, result_num): # # Find the closest {result_num} sentences of the corpus for each query sentence based on cosine similarity
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor = True)
    top_k = min(result_num, len(corpus))
    query_results = {}
    for query in queries:
        query_results[query] = []
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        # We use cosine-similarity and torch.topk to find the highest top_k scores
        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)
        tmp_result = {}
        for score, idx in zip(top_results[0], top_results[1]):
            tmp_result[corpus[idx]] = score.item()
        query_results[query].append(tmp_result)
    return query_results

models = texts.models

en_sentiment = models['SenticNet En']
cn_sentiment = models['SenticNet Cn']

def calc_polarity_value(word, lang='cn'):  # If sn/cn_sn(word) then score += polar_value, else find whether word is in semantics_union, finally += similarity * score
    if lang == 'cn':
        sentiment_words = cn_sentiment
        net = cn_sn
    else:
        sentiment_words = en_sentiment
        net = sn
    if word in sentiment_words:
        #print('Word found in default dict!')
        score = sentiment_words[word]
    else:
        try:    # try to find the word's polarity value first
            score = float(net.polarity_value(word))
            #print('Word polarity value found!')
        except:
            top_results = list(semantic_search(list(sentiment_words.keys()), [word], 5).values())[0][0]    # search top 5 words to determine positive or negative
            #print(word, top_results)
            positive = negative = 0
            for key in top_results:
                if sentiment_words[key] >= 0:
                    positive += 1
                elif sentiment_words[key] < 0:
                    negative += 1
            is_positive = (positive - negative > 0)
            for key in top_results:
                if (is_positive == True) and (sentiment_words[key] >= 0):
                    sema_score = top_results[key]
                    sema_word = key
                    break
                elif (is_positive == False) and (sentiment_words[key] < 0):
                    sema_score = top_results[key]
                    sema_word = key
                    break
            score = sentiment_words[sema_word] * sema_score
    return score

def slice_doc_by_sparkle(doc, ent_sparkles=['GPE', 'LOC', 'PRODUCT', 'WORK_OF_ART', 'EVENT', 'MONEY'], sent_dis=3):
    # Slice the doc by ent_type_ in spakles, with this algorithm:
    # if dis(Sn,Sn+1) < sent_dis, Sn.rear = (Sn, ..Sn+1)
    sents = list(doc.sents)
    slices = []
    sent_index_with_sparkles = []
    for sent in sents:
        sent_index = sents.index(sent)
        for token in sent:
            if token.ent_type_ in ent_sparkles and sent_index  not in sent_index_with_sparkles:
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
                    result[key] += word_similarity(c, sent, model_name='sts') / s_len
                    #if (word_similarity(c, sent) >= bar):
                    #    result[key] += word_similarity(c, sent, model_name='sts') / s_len
        sorted_result = sort_dict(result)
        topk_items = list(sorted_result.items())[:top_k]
        final_result = {}
        for item in topk_items:
            final_result[item[0]] = item[1]
        return final_result
