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

sn = SenticNet()
cn_sn = BabelSenticNet('cn')    # Use SenticNet to analysis.

p = Path('.')   # current Path

embedder = SentenceTransformer('./models/distiluse-base-multilingual-cased')    # Trying use SentenceTransformer to re-calculate word_similarity
sts_embedder = SentenceTransformer('./models/stsb-xlm-r-multilingual') # Optimized for Semantic Textual Similarity

def count_names_with_attrs(names, words_related, docs, name_attrs={'dep': ['nsubj']}, attrs_related={'pos': ['VERB', 'ROOT', 'ADJ', 'VERB|ROOT']}, count_num=20, result_cap=1024):    
    # count names with {name_attrs} and words_related with {attr_related} in docs, return a {'some key': sent} dict
    pass