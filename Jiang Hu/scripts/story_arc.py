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
    docs = []
    for txt in txts:
        docs.append(nlp(txt))
    doc_milestone = persona.find_sents_with_specs(docs, ['PERSON', 'LOC', 'GPE', 'EVENT'])[1]
    queries = list(propp_models.values())
    semantic_search(doc_milestone, queries, 10)

test()
