# English NLP test via spaCy

import spacy
from collections import Counter
#from tabulate import tabulate
from pathlib import Path

p = Path('.')   # current Path

# spacy.prefer_gpu()  # Using GPU to run programm

#nlp = spacy.load("zh_core_web_lg")
nlp = spacy.load('en_core_web_trf') # spacy 3.0 stable model.

person_names = ["Alice"]

alice_txts = list(p.glob('fairy_tales/Alice in Wonderland/*.txt'))
alice_txts.sort()

def read_chapters(book):
    text = ''
    for chapter in book:
        with open(chapter, 'r') as file:
            text += file.read()
    return text

def token_is_subject_with_action(token):
    nsubj = token.dep_ == 'nsubj'
    head_verb = token.head.pos_ == 'VERB'
    person = token.ent_type_ == 'PERSON'
    in_person = token.text in person_names
    return nsubj and head_verb and person and in_person

def token_is_object(token):
    dobj = token.dep_ == 'dobj'
    return dobj

def check_action(token):
    pass

def check_battle_words(token):  # 暂时将含有「杀」、「伤」、「胜」、「败」的动词统计进来
    is_verb = token.pos_ == 'VERB'
    has_keywords = ('杀' in token.text) or ('伤' in token.text) or ('胜' in token.text) or ('败' in token.text)
    return is_verb and has_keywords

def count_ents(doc):
    entities = {}
    for ent in doc.ents:
        if ent.text in entities:
            entities[ent.text] += 1
        else:
            entities[ent.text] = 1
    print("top entities{}".format(sorted(entities.items(), key=lambda kv: kv[1], reverse=True)[:30]))

def walk_through(name, doc):    # Timeline for {name} wandering in {doc}
    assert doc.has_annotation("SENT_START")
    bio = []
    for sent in doc.sents:
        if name in sent.text:
            bio.append(sent)    # Span collection
    return bio

def calc_similarity(doc1, doc2):
    return doc1.similarity(doc2)

def time_line(person, doc):
    assert doc.has_annotation("SENT_START")
    person_track = []
    for sent in doc.sents:
        has_person = False
        for token in sent:
            if token.text == person:    # Find person here.
                has_person = True
        if has_person:
            person_track.append(sent)
                
    return person_track

def abstract(sent):
    abs = ' '
    for token in sent:
        if token.dep_ == 'nsubj':
            abs += token.text + ' '
        elif token.dep_ == 'ROOT':
            abs += token.text  + ' '
        elif token.dep_ == 'dobj':
            abs += token.text + ' '
    return abs

def count_labels(doc):
    labels = {}
    for ent in doc.ents:
        if ent.label_ in labels:
            labels[ent.label_] += 1
        else:
            labels[ent.label_] = 1
    print(sorted(labels.items(), key=lambda kv: kv[1], reverse=True))

def life_of_yours(name, doc):
    events = []
    lifeline = time_line(name, doc)
    for spark in lifeline:
        events.append(abstract(spark))
    return events

# Labels Counting results in 「射雕英雄传」
# [('PERSON', 19555), ('CARDINAL', 7091), ('GPE', 1758), ('DATE', 1655), ('LOC', 665), ('ORG', 585), ('WORK_OF_ART', 416), 
# ('TIME', 362), ('ORDINAL', 341), ('FAC', 334), ('NORP', 216), ('EVENT', 87), ('PERCENT', 78), ('QUANTITY', 62), ('PRODUCT', 43), ('LANGUAGE', 30), ('MONEY', 19), ('LAW', 1)]

def check_ents(doc, ent_type):
    events = {}
    assert doc.has_annotation("SENT_START")
    for sent in doc.sents:
        for ent in sent.ents:
            if ent.label_ == ent_type:
                events[ent.text] = [ent.root.tag_, sent.text]
    return events

def whether_talk(words): # check a list of [token] containing a buzz word or not.
    #buzz_words = ['言', '说', '道', '谈', '论']
    buzz_words = ["say"]
    is_talking = False
    for token in words:
        for bw in buzz_words:
            if (bw in token.text) and (token.pos_ == 'VERB'):
                is_talking = True
    return is_talking

def dixit(name, doc): # Etymology Borrowed from Latin ipse dīxit (“he himself said it”), calque of Ancient Greek αὐτὸς ἔφα (autòs épha). 
                      # Originally used by the followers of Pythagoreanism, who claimed this or that proposition to be uttered by Pythagoras himself.
    bio = walk_through(name, doc)
    dixit_words = []
    for spark in bio:
        for word in spark:
            words_related = word.children # Word's children.
            if (name in spark.text) and (whether_talk(words_related) == True): # Check format like {name} [said]
                dixit_words.append(spark)
    return dixit_words

def test():
    alice = read_chapters(alice_txts)
    alice_doc = nlp(alice)
    # alice_words = dixit('Alice', alice_doc)
    alice_timeline = life_of_yours("Alice", alice_doc)
    for at in alice_timeline:
        print(at)

test()
