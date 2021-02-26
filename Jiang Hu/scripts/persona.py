# persona.py

# Initilization

import spacy
from collections import Counter
#from tabulate import tabulate
from pathlib import Path

p = Path('.')   # current Path

# 杯雪
beixue = list(p.glob('books/beixue/*.txt'))
beixue.sort()

# spacy.prefer_gpu()  # Using GPU to run programm

#nlp = spacy.load("zh_core_web_lg")
nlp = spacy.load('zh_core_web_trf') # spacy 3.0 stable model.

def read_chapters(book):
    text = ''
    for chapter in book:
        with open(chapter, 'r') as file:
            text += file.read()
    return text

# Part I: Sentiment Analysis to determine A char's 「心」traits

# Part II: Skill analysis

# Part III: Body measurement traits analysis

# Part IV: Intelligence analysis

# Test units here.

def test(): 
    cup_snow = read_chapters(beixue)
    cup_snow_doc = nlp(cup_snow)
    print('finished!')

test()