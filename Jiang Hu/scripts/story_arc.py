# story_arc

# Template from Wuxia novels, Greek mythology, Shakespeare, etc
# Most of them would be defined like:
#   choices = [c1, c2, ..., cn]
#   if f(hero.persona) ==/!=/>/< sth:
#       hero.choice = c(h)
#   action_lists = execute(hero.choice)

import spacy
from pathlib import Path
import inspect
import math
import texts
import nlp

# spaCy init

spacy.prefer_gpu()  # Using GPU to run programm

cn_nlp = spacy.load('zh_core_web_trf') # spacy 3.0 stable model.
en_nlp = spacy.load('en_core_web_trf')

p = Path('.')   # current Path

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

vonnegut_model = models['Vonnegut']
key_pos = ['ADJ', 'ADV', 'NOUN', 'VERB', 'PROPN', 'INTJ']

def emotional_arc_analysis(doc, lang='cn', keyattrs = key_pos):    # Analysis the doc using calc_polarity_value and the Vonnegut model.
    doc_slices = nlp.slice_doc_by_sparkle(doc)
    result = {}
    for ds in doc_slices:   
        if len(ds) > 4: # assert len(one_slice) >= 5 so that we could calc the shape.
            five_points = []
            shapes = []
            gap = int(len(ds)/5)
            ds_sent = ''
            for i in range(0, len(ds)-1, gap):
                selected_sent = ds[i]
                ds_sent += selected_sent.text
                score = 0.0
                for token in selected_sent:
                    if token.pos_ in keyattrs:
                        score += nlp.calc_polarity_value(token.text)
                five_points.append(score)
            for i in range(4):
                if five_points[i] < five_points[i+1]:
                    shapes.append(-1)
                else:
                    shapes.append(1)
            for key in vonnegut_model:
                if vonnegut_model[key] == shapes:
                    result[ds_sent] = key
    return result

wuxia_battle_model = models['wuxia_battle']
battle_words = []
for key in wuxia_battle_model:
    battle_words += wuxia_battle_model[key]

def is_battle_word(word, lang='cn', bar=0.8):    # Check whether a word is a battle word in model[wuxia_battle] 
    if (word in battle_words):
        return True
    elif (list(nlp.semantic_search(battle_words, [word], 1)[word][0].values())[0] >= bar):  # SO UGLY...
        return True
    return False

def check_dual_pairs(doc, lang='cn'): # return pairs with [PERSON], by check battle status and nsubj/dobj.
    pairs = {}
    for sent in list(doc.sents):
        nsubj = []
        dobj = []
        is_battle = False
        for token in sent:
            if token.ent_type_ == 'PERSON' and token.dep_ == 'nsubj':
                nsubj.append(token.text)
            elif token.ent_type_ == 'PERSON' and token.dep_ == 'dobj':
                dobj.append(token.text)
            if is_battle_word(token.text) == True:
                is_battle = True
        if (is_battle == True) and (len(nsubj) > 0) and (len(dobj) > 0):
            for ns in nsubj:
                if (ns not in pairs):
                    pairs[ns] = dobj
                else:
                    pairs[ns] += dobj
    return pairs

def battle_result_analysis(doc, lang='cn'): # return {p1} <score> {p2} <score}
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
        sims[action] = nlp.word_similarity(word, action)
    sorted_sims = nlp.sort_dict(sims)
    if (list(sorted_sims.values())[0] >= bar):
        return list(sorted_sims.keys())[0]
    else:
        return 'none'

def script_extractor(doc, model_name='JiangHu', bar=0.25): # extract scripts-like information from doc
    scripts = []
    nsubj = dobj = script = ''
    sents = doc.sents   # We'll use sentence for basic units
    ent_table = {'GPE': ' MOVE TO: ', 'LOC': ' MOVE TO: ', 'PRODUCT': ' GAIN: ', 'MONEY': ' GAIN: ', 'WORK_OF_ART': ' GAIN: ', 'EVENT': ' ATTEND: '}
    action_table = {'TALK': ' TALK TO: ', 'DEFEAT': ' DEFEAT: ', 'FIGHT': ' FIGHT WITH: ', 'KILL': ' KILL: '}
    for sent in sents:
        sent_in_JiangHu = nlp.text_classification([sent.text], model_name='JiangHu Script')
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
                doc = cn_nlp(txt)
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
    txts = texts.read_chapters(texts.shediao)
    print('===Finish Reading===')
    for txt in txts:
        doc = cn_nlp(txt)
        print('===Chapter %d spacy NLP done!==='%(txts.index(txt)+1))
        fighters = check_dual_pairs(doc)
        print(fighters)
        #ana = emotional_arc_analysis(doc)
        #for key in ana:
        #    print('===Sparkles===\n', key, '\nEmotional Shape: ', ana[key], '==========\n')

test()