# Test of stanford NLP.

import stanza

from collections import Counter
#from tabulate import tabulate
from pathlib import Path

p = Path('.')   # Current Path
nlp = stanza.Pipeline('zh')

# 飞雪连天射白鹿，笑书神侠倚碧鸳

feihu = list(p.glob('../novels/jinyong/feihuwaizhuan/*.txt'))
feihu.sort()

xueshan = list(p.glob('../novels/jinyong/xueshanfeihu/*.txt'))
xueshan.sort()

liancheng = list(p.glob('../novels/jinyong/lianchengjue/*.txt'))
liancheng.sort()

tianlong = list(p.glob('../novels/jinyong/tianlongbabu/*.txt'))
tianlong.sort()

shediao = list(p.glob('../novels/jinyong/shediaoyingxiongzhuan/*.txt'))
shediao.sort()

baima = list(p.glob('../novels/jinyong/baimaxiaoxifeng/*.txt'))
baima.sort()

luding = list(p.glob('../novels/jinyong/ludingji/*.txt'))
luding.sort()

xiaoao = list(p.glob('../novels/jinyong/xiaoaojianghu/*.txt'))
xiaoao.sort()

shujian = list(p.glob('../novels/jinyong/shujianenchoulu/*.txt'))
shujian.sort()

shendiao = list(p.glob('../novels/jinyong/shendiaoxialv/*.txt'))
shendiao.sort()

xiakexing = list(p.glob('../novels/jinyong/xiakexing/*.txt'))
xiakexing.sort()

yitian = list(p.glob('../novels/jinyong/yitiantulongji/*.txt'))
yitian.sort()

bixuejian = list(p.glob('../novels/jinyong/bixuejian/*.txt'))
bixuejian.sort()

yuanyang = list(p.glob('../novels/jinyong/yuanyangdao/*.txt'))
yuanyang.sort()

person_names = []
with open('../novels/jinyong/person_list.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        person_names.append(line.rstrip('\n'))

def read_chapters(book):
    text = ''
    for chapter in book:
        with open(chapter, 'r') as file:
            text += file.read()
    return text

#shendiao_text = read_chapters(shendiao)
shediao_text = read_chapters(shediao)
#tianlong_text =read_chapters(tianlong)

#shendiao_doc = nlp(shendiao_text)
shediao_doc = nlp(shediao_text)
#ianlong_doc = nlp(tianlong_text)
