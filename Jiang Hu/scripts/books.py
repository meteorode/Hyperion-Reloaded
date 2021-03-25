# books.py

from pathlib import Path

p = Path('.')

# 飞雪连天射白鹿，笑书神侠倚碧鸳

feihu = list(p.glob('novels/jinyong/feihuwaizhuan/*.txt'))
feihu.sort()

xueshan = list(p.glob('novels/jinyong/xueshanfeihu/*.txt'))
xueshan.sort()

liancheng = list(p.glob('novels/jinyong/lianchengjue/*.txt'))
liancheng.sort()

tianlong = list(p.glob('novels/jinyong/tianlongbabu/*.txt'))
tianlong.sort()

shediao = list(p.glob('novels/jinyong/shediaoyingxiongzhuan/*.txt'))
shediao.sort()

baima = list(p.glob('novels/jinyong/baimaxiaoxifeng/*.txt'))
baima.sort()

luding = list(p.glob('novels/jinyong/ludingji/*.txt'))
luding.sort()

xiaoao = list(p.glob('novels/jinyong/xiaoaojianghu/*.txt'))
xiaoao.sort()

shujian = list(p.glob('novels/jinyong/shujianenchoulu/*.txt'))
shujian.sort()

shendiao = list(p.glob('novels/jinyong/shendiaoxialv/*.txt'))
shendiao.sort()

xiakexing = list(p.glob('novels/jinyong/xiakexing/*.txt'))
xiakexing.sort()

yitian = list(p.glob('novels/jinyong/yitiantulongji/*.txt'))
yitian.sort()

bixuejian = list(p.glob('novels/jinyong/bixuejian/*.txt'))
bixuejian.sort()

yuanyang = list(p.glob('novels/jinyong/yuanyangdao/*.txt'))
yuanyang.sort()

# 小李飞刀成绝响，人间不见楚留香

dashamo = ['novels/gulong/chuliuxiang-dashamo.txt']

# 杯雪
beixue = list(p.glob('books/beixue/*.txt'))
beixue.sort()

# EN books
alice = list(p.glob('fairy_tales/Alice in Wonderland/*.txt'))
alice.sort()

def read_names(category ='novels', author='jinyong'):
    names = []
    with open('%s/%s/person_list.txt' %(category, author), 'r') as file:
        lines = file.readlines()
        for line in lines:
            names.append(line.rstrip('\n'))
    return names

def read_chapters(book):
    txts = []
    for chapter in book:
        with open(chapter, 'r') as file:
            txts.append(file.read())
    return txts

jinyong_names = read_names()
gulong_names = read_names(author='gulong')
xiaoduan_names = read_names(author='xiaoduan')