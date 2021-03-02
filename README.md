# Hyperion Reloaded

## Intro

“Hyperion"(海伯利安)是一个在规定主题（即关于虚拟世界的通用知识）和人物属性（即可以量化的参与游戏机制的数据）以及脚本模版（即结构化的数据形式）的前提下自动代替DM的工具

## 目录结构

1.  Cards() is a data structure used to:   
    -   create pdf/jpeg files for print
    -   video game version of board/card games
    -   make other video games
2.  JiangHu is a Wuxia theme game trying to used Hyperion engine
    -   See [Persona](Jiang%20%Hu/Persona.py) for details.
3.  Map Tools 将形式化（甚至更进一步的，自然语言化）的地图描述解析成一个M*N的tiles map，每个tile上可以放置一些事先规定好的Obj，包括但不限于：
    *   Heroes
    *   Friends
    *   Treasues
    *   NPCs
    *   Enemies
    *   Events
    *   Resources
4.  Mechanics and Theme are still under construction
5.  NLP is an experiment tried to
    -  挑出若干人物，看能否给出时间线
        *   分析人物相关的「形容词」
    -  试图寻找分析「战斗结果」的语言模型
    -  设计script 模版，看能否自动往里面填入数据
    -  统计token.ent, 看能否获取有用的位置信息
    -  Count Events.
6.  Quests is designed to generate scripts/quests in games.