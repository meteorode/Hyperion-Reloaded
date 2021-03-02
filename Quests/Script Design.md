# Script Design

## *Net like

以[JiangHu L]() 的结构为例（以及很多类似的设计），需要解决的问题其实是：

{Word 1} + {Sent 1} + ... + {Word N} + {Sent M} = ?

## Wuxia Theme

如果给定如下条件：

1.  游戏背景、世界观和人物属性限定为类似于[江湖 II](https://github.com/wagangmiao/JiangHu-L/tree/master/j2me%20version)的传统武侠
2.  战斗方式限定为基于格子的战棋形式（便于扩展到其他背景）
3.  游戏里的角色交流方式主要类似[太阁立志传](https://baike.baidu.com/item/%E5%A4%AA%E9%98%81%E7%AB%8B%E5%BF%97%E4%BC%A0/1898)或[侠客游](https://baike.baidu.com/item/%E4%BE%A0%E5%AE%A2%E6%B8%B8/6045048)，基于大地图-城市-地点-人物四层结构

应该如何设计脚本结构？

### A Sample

