# 论文阅读笔记
这里是我在阅读论文时的一些随手记. 


# LOG
- 【2021/10/16】  KG embedding综述里将模型划分为了Translational distance和Semantic matching，计划从语义匹配模型看起
- 【2021/10/21】  看了一遍RESCAL，DistMult，HolE，论文里好多数学公式，把我整懵了..还需要花时间仔细看
- 【2021/10/22】  老师说先看RESCAL和 DistMult, HolE 看不懂可以先放放, 常用的还有 ComplEx
- 【2021.10.23】  开始仔细看RESCAL，这篇文章比较早，也是语义匹配模型的开山之作(老大哥)。主要是Bilinear的概念之前没见过，张量分解不是很熟悉，参数更新的方法很数学，所以读起来有些晦涩
- 【2021.10.26】  RESCAL看了个七七八八，开始看DistMult。奇怪的是DistMult的作者并没有怎么提RESCAL，反而总是引用NTN和TransE
- 【2021.10.29】  给老师讲了RESCAL和DistMult，两个模型之间的递进关系没有搞明白，为什么可以替换成对角阵而效果也不差呢？
- 【2021.10.31】  看完了NTN，这篇提出了用张量表示关系(RESCAL中使用矩阵表示关系)，结合了神经网络的框架
- 【2021.12.15】  断更一个半月，期间看了ComplEx、KG2E、研究了OpenKE的源码、还有毕设相关的几篇关于不确定性知识图谱嵌入的文章。ComplEx使用复数表示实体和关系，可以表示asymmetric relation；KG2E使用高斯分布表示实体和关系，考虑到它们本身的(un)certainty。[要继续更新鸭！Be dilligent]
- 【2021.12.27】  看了TransE和TransH。TranE是第一个translation-based模型，简单却有效，思想是将实体建模成向量，关系建模成实体间的translation向量，也就是头实体+关系=尾实体；针对TransE不能处理reflexive（(h,r,t)和(t,r,h)都成立）/1-m（(h,r,t_i),i=1...m都成立）关系，TransH提出将头尾实体先映射到一个relation-specific的超平面上，在这个超平面上度量头实体+关系=尾实体，也就是说一个实体在不同的关系中有不同的表示。
- 【2022/01/07】  看了几篇 Information Content (IC) 相关的论文，与毕设相关
- 寒假期间搞定了毕设方案，没有怎么看论文。。。
- 【2022/02/26】  ConvE，一个基于卷积神经网络的KG嵌入模型
- 【2022/03/06】  ANALOGY，建模KG的analogy structure，双线性模型，将关系限制为normal matrix
