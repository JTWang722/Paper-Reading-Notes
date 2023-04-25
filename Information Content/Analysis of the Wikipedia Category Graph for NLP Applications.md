## Analysis of the Wikipedia Category Graph for NLP Applications

jtwang	2022.1.6[^1]

[^1]: [] 中是我的注释

> *论文: Zesch T, Gurevych I. Analysis of the Wikipedia category graph for NLP applications[C]//Proceedings of the Second Workshop on TextGraphs: Graph-Based Algorithms for Natural Language Processing. 2007: 1-8.*

[总结: 分析了 Wikipedia category graph 的性质]



### Abstract

在本文我们讨论 Wikipedia 中的两个图: (1) the article graph, (2) the category graph.

我们用图理论分析了 category graph, 证明它是 scale-free, small world graph, 跟其他 lexical semantic networks 一样. 证明方法是将定义在 WordNet 上的 semantic relatedness algorithm 迁移到 Wikipedia category graph 上. 



### 1  Introduction

- **Wikipeida Category Graph (WCG)**

我们证明了 WCG (collaboratively constructed) 和其他 well-known lexical semantic networks (e.g. WordNet constructed by experts) 具有一些相同的性质, 因此也可以作为 NLP 应用的资源.

**Article graph** 	已经有很多研究, 这里不再讨论. 每篇 article 是一个结点, article 间的引用是边

**Category graph**	维基百科中的 category 是以 taxonomy-like 结构组织的. 每个 category 可以有任意数量的 subcategories, 它们之间通常是 hyponymy [上下位] / meronymy [部分] 关系. 例如, 一个类 *vehicle* 有子类 *aricraft*, *watercraft*. 所以, WCG 和 WordNet, GermaNet 等词汇网络很相似. 



### 2  Graph-theoretic Analysis of WCG

- **graph based semantic relatedness measures**

将有向的 WCG 看做无向图, 因为 category 间的关系是可逆的. 使用 a snapshot of the German Wikipedia from 2006.5.15, 只考虑它的最大连通分量, 包含 99.8% 的结点.

graph parameters:

- $G:=<V,E>$
- degree $k$
- average degree $\bar k$
- degree distribution $P(k)\approx k^{-\gamma}$: 任意一个结点, 它的度是 k 的概率
- path $p_{i, j}$
- path length $l(p_{i,j})$
- shortest path length $L_{i,j}=\min{l(p_{i,j})}$
- average shortest path length $\bar{L}$
- diameter $D=\max{L_{i,j}}$
- cluster coefficient of node i $C_i$ : 聚类系数, 在社交网络中, 用来衡量 how many of my friends are friends themselves
- cluster coefficient for the whole graph $C=\bar{C_i}$  全连通图聚类系数是1

<img src="E:\jt's notes\论文阅读笔记\fig\1.png" style="zoom:40%;" />

上图是分析结果, 我们可以推断出

- 表1中的图都是 small world graphs. small $\bar{L}$ & large $C$
- 都是 scale-free graphs. 度分布遵循 power law. 原因可能是这些图都是按 preferential attachment 增长的. [preferential attachment???]
- WordNet 和 WCG 参数很相似, 所以推断 WordNet 上的算法可以迁移到 WCG



### 3  Graph Based Semantic Relatedness Measures

- semantic similarity (SS). defined via synonymy (automobile - car) [同义] and hypernymy (vehicle - car) [上下位] relations
- semantic relatedness (SR). defined to cover any kind of lexical or functional association. 不相似的两个词语义上也可以相关, 如 (night - dark) 或者反义词 (high - low)

##### 3.1  Wordnet Based Measures

$$
dist_{PL}=l(n_1,n_2)\\
sim_{LC}(n_1,n_2)=-log\frac{l(n_1,n_2)}{2\times depth}
$$

