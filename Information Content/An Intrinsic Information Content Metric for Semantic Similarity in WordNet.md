## An Intrinsic Information Content Metric for Semantic Similarity in WordNet

jtwang	2022.1.6[^1]

[^1]: [] 中是我的注释

> Seco N, Veale T, Hayes J. An intrinsic information content metric for semantic similarity in WordNet[C]//Ecai. 2004, 16: 1089.

[总结: 提出一种利用下位词数量计算 IC 的方法, 这样只用到了 taxonomy 结构, 而不再需要外部语料库提供的统计信息]

#### Abstract

- Information Content (IC), 用来衡量两个 terms 或者词义的相似性
- 传统方法计算 IC: 本体的 hierarchical structure + statistics in text derived from a large corpus
- 我们提出了一个 wholly intrinsic measure of IC, 只依赖 hierarchical structure



#### Introduction

- semantic similarity (SS) -- Information Theoretic formulas
- 我们提出的 metric of IC, 不需要外部资源提供的 statistics



#### Information Theoretic Approaches

$$
ic_{res}(c)=-logp(c) \tag 1
$$

- c: a concept

- p(c): 在给定语料中 c 出现的概率
- 含义: 一个概念出现的概率越高, 它表达的信息就越少

$$
sim_{res}(c_1,c_2)=max_{c\in S(c_1,c_2)}ic_{res}(c) \tag 2
$$

- S(c1, c2) 是包含 c1 and c2 的概念的集合
- SS = ic(MSCA), Most Specific Common Abstraction

$$
sim_{lin}(c_1,c_2)=\frac{2\times sim_{res}(c_1,c_2)}{ic_{res}(c_1)+ic_{res}(c_2)} \tag 3
$$

$$
dist_{jcn}(c_1,c_2)=(ic_{res}(c_1)+ic_{res}(c_2))-2\times sim_{res}(c_1,c_2) \tag 4
$$



#### Information Content in WordNet

$$
ic_{wn}(c)=1-\frac{log(hypo(c)+1)}{log(max_{wn})}
$$

- hypo(c): c 的下位词的数量
- max_wn: constant, taxnomy 中最大的概念数
- 含义: 概念 c 的下位词越多, 传递的信息越少; 叶子信息量最多
