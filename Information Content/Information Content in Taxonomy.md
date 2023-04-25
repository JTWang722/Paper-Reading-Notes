## Information Content in Taxonomy

jtwang	2022/01/10



#### Information Content

$$
IC(c)=-\log p(c)
$$

- $p(c)$: the probability of encountering an instance of concept $c$
- $p(c)$越大, $IC(c)$越小, 也就是说, $c$ 出现的频率越高, 传递的信息量越少; 
- 在taxonomy中越往上的概念, $p(c)$ 越大, 所以 $IC(c)$ 越小



#### Semantic Similarity

使用 information content 度量相似度$^{[1]}$
$$
概念的相似度：sim(c_1,c_2)=\max_{c\in S(c_1,c_2)}[IC(c)]\\
单词的相似度：sim(w_1,w_2)=\max_{c_1,c_2}[sim(c_1,c_2)]
$$

- $S(c_1, c_2)$: the set of concepts that subsume both $c_1$ and $c_2$
- $s(w)$: the set of concepts in the taxonomy that are senses of word $w$
- $c_1$ ranges over $s(w_1)$, $c_2$ ranges over $s(w_2)$

此外, 还可以使用最短路径长度。



#### Intrinsic Information Content$^{[2]}$

$$
IIC(c)=1-\frac{log(hypo(c)+1)}{log(max_{wn})}
$$

- $hypo(c)$: the number of hyponyms of $c$
- $max_{wn}$: the maximum number of concepts that exist in the taxonomy
- $c$ 的下位词越多, 表达的信息量越少
- 这种方法不依赖语料, 因为不需要统计 $p(c)$



#### Analysis of WCG$^{[3]}$

articles & categories: 

- each article can link to an arbitrary number of categories, each category is a kind of semantic tag for that article.

把适用于 WordNet 的 semantic relatedness (SR) 度量迁移到 WCG 中

- WCG 中的结点不是 synset or single term, 而是 generalized concept or category; 而且 WCG 的覆盖范围不够大, 所以要做一些调整
- 为了计算两个 terms 间的 SR, 先找到相关的 articles, 再度量 articles 对应的 categories 间的 SR
- SR between terms -> article -> SR between categories



#### Encoding Category Correlations into bilingual topic modeling for CLTA$^{[4]}$

将 category correlation 加入 bilingual topic model 中

- **co-occurence correlation:** between category and their co-occurring words in text
- **structural correlation:** ancestor - descendant relationships in a taxonomy
  - based on IC: 结点的 IC 值决定了它的重要程度
  - based on path length: 从子节点传播到父节点

具体的做法

1. 将两种correlation转换为一个prior category distribution of each modeling object
2. 将所有的先验分布加入到 bilingual topic modeling







> Reference
>
> [1]	Resnik P. Using information content to evaluate semantic similarity in a taxonomy[J]. arXiv preprint cmp-lg/9511007, 1995.
>
> [2]	Seco N, Veale T, Hayes J. An intrinsic information content metric for semantic similarity in WordNet[C]//Ecai. 2004, 16: 1089.
>
> [3]	Zesch T, Gurevych I. Analysis of the Wikipedia category graph for NLP applications[C]//Proceedings of the Second Workshop on TextGraphs: Graph-Based Algorithms for Natural Language Processing. 2007: 1-8.
>
> [4]	Wu T, Zhang L, Qi G, et al. Encoding category correlations into bilingual topic modeling for cross-lingual taxonomy alignment[C]//International Semantic Web Conference. Springer, Cham, 2017: 728-744.