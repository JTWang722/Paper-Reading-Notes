## Using Information Content to Evaluate Semantic Similarity in a Taxonomy

jtwang	2022/1/7

> Resnik P. Using information content to evaluate semantic similarity in a taxonomy[J]. arXiv preprint cmp-lg/9511007, 1995.

[总结: 在 IS-A taxonomy 中, 提出用 IC 计算概念间的相似度.]



### Abastrct

基于 information content 的概念, 提出了一个 measure of semantic similarity in an IS-A taxonomy.



### 1	Introduction

- semantic relatedness 语义相关性. 举个例子, cars-gasoline / cars-bicycle, 前者更相关, 后者更相似
- semantic similarity 语义相似性. 是语义相关性的特殊情况, 相当于语义网中的 taxonomic (IS-A) links.

**Edge counting method**	在 taxonomy 中衡量语义相似性的一个 natural 方法是两个节点间的最短路径长度. 但是这种方法有缺点, 它要求节点间的距离是 uniform [因为路径长度认为每个相邻节点间的距离都是1].



### 2	Similarity and  Information Content

两个概念的相似性取决于它们共享信息的程度. 对于一个 IS-A taxonomy, 用一个 highly specific concept that subsumes the both 来表示.  Edge counting 方法是符合这个直觉的: 两个概念离得越远, 就需要往上找更抽象的概念作为 least upper bound. 以下图为例, (NICKEL, DIME) are both subsumed [上位词] by COIN; (NICKEL, CREDIT CARD) 最具体的超类是 MEDIUM OF EXCHANGE.

<img src=".\fig\IC\2.png" style="zoom:45%;" />

- p(c): probability of encountering an instance of concept c
  - taxonomy 往上, p(c) 单调递增. 即越抽象的概念, p(c) 越大: if c1 IS-A c2, then p(c2) >= p(c1).
- Information content: $-\log p(c)$. 
  - 越抽象的概念, 传递的信息量越少

$$
概念相似度: sim(c_1,c_2)=\max_{c\in S(c_1,c_2)}[-\log p(c)]\\
单词相似度: sim(w_1,w_2)=\max_{c_1,c_2}[sim(c_1,c_2)]
$$

- $S(c_1,c_2)$ 是 subsume c1 & c2 的概念的集合. 找到两个概念的最小上界, 在 taxonomy 中最低的上界
- c1 是能表达 w1 词义的 taxonomy 中的概念, c2 同理. 两个概念集中最相似的就是这两个单词的相似度.



### 3	Evaluation

##### Implementation

- 数据集使用 WordNet taxonomy of concepts[^1] represented by nouns

  [^1]: 这里的 concept 是 synset, 指 taxonomy 中的一个结点 

- 概念的频率 [p(c)] 由大型语料库 xxxxx 计算得到. 公式如下, words(c) 是被概念 c 包含的单词的集合, N 是单词总数
  $$
  freq(c) =\sum_{n\in words(c)}count(n)\\
  p(c)=\frac{freq(c)}{N}
  $$

##### Results

- 基准: human subject data provided by Miller and Charles, 有 30 pairs
- 作者重复了一遍他们的实验, 得到10个人的得分均值 r = 0.8848, 作为上界

- 下表是实验结果, 由于 30 个 pairs 中有单词不在 WordNet 中, 所以最后只有 28 pairs. Probability 方法是将 log p(c) 换为 p(c). 

<img src=".\fig\IC\3.png" style="zoom:45%;" />

##### Discussion

对于多义的词语, 这个方法就不好用了. horse 是 heroin 的俚语, 所以导致 tobacco 与 horse 的相似度很高,  甚至高于 (tabacco, alcohol / sugar),  这与常识不符. 改进一下计算公式, 在计算两个概念的相似度时, 考虑所有包含 c1 和 c2 的概念, 而不是只考虑最大值.
$$
sim(c_1,c_2)=\max_{c\in S(c_1,c_2)}[-\log p(c)]\rightarrow \sum_i\alpha(c_i)[-\log p(c)]
$$
