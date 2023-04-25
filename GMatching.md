## One-shot Relational Learning for Knowledge Graphs

jtwang 12.14

### Abstract

我们观察到呈长尾分布的关系在KG中更常见，一些新加入的关系也通常没有很多已知的三元组用来训练。所以，我们设置了一个challenging的场景，只有一个训练样本条件下，预测新的事实三元组。我们提出了一个one-shot关系学习框架，利用嵌入向量和one-hop图结构学习一个matching metric。

### Introduction

大规模知识图谱利用实体间的二元关系表示信息，通常是三元组的形式。这种结构化的知识对于许多下游任务，如QA和与以往是非常必要的。

尽管KG的规模很大，但众所周知，它们非常incomplete。为了自动地补全KG，许多研究工作构建了关系学习模型，可以从已知的三元组学习，从而推断出missing triple。这些方法explore三元组的统计信息或者path pattern，取得了considerable performance。

然而之前的模型使用的数据集只包含了common relations。但实际上，由于下面两点，desired KG补全模型可以利用少量样本进行预测，但现有的模型都是在有充分样本的前提下。

1. 大部分关系是long-tail，也就是说有很少的instances
2. KG是dynamic，新的关系不断加入。

我们提出了仅利用实体嵌入和local graph structure的模型。目标在于学习一个matching metric，在给定一个reference三元组的前提下，去发现更多相似的triples。metric model基于一个permutation-invariant网络，可以有效的编码实体的one-hop邻居，同时一个RNN allows multi-step matching。一旦训练完成，我们的模型就可以预测任意一个关系，while 现有的方法还需要fine-tuning to adapt new relations。在新构建的两个数据集上，我们的模型在one-shot link prediction任务中取得了consistent improvement。

##### Contributions

- 首次考虑long-tail关系，将这个问题定义为few-shot relational learning
- 提出一个one-shot学习框架
- 提出两个新的数据集for one-shot KG completion

### Related Work

##### Embedding Models for Relational Learning

RESCAL，TransE，DistMult，ComplEx，ConvE等。这些模型通常假设所有的关系和实体都有足够的训练样例，没有考虑稀疏性。一些模型利用text description处理unseen entities。相比于这些方法，我们的模型关注one-shot关系学习，处理long-tail和新加入的关系，没有使用外部信息。

##### Few-shot Learning

有两类。大多在vision和imitation learning领域。

- metric based，学习generalizable metrics和对应的matching functions。大多数方法采用deep siamese network提出的general matching  framework。
- meta-learner based，学习模型参数的optimization。

### 3 Background

### 4  Model

核心是一个similarity function，比较两个entity-pair（训练样本、测试样本）的相似度。要解决两个问题

1. entity pairs的表示
2. comparison function between two entity-pair representations

整体框架如下图所示，两部分

- Neighbor encoder，利用local graph structure增强实体表示
- Matching processor，multi-step matching between 2 entity-pairs，输出相似度得分

<img src="fig\GMatching\1.png" style="zoom:35%;" />

#### 4.1  Neighbor Encoder

纳入图结构信息到实体表示中，为maintain efficiency，我们只考虑实体的one-hop neighbor。
$$
f(N_e)=\sigma(\frac{1}{|N_e|}\sum_{(r_k,e_k)\in N_e}C_{r_k,e_k})\\
C_{r_k,e_k}=W_c(v_{r_k}\oplus v_{e_k})+b_c\\
v_{r_k}=\mathbf{emb}(r_k),v_{e_k}=\mathbf{emb}(e_k)
$$

#### 4.2  Matching Processor

对于reference entity pair $(h_0,t_0)$ 和query entity pair $(h_i,t_{ij})$ ，将四个实体代入$f(N_e)$，concatenate头尾实体表示。

利用LSTM-based recurrent processing block进行multi-step matching。K步匹配，得到$score_K$
$$
s=f(N_{h_0})\oplus f(N_{t_0}),q=f(N_{h_i})\oplus f(N_{t_{ij}})\\ 
h'_{k+1},c_{k+1}=LSTM(q,[h_k\oplus s,c_k])\\
h_{k+1}=h'_{k+1}+q\\
score_{k+1}=\frac{h_{k+1}\odot s}{\parallel h_{k+1}\parallel \parallel s\parallel}
$$

#### 4.3  Loss function and Training

对于一个query relation r，reference/training三元组为$(h_0,r,t_0)$，我们收集一组positive query triples$\{(h_i,r,t_i^+)\}$，通过pollute尾实体负采样得到$\{(h_i,r,t_i^-)\}$。使用hinge loss
$$
l_\theta=max(0,\gamma+score_\theta^--score_\theta^+)
$$

### 5  Experiments

#### 5.1  Datasets

构建了两个新的数据集NELL-One和Wiki-One。第一个数据集基于NELL，使用最近的dump，去除inverse relation，选择有50~500三元组的关系。第二个数据集基于Wikidata，规模更大。

