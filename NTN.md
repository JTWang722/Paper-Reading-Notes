

# Neural Tensor Network (NTN)

> 论文：*Reasoning with neural tensor networks for knowledge base completion*

jtwang	2021.11.3

#### Abstract

- 提出了neural tensor network，用于知识补全
- 实体用平均词向量来初始化
- 任务：predict additional true relations

#### Introduction

本体和知识库（KB）如WordNet、Yago对query expansion、coreference resolution、QA、IE都是极其有用的，但是现有的知识库往往是不完整的，也缺乏推理能力。

许多工作利用从large text corpora中获得的pattern或者classifiers来扩展KB，但不是所有的common knowledge都以text的形式表示。我们的目标是通过预测additional true facts进行KB的补全。

Contributions:

- 提出了NTN（neural tensor network），整合了许多神经网络模型，更有效的建模关系信息

- 用平均词向量表示实体，allowing the sharing of statistical strength[^ 1]

  [^1]: sharing? statistical strength?

- The incorporation of word vectors，无监督预训练词向量的引入使得所有模型都可以更准确地预测关系

#### Neural Models for Reasoning over Relations

这部分我们将介绍neural tensor network。如下图所示，每一个关系三元组用一个neural network表示，输入是两个实体。如果关系存在则输出的score很高。

<img src=".\fig\ntn\1.png" alt="NTN1" style="zoom:50%;" />

##### Neural Tensor Networks

- 任务：common sense reasoning（根据存在的关系推理出一些一定存在的facts，e.g.新物种猴子）、link prediction
- 目标：推断出 $(e_1, R, e_2)$ 是否存在以及存在的可能性有多大
- NTN-based function：相比于神经网络多了一个双线性
- d是实体向量维度，$e_1,e_2 \in \mathbb{R^d}$，$W_R\in \mathbb{R}^{d\times d\times k}$，$V_R\in \mathbb{R}^{k\times 2d}$，$u_R\in \mathbb{R}^{k}$，$b_R\in \mathbb{R}^{k}$
- 每种关系 R 对应一个张量 $W_R$，k是超参数

<img src=".\fig\ntn\3.png" alt="NTN3" style="zoom: 50%;" />

<img src=".\fig\ntn\2.png" alt="NTN2" style="zoom: 67%;" />

##### Related Models and Special Cases

用函数 $g$ 表示三元组的得分，得分越高，则这个三元组存在的可能性更大。

**Distance Model**   *[8] A. Bordes et al. Learning structured embeddings of knowledge bases. In AAAI, 2011.*

将两个实体通过关系映射矩阵映射到同一空间，比较二者的 $L_1$ 距离，$W_{R,1},W_{R,2}\in \mathbb{R}^{d \times d}$ 是关系 R 的参数。训练目标是：如果 $(e_1,R,e_2)$ 存在，则 g 会最小。

模型的主要问题是两个实体向量没有交互，被单独地映射到 common space。

<img src=".\fig\ntn\7.png" style="zoom: 50%;" />

**Single Layer Model**

通过神经网络的非线性函数进行两个实体的交互，$f=tanh$，$W_{R,1},W_{R,2}\in \mathbb{R}^{k \times d}$，$u_R\in \mathbb{R}^{k\times 1}$ 是关系 R 的参数。相比于优化代价，实体间的交互还不够强 weak interaction。这是 NTN 没有张量的特殊情况。

<img src=".\fig\ntn\8.png" style="zoom: 50%;" />

**Hadamard Model**  *[10] A. Bordes et al. Joint Learning of Words and Meaning Representations for Open-Text Semantic Parsing. AISTATS, 2012.*

通过多种矩阵乘法提供了更强的实体间交互，引入了 Hadamard products。将关系也建模成向量，用相同的方法对待实体和关系。$W_1, W_{rel,1},W_2,W_{rel,2}\in \mathbb{R}^{d\times d}\quad b_1,b_2\in \mathbb{R}^{d\times 1}$ 是共享的参数，只有 $e_R$ 是 relation specific。但是我们实验证明为每个关系提供单独的 matrix operator 效果更好，而且实体向量间的 Bilinear 交互更 desirable。

<img src=".\fig\ntn\9.png" style="zoom: 50%;" />

**Bilinear Model**  *[9] R. Jenatton, N. Le Roux, A. Bordes, and G. Obozinski. A latent factor model for highly multi-relational data. In NIPS, 2012.*  *[11] I. Sutskever, R. Salakhutdinov, and J. B. Tenenbaum. Modelling relational data using Bayesian clustered tensor factorization. In NIPS, 2009.*

通过 relation-specific bilinear form 加强实体间交互。$W_R\in \mathbb{R}^{d\times d}$ 是关系 R 的唯一参数。这种实体间的交互方式更简单有效。但是这种模型受限于参数规模，而且只能建模线性关系，不能 fit 更复杂的评分函数。这是 NTN 的一种特殊情况。相比于 Bilinear，NTN 表达能力更强，更适用于大型数据库。
$$
g(e_1,R,e_2)=e_1^TW_Re_2
$$

##### 训练目标函数 Training Objective and Derivatives

- $T^{(i)}=(e_1^{(i)},R^{(i)},e_2^{(i)})$
- $T_c^{(i)}=(e_1^{(i)},R^{(i)},e_c)$ 为corrupted triplet
- $\Omega = u,W,V,b,E$ 为全体参数
- N为训练集中三元组个数，C为corrupted三元组个数
- 目标函数：max-margin，正项和负项的最大margin为1，$L_2$正则项

<img src=".\fig\ntn\4.png" alt="NTN4" style="zoom:50%;" />

- 优化方法：梯度下降。对j-th slice求导，得到

<img src=".\fig\ntn\5.png" alt="image-20211030165629623" style="zoom:50%;" />

##### Entity Representations Revisited

以往的模型随机初始化实体向量，我们提出了两点改进：

1. 用词向量表示实体
2. 用预训练的向量初始化词向量

优点：allow the sharing of statistical strength between the words describing each entity

我们用 d 维向量表示每一个单词，用实体的平均词向量表示实体。e.g. 如果存在 (homo sapiens, type of, homind)，我们用 $v_{home}$ 和 $v_{sapiens}$ 表示实体 homo sapiens，即 $v_{homo sapiens}=0.5(v_{home}+v_{sapiens})$。这种关系可以扩展到 unseen 实体 homo erectus

我们也尝试了 RNN 进行组合，但是 WordNet 中60%实体只有一个单词，90%的实体包含≤2个单词，而且Freebase 中有很多人名，没有 compositional structure，所以 RNN 的效果相比于简单地平均没有明显提升。

我们使用 d=100 无监督的预训练词向量进行初始化。这种方法不能处理多义的单词，每个 word 只有一个向量。

### 4 Experiments

任务：common sense reasoning over know facts / link prediction in relationship networks。例如，如果一个人在伦敦出生，那么他的国际是英国；如果 German Shepard 是一种狗，那么它也是脊椎动物。

##### 4.1 数据集

WordNet，Freebase，如下表所示。

我们对测试集进行了filtering。对于 WordNet，如果训练集中存在 $(e_1,r,e_2)$，那么我们会去掉测试集中的 $(e_2,r,e_1)$ 与 $(e_1,r_i,e_2)$；对于 Freebase，我们只采样了 People domain 中的关系三元组，在测试集中去除了 6 种关系 (place of death, place of birth, location, parents, children, spouse)，因为这些很难预测。

WordNet 与 Freebase差别很大，前者 $e_1$ 和 $e_2$ 可以使任意向量；后者 $e_1$ 被限制为只能是人名，$e_2$ 只能从有限答案集中选择。例如，如果 $R=gender$，那么 $e_2$ 只能是 $male$ 或者 $female$。

<img src=".\fig\ntn\6.png" style="zoom:55%;" />

##### 4.2 Relation Triplets Classification

目标：预测测试数据中 $(e_2,R,e_2)$ 存在的可能性。

方法：使用 development set 确定阈值 $T_R$，如果 $g(e_1,R,e_2)>T_R$，则关系存在。

构建测试集：随机替换正确三元组中的实体，得到 2×#Test 大小的三元组，正项：负项=1:1。使用同样的方法构建 development set。

评价指标：三元组被正确分类的准确率

**Model Comparison**

使用 development set 交叉验证找最优参数：1）初始化向量；2）正则项参数；3）隐藏层维度；4）训练迭代次数。最终 NTN 模型的 slice 数量为4，也就是 k=4。

下表是不同模型在两个数据集上的分类准确率，NTN 效果最好。

<img src=".\fig\ntn\10.png" style="zoom:50%;" />

实体向量不同的初始化方法，EV - 实体向量；WV - 随机初始化词向量；WV-init - 用预训练词向量初始化。结果表明：

- 在 WordNet 中，word vector 明显好于 entity vector，可能是因为 WordNet 中的实体 share more common words
- WV-init 可以提升模型效果

<img src=".\fig\ntn\11.png" style="zoom:50%;" />

##### 4.3 Examples of Reasoning

- WordNet

<img src=".\fig\ntn\12.png" style="zoom:50%;" />

- Freebase

  通过 (Francesco Guicciardini, place\_of\_birth, Florence)和 (Francesco Guicciardini, profession, historian)推断出他的性别和国籍。后者可能是因为在训练集中存在三元组（Matteo Rosselli，location，Florence）和（Matteo Rosselli，nationality，Italy）；前者可能是因为训练集中存在（Francesco Patrizi，nationality，Italy）和（Francesco Patrizi，gender，male），Francesco Patrizi 和 Francesco Guicciardini 没有明显的联系，但是 Francesco Patrizi 和 Francesco Guicciardini 有共同的单词，这就是用词向量表示实体的好处。

<img src=".\fig\ntn\13.png" style="zoom:50%;" />

#### Conclusion

提出了 NTN 用于知识库补全任务。我们的模型通过向量进行实体交互。我们的方法只利用了给定了知识库，在没有 external textual resources 时也能有很好的效果。我们还表明了用无监督预训练的词向量表示实体能提升模型效果。



