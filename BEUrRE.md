## Probabilistic Box Embeddings for UKG Reasoning

jtwang	2021.11.30

### Abstract

KB中通常包含多个来源的facts，有许多是noisy或者互相冲突的，这造成了每个三元组的uncertainty。KB通常是不完整的，需要用embedding的方法进行补全。然而现有的方法值只建模triple-level的不确定性，缺乏global consistency的推理。

我们提出了BEUrRE，将每个实体建模成box（axis-aligned hyper-rectangle），将关系建模成头尾实体box之间的affine transforms。box的geometry可以有效地计算intersection和volume，赋予模型calibrated probabilistic semantics，并且可以融合relation constraints。

### Introduction

UKG数据集：ProBase[1]、NELL[2]、ConceptNet[3]为每个fact添加了一个confidence score，代表这个fact为真的可能。

举个例子。confidence information在回答“Who is the main conpetitor of Honda”是很有用的

- （Honda, competeswith, Toyota）（Honda, competeswith, Chrysler）都是正确的，但前者应该有更高的confidence，因为Honda和Toyota都是日本公司，有更多重叠的用户基础。
- （The Beatles, genre, Rock）（The Beatles, genre, Pop）都正确，但前者有更高的confidence，因为Beatles通常被认为是一个摇滚乐队。

现有的UKG embedding方法[4] [5]，是将实体当做低维空间中的点，用vector similarity度量triple plausibility，然后将plausibility映射到[0,1]区间作为置信度。把triple当做binary random variable。（binary 随机变量是什么？？？？）为了go beyond triple-level uncertainty，我们将每个实体看做binary random variable。

但是在嵌入空间中，使用简单的geometric objects如vector很难建模实体的边缘和联合概率分布，是non-trivial。已有工作研究在嵌入空间中的概率分布，[6] [7] [8]使用更复杂的geometric object表示随机变量，如圆锥、box，使用体积作为probability measure。

基于此，我们提出了BEUrRE（Box Embedding for Uncertain RElaitonal data），将实体表示为box，关系表示为两个separate affine transforms，confidence表示为两个transformed boxes之间的intersection。

这种表示方法不仅符合人们的常识--实体有不同的granularity，也可以引入更多的domain knowledge。UKGE已经证明引入domain knowledge（关系的传递性）可以增强推理。但是UKGE使用PSL推理unseen facts，添加额外的样本用于训练。这种方法会导致error propagation，而且当图谱是sparse时无效。我们提出了sufficient conditions for these relation properties to be preserved in the embedding space，并且直接建模关系的属性by regularizing relation-specific transform based on constraints。这样做可以对噪声更鲁棒，不受限于稀疏性。

### Related Work

##### UKG Embedding

数据集：ConceptNet；NELL；Probase

嵌入方法：UKGE使用PSL；UOKGE使用本体结构，但通常本体不可获得（实体type）；URGE对uncertain graph嵌入，只有结点嵌入，不能处理多关系数据。

##### Geometric Embeddings

使用比向量更复杂的结合对象进行嵌入。一些图嵌入领域的方法……把已有的box embedding用来表示实体，用transform表示关系也有前人的工作。

### Background

##### Probabilistic Box Embeddings

box：n维hyper-rectangle，intervals的乘积。体积volume是边长的乘积，是未归一化的概率。

<img src=".\fig\beurre\1.png" style="zoom:35%;" />

$\Omega_{Box}\subseteq \mathbb{R}^n$

$B(\Omega_{Box})\subseteq \varepsilon $

Y：binary random variables on $\Omega_{Box}$，$Y^{-1}\in B(\Omega_{Box})$    ??????????

$f(s)=:Y_s,Y_s^{-1}=:Box(s)$

一个例子

<img src=".\fig\beurre\2.png" style="zoom:35%;" />

#### Gumbel Box

probabilistic box embedding不能通过梯度下降训练，许多损失函数都无梯度signal。为了缓解这个问题，有人提出了一个latent noise model，用Gumbel分布建模端点。如下图，μ是location，β是variance。

<img src=".\fig\beurre\3.png" style="zoom:70%;" />

Gumbel box are closed under intersection？？？？？？？？？？？？？

体积的期望是

<img src=".\fig\beurre\4.png" style="zoom:35%;" />

条件概率

<img src=".\fig\beurre\5.png" style="zoom:35%;" />

则有如下，这个定义不严格
$$
Box(X)\sube Box(Y)\rightarrow P(Y|X)=1
$$

### Method

将实体建模成Gumbel boxes，将关系建模成affine transforms。还在训练时融合了logical constraints。

实体：Gumbel box 的参数有center和offset。

<img src=".\fig\beurre\6.png" style="zoom:60%;" />

关系：仿射变换的参数有平移和缩放向量。对于一个关系头尾实体有不同的变换f_r和g_r

<img src=".\fig\beurre\7.png" style="zoom:60%;" />

三元组（h,r,t），使用条件概率建模三元组的confidence

<img src=".\fig\beurre\8.png" style="zoom:60%;" />

$f_r(Box(h))$认为是在关系r头实体的语境下，表示概念h的support set of a binary random variable。

##### Logical constraints

现实世界中的UKG是稀疏的。UKGE引入了关于关系属性（传递性）的domain knowledge。使用PSL利用一阶逻辑规则进行推理，得到一些unseen facts。这个方法是在关系属性上的限制。由于图谱的稀疏性，这种方法很受限，不能广泛应用。

我们提出了两个logical constraints：transitivity和composition。

传递性：限制$g_r(Box(B))$ 包含$f_r(Box(B))$。损失函数

<img src=".\fig\beurre\9.png" style="zoom:30%;" />

<img src=".\fig\beurre\10.png" style="zoom:60%;" />

composition：限制$f_{r3}=f_{r2}\cdot f_{r1};g_{r3}=g_{r2}\cdot g_{r1}$，复合关系，也就是$f_{r3}(u)=f_{r2}(f_{r1}(u))$

<img src=".\fig\beurre\11.png" alt="image-20211205190537004" style="zoom:50%;" />

<img src=".\fig\beurre\12.png" style="zoom:60%;" />

<img src=".\fig\beurre\13.png" style="zoom:60%;" />

##### Learning Objective

损失分两部分，最终的损失函数=J1+J2。

相当于一个回归任务，使用MSE损失，惩罚未观测到的三元组，使它们的confidence低。

<img src=".\fig\beurre\14.png" style="zoom:45%;" />

添加的两个限制，分别给权重。

<img src=".\fig\beurre\15.png" style="zoom:66%;" />

### Experiments

#### 数据集

跟UKGE中一样，使用CN15k和NL27k。85%、7%、8%用来训练、验证、测试

#### Baseline

- UKGE、UKGE（rule+）

- URGE：本来是嵌入概率异质图的，不能处理multi-relational graphs，所以忽略关系信息

- TransE、DistMult、ComplEx、RotatE、TuckER 用来比较ranking task

#### Confidence prediction

预测unseen triple的confidence，使用MSE和MAE评估。

<img src=".\fig\beurre\16.png" style="zoom:60%;" />

##### case study：

atLocation谓语的宾语，按体积排序，前10个是place、town、bed……，一些general concepts覆盖范围更广，而一些具体的Tunisia，Morocco……specific location覆盖范围最小。表明 box volume可以表示概率，和概念的specificity/granularity。

#### Fact ranking

给定头实体和关系，根据confidence预测尾实体的排名。使用nDCG评估排名。







Reference

[1] Wentao Wu, Hongsong Li, Haixun Wang, and Kenny Q Zhu. 2012. Probase: A probabilistic taxonomy for text understanding. In Proceedings of ACM SIGMOD International Conference on Management of Data (SIGMOD).

[2] Tom Mitchell, William Cohen, Estevam Hruschka, Partha Talukdar, B Yang, J Betteridge, A Carlson, B Dalvi, M Gardner, B Kisiel, et al. 2018. Neverending learning. Communications of the ACM.

[3] Robert Speer, Joshua Chin, and Catherine Havasi. 2017. Conceptnet 5.5: An open multilingual graph of general knowledge. In Proceedings of AAAI Conference on Artificial Intelligence (AAAI).

[4] UKGE

[5] Natthawut Kertkeidkachorn, Xin Liu, and Ryutaro Ichise. 2019. Gtranse: Generalizing translationbased model on uncertain knowledge graph embedding. In Annual Conference of the Japanese Society for Artificial Intelligence, pages 170–178. Springer.

[6] Alice Lai and Julia Hockenmaier. 2017. Learning to predict denotational probabilities for modeling entailment. In EACL.

[7] Luke Vilnis, Xiang Li, Shikhar Murty, and Andrew McCallum. 2018. Probabilistic embedding of knowledge graphs with box lattice measures. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, ACL 2018, Melbourne, Australia, July 15-20, 2018, Volume 1: Long Papers, pages 263–272. Association for Computational Linguistics.

[8] Shib Sankar Dasgupta, Michael Boratko, Dongxu Zhang, Luke Vilnis, Xiang Lorraine Li, and Andrew McCallum. 2020. Improving local identifiability in probabilistic box embeddings. In Advances in Neural Information Processing Systems.

