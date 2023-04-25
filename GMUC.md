# Gaussian Metric Learning for Few-Shot UKG Completion

jtwang	2021.11.17

### Abstract

知识图谱的自动构建过程引入了许多不确定的事实。Uncertain knowledge graph 如 NELL 和 Probase 将这种不确定性建模成事实的置信度，用来更准确地描述 knowledge。现有的 UKG 补全方法对于每种关系类型都需要充足的训练数据。然而，在 real-world UKG 中，大多数关系都只有几个事实三元组。为了解决这个问题，我们提出了一个基于高斯度量学习的方法，用少量可用样本完成 UKG 上的知识补全（补全缺失的事实和置信度）。

### Introduction

大多数知识图谱如 Freebase、DBpedia、Wikidata 包含确定的事实，是 Deterministic KG（DKG）。由于自动化构建技术被广泛用于构建大规模知识图谱，许多不确定的事实使得 DKG 的确定性难以保证；此外，许多领域知识如医疗、金融方面的，本身就不能表示为 deterministic facts。所以，UKG 用置信度表示这种不确定性，提供了更准确的信息。UKG 在 highly risk-sensitive 的应用前景很好，如 drug discovery、investment decisions。

现有的 UKG 补全的研究通常假设对每一种关系都有足够多的的训练样例，但事实上，由于数据的长尾分布，大多数关系只有 few facts in real-world UKG。

在小样本设置下补全 UKG 是 non-trivial 的：（1）实体和关系本身的 internal uncertainty 是必要的，但是以前的工作忽略了这一点（2）现有的小样本 DKG 补全方法不能直接用于 UKG，因为这些模型假设图谱中的所有事实都完全正确，没有任何噪声，忽略了事实的质量不同，而且只能补全缺失的事实而不能预测置信度。

为了解决上述问题，我们提出了一个基于 Gaussian metric learning 的小样本 UKG 补全方法，目标是学习一个相似度的度量 that 可以补全 misssing  facts and  confidence score。首先使用 Gaussian neighbor encoder 将三元组表示为一个多元高斯分布，均值表示语义信息，方差表示 internal uncertainty。Gaussian matching function 用来发现 new facts 并预测置信度。

在实验中，我们新构建了 a four datasets 在小样本场景下。为了评估在 real-world UKG 下的表现，这些数据集有不同数量的噪声（uncertainty level）。两个任务：link prediction、confidence prediction。

我们的贡献

- 第一个考虑了 UKG 中关系的长尾分布，定义其为 few-shot UKG completion
- 提出来一个新的方法，用于小样本 UKG
- 构建了新的数据集，有不同的噪声程度
- 在两个任务上评估，链接预测和置信度预测

### 3  Problem Definition

**Few-shot UKG completion**：对于某个关系，基于小样本的 support set，根据头实体，预测尾实体和置信度。下图是一个例子。

<img src=".\fig\GMUC\1.png" style="zoom:50%;" />

将测试数据称为 query set：{r: <($h_j$, ?), ?>}

### 4  Methodology

<img src=".\fig\GMUC\2.png" style="zoom:40%;" />

#### 4.1  Gaussian Neighbor Encoder

用来 encode support set 和 queries。

- 将每个实体和关系表示为一个多维高斯分布 $N(e^\mu, e^\Sigma)$，mu 和 Sigma 是维度为d的向量。均值表示语义信息，方差表示 internal uncertainty
- 使用 heterogeneous neighbor encoders[33] 使用邻居（图结构信息）加强每个实体的表示。对于一个实体h，它的增强表示为 $N(NE_\mu(h),NE_\Sigma(h))$，NEmu 和 NESigma 是两个 heterogeneous neighbor encoders，还用了attention赋予每种关系不同的权重，计算公式如下，

<img src=".\fig\GMUC\3.png" style="zoom:70%;" />

- 每个三元组被表示为 $N(\mu,\Sigma)$，计算公式如下，使用增强后的头尾实体来表示三元组

<img src=".\fig\GMUC\4.png" style="zoom:70%;" />

- 每个 query：$N(\mu_q,\Sigma_q)$
- 一个 support set：$N(\mu_s,\Sigma_s)$，是集合中的所有三元组经过 max-pooling 后得到的

#### 4.2  Gaussian Matching Function

用来度量 queries 和 support set 间的相似度。

将相似度建模成一维高斯分布 $Simlarity~N(\varepsilon ,\delta )$，均值表示 most likely similarity value，方差在 [0, 1] 区间表示这个 similarity value 的不确定度。

使用 LSTM-based matching network[27] 计算。相比于简单的 cosine，网络可以进行multi-step matching process，效果更好。计算公式如下，MN是网络。

<img src=".\fig\GMUC\5.png" style="zoom:70%;" />

query 和 support 间的相似度，置信度 $R_{similarity},R_{confidence}$ 定义为，是基于 $Similarity$ 的分布得到的

<img src=".\fig\GMUC\6.png" style="zoom:70%;" />

#### 4.3  Learning Process

对于一个关系，随机采样几个正例作为 support set，其他的正例作为 positive queries。L_mse 用来最小化 confidence 的损失。与 TransE 相同，用 margin-based ranking loss L_rank 使正例的 $\varepsilon$ 尽可能高。对训练样本做了一个过滤，只采用置信度高于阈值的三元组训练。

<img src=".\fig\GMUC\7.png" style="zoom:70%;" />

<img src=".\fig\GMUC\8.png" style="zoom:70%;" />

<img src=".\fig\GMUC\9.png" style="zoom:70%;" />

### 5 experiments

#### 5.1 datasets

NL27k，但是数据集很少噪声和uncertain data，所以我们人工添加了不同数量的noisy triples去模仿真实世界中的UKG。构建了4个数据集NL27k-N0，NL27k-N1，NL27k-N2，NL27k-N3，分别包含0%，10%，20%，40%，然后使用CRKL[29]非配confidence score，计算公式如下。

然后我们选择有50-500个triples的关系用于few-shot任务。剩下的关系作为background relations。

#### 5.2 Baseline

UKGE，GMatching，GMUC-noconf（不考虑triple质量），GMUC-point（只是用均值，不用方差）

