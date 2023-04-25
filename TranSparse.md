# Knowledge Graph Completion with Adaptive Sparse Transfer Matrix

jtwang	2022/03/26[^1]

[^1]: 【】里是我的注释

> Ji G, Liu K, He S, et al. Knowledge graph completion with adaptive sparse transfer matrix[C]//Thirtieth AAAI conference on artificial intelligence. 2016.

## Abstract

关注知识图谱的heterogeneity（一些关系链接许多实体对，其他关系却没有）和imbalance（一个关系头尾实体的数量不均衡），本文提出TranSparse解决这两个问题。将transfer matrices替换为adaptive sparse matrices，sparse degree由关系链接的实体数量决定。在实验部分，我们为transfer matrices设计了结构化和非结构化的sparse pattern，并分析了它们的优缺点



## Introduction

- Heterogeneity：一些关系链接了许多实体对（复杂关系）；另一些没有（简单关系）

- Imbalance：一些关系链接了许多头（尾）实体，却只有很少的尾（头）实体，如gender

<img src=".\fig\TranSparse\1.png" style="zoom:50%;" />

因此，我们假设复杂度不同的关系应该使用不同的表示模型去学习，关系两边应该单独建模。本文采用sparse matrices建模关系。

为了解决heterogeneity，我们提出TranSparse(share)模型，transfer matrices的sparse degree由关系链接的实体对数量决定，关系两边共享相同的transfer matrices。复杂关系的transfer matrices would be less sparse than 简单关系。

为了解决imbalance，我们在TranSparse(share)上做了改进，提出了TranSparse(separate)模型，每个关系有两个单独的sparse transfer matrices，分别对应头尾实体。sparse degree由头（尾）实体的数量决定。

我们的贡献：

- 提出了新的方法嵌入知识图谱，考虑了heterogeneity和imbalance
- 我们的方法efficient，参数更少，可以扩展到大规模知识图谱
- 提出了两个sparse patterns for transfer matrices，并分析了它们的优缺点
- 在三元组分类和连接预测任务中取得了SOTA效果



## Our Model

**Sparse matrix**	大部分元素为0的矩阵。只需要保存不为0的元素，节省存储空间；还能减少计算成本。下图是两个典型的稀疏矩阵，（a）是结构化的，非零元素集中在对角线上，这种有利于matrix-by-vector乘法运算，更容易扩展到大规模知识图谱；（b）是非结构化的稀疏矩阵，非零元素随机均匀分布，这种通用性更强，实验结果更好。

**Sparse degree**	矩阵中0元素的占比。使用$M_\theta$表示sparse degree=$\theta$的稀疏矩阵$M$

<img src=".\fig\TranSparse\2.png" style="zoom:50%;" />

**Sparse matrix vs low-rank matrix**	我们的motivation是使用higher and lower degrees of freedom【矩阵的自由度反映了矩阵中每个元素互相约束的状态】的矩阵学习复杂关系和简单关系。low-rank和sparsity都可以减少degrees of freedom，因为它们都对矩阵做了一些限制。但是稀疏矩阵更适合我们的任务，原因有两点：

- 稀疏矩阵比低秩矩阵更灵活。对于一个矩阵$M^{m\times n}$，只有$\min(m,n)$个低秩矩阵，但是有$m\times n$个稀疏矩阵。对于包含很多关系的数据集，如FB15k包含1345个关系，稀疏矩阵更flexible
- 稀疏矩阵比低秩矩阵更efficient。只有非零元素参与运算，减少计算成本

##### TranSparse

每个关系都有一个transfer matrix。为了避免欠拟合或过拟合，复杂关系链接更多的头尾实体，需要更多的参数to learn fully；简单关系需要较少的参数to learn properly。在我们的模型中，每个关系的transfer matrix都是稀疏矩阵。我们认为关系的复杂度与它链接的三元组（或者实体）数量成正比，因为一个关系链接的数据越多，它包含的知识就越多。

**TranSparse(share)**	为每个关系$r$分配一个sparse transfer matrix $\mathbf{M}_r(\theta_r)$ 和一个translation vector $\mathbf{r}$。$N_r$表示关系$r$链接的实体对数量，$N_{r^*}$是最大值，$r^*$链接最多的实体对。为$\mathbf{M}_{r^*}$设置一个最小的稀疏度$0<=\theta_{min}<=1$（超参数）。Transfer matrix的稀疏度定义为
$$
\theta_r=1-(1-\theta_{min})N_r/N_{r^*}
$$
投影头尾实体向量为
$$
\mathbf{h}_p=\mathbf{M}_r(\theta_r)\mathbf{h}\quad\quad \mathbf{t}_p=\mathbf{M}_r(\theta_r)\mathbf{t}
$$
**TranSparse(seperate)**	为每个关系分配两个稀疏矩阵$\mathbf{M}_r^h(\theta_r^h)$和$\mathbf{M}_r^t(\theta_r^t)$。$N_r^l(l=h,t)$表示关系链接的头尾实体数量，$N_{r^*}^{l^*}$表示最大值。为$\mathbf{M}_{r^*}^{l^*}$设置一个超参数$0<=\theta_{min}<=1$。Transfer matrix的稀疏度定义为：
$$
\theta_r^l=1-(1-\theta_{min})N_r^l/N_{r^*}^{l^*}\quad (l=h,t)
$$
实体映射为
$$
\mathbf{h}_p=\mathbf{M}_r^h(\theta_r^h)\mathbf{h}\quad\quad \mathbf{t}_p=\mathbf{M}_r^t(\theta_r^t)\mathbf{t}
$$
两个模型的评分函数都为
$$
f_r(h,t)=\norm{\mathbf{h}_p+\mathbf{r}-\mathbf{t}_p}^2_{\mathscr{l}_{1/2}}
$$

##### Algorithm implementation

使用TransE初始化实体和关系嵌入，限制transfer matrix为方阵，使用单位矩阵【只有对角线为1，其余为0】初始化。

对于一个transfer matrix $\mathbf{M}(\theta)\in\mathbb{R}^{n\times n}$，它总共有$nz=\theta\times n\times n$个非零元素。除了对角线上还有$nz'=nz-n$个非零元素，当$nz<=n$时，我们设置$nz'=0$（这种情况下transfer matrix是单位矩阵）。

当使用structured pattern for $\mathbf{M}(\theta)$，使nz‘个非零元素位于对角线的两边并且对称，可以适当调整nz’使之满足这个条件；当使用unstructured pattern，则随机分散nz'个非零元素。

在训练之前，首先设置超参数$\theta_{min}$，然后计算所有transfer matrix的稀疏度；使用structured/unstructured pattern构建spares transfer matrix。在训练时只更新非零元素。
