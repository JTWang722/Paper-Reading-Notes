# ComplEx

> *论文：T. Trouillon et al, Complex embeddings for simple link prediction. ICML 2016*

jtwang	2021.11.3

### 1. Introduction

- 任务：link prediction，是统计关系学习中一个主要问题
- 将link prediction看成一个3D binary tensor 补全问题，张量的每个slice是一种关系的邻接矩阵。
- 基于 low-rank factorization / embeddings
- KB中的二元关系有多重pattern：层级IsOlder、组合、相等
- TransE中说一个关系模型应该（a）学到所有这些性质的组合，自反/非自反，对称非对称，传递性（b）能够在时间和空间上都是线性
- embeddings 的 Dot product 可以很好地扩展，而且而已处理对称和（非）自反的关系；使用恰当的损失函数可以保证传递性。然后，处理非对称的关系通常意味着参数量的激增，使得模型容易过拟合。因此，需要在 expressiveness和参数空间大小之间寻找平衡
- 本文证明了嵌入间的标准点乘是一个非常有效的 composition function，只要使用了 right representations。我们讨论并证明了 complex 嵌入的有效性。Hermitian dot product 包含了其中一个向量的共轭转置。因此，dot product不再是对称的，非对称的关系可以获得不同的分数。所以复值向量可以很好地 capture 非对称的关系，同事要保持了点乘的 efficiency，在时间和空间上都是线性复杂度。
- 文章结构
  - 使用复值嵌入的动机，在一个关系的方阵上
  - 扩展到张良中的多个方阵，多种关系
  - 实验，简单快速，且给出了 systematic accuracy improvement



### 2. Relations as real part of low-rank normal matrices

我们讨论了复值嵌入在 low-rank matrix factorization 中的使用，通过一个简化的链接预测任务来阐明它（仅有单独一个关系类型）。normal matrices，左右两边的嵌入有相同的 unitary basis。

#### 2.1 Modelling relations

符号

- $\varepsilon$：实体集合，$|\varepsilon|=n$ 
- $Y_{so}\in \{-1,1\}, s\in \varepsilon, o\in \varepsilon$：实体间关系，partially observed sign matrix
- $X_{so}\in \mathbb{R}^{n\times n}$：是 latent matrix of scores

$$
P(Y_{so}=1)=\sigma(X_{so})
$$

我们的目标是寻找 X 的 generic structure，that leads to a flexible approximation of common relations in real world KBs.

标准的矩阵分解：$X\approx UV^T,\quad U,V\in \mathbb{R}^{n\times K}$，K是矩阵的秩。这个公式假设相同的实体作为主语和宾语有不同的表示。这个方法与 SVD 很相关，可以很好的 fit 当矩阵 $X$ 是矩形。

但是，在许多链接预测问题中，相同的实体既可以作为主语也可以作为宾语出现，所以学习实体的 joint embeddings seems natural。即一个实体不管是作为主语还是宾语都有相同的表示。

为了使主语和宾语有相同的嵌入，研究者们将 dot product 扩展到 scoring functions（composition functions）中，以特定的方式 combine 嵌入。

使用相同的嵌入表示左右两边的因子可以归结为特种特征值分解，常被用来估计实对称矩阵（协方差矩阵，核函数，距离/相似度矩阵）。所有的特征值和特征向量都是实数，E 是正交矩阵。
$$
X=EWE^{-1}
$$
我们关注的是反对称（antisymmetric）的矩阵。这种情况只存在复空间中的分解。对于复特征向量 $E\in \mathbb{C}^{n\times n}$，求矩阵的逆计算复杂度很高。幸运的是，可以不用求逆。矩阵是 normal 当且仅当它是 unitarily diagonalizable。W 是特征值的对角阵，E 是单位特征向量矩阵。
$$
X=EW\overline{E}^T, W\in \mathbb{C}^{n\times n},E\in \mathbb{C}^{n\times n}
$$
Hermitian product：$<u,v>:=\overline{u}^Tv$。$\overline{u}$ 是u的共轭。$\overline{x}^Tx=0\Rightarrow x=0$，而 $x^Tx$ 就不适于这种情况。

normal matrix：对于 n×n 的复矩阵 X，$X\overline{X}^T=\overline{X}^TX$。所有的对称和反对称 sigh matrices都是normal。

我们只保留实数部分，这种映射对任意的实方阵都成立，不仅是 normal ones。
$$
X=Re(EW\overline{E}^T)
$$
对比 SVD，这种特征值分解有两点不同（1）特征值不一定是正的或者是实数（2）对于一个实体，它的主语嵌入与宾语嵌入共轭。

#### 2.2 Low-rank decomposition

在链接预测任务中，关系矩阵是未知的，目标是通过有噪声的观察完整的恢复出它。因为我们处理的是二元关系，我们假设它有 low sign-rank。sign matrix 的 sigh-rank 是一个有相同的 sign-pattern Y 的是矩阵的最小秩。sigh-rank 表示 sigh matrix 的复杂度，而且与 learnability 相关。
$$
rank_{\pm}=min_{A\in \mathbb{R}^{m\times n}}\{rank(A)|sign(A)=Y\}
$$
如果观察矩阵 Y 是 low-sigh-rank，那么我们的模型可以分解它用最多两倍的 sigh-rank。也就是，对于任何 $Y_{so}\in \{-1,1\}$，总存在一个矩阵 $X=Re(EW\overline{E}^T)，sigh(X)=Y$，rank(X) 至多是 $rank_{\pm}(Y)$ 的两倍。举一个例子：对于秩为 n×n 的单位矩阵，$rank(I)=n,rank_{\pm}(I)=3$，对于 2j 列和 2j+1 列，矩阵 I 对应的关系是 marriedTo，一个众所周知很难分解的关系。但是我们的模型可以 express it in rank 6，for any n。

通过利用一个远小于n的 low-rank K，对角阵W只有前K 个项非零，我们可以得出 $W\in \mathbb{C}^{K\times K},E\in \mathbb{C}^{n\times K}$.
$$
X_{so}=Re(e_s^TW\overline{e}_o),\quad e_s,e_o\in \mathbb{C}^{K}
$$
总结一下：

- 我们的模型包含了所有可能的二元关系
- 可以准确地描述对称、反对称关系
- 可以通过一个简单的 low-rank factorization 有效地近似关系表示，使用复数表示 latent factors（实体）。

### 3. Application to binary multi-relational data

上述的只关注一种关系类型，现在我们扩展到多种关系。

- $R,\varepsilon, X_r$：关系、实体集合，每个关系的分数矩阵

- 两个实体 s 和 o，r(s, o) 的可能性是，φ是基于关系分解的评分函数，θ是参数
  $$
  P(Y_{rso}=1)=\sigma(\phi(r,s,o;\Theta))
  $$

- $\Omega=R\otimes \varepsilon \otimes \varepsilon$

- $\{Y_{rso}\}_{r(s,o)\in \Omega}\in \{-1,1\}^{|\Omega|}$：事实集合。目标是预测 $\{Y_{r's'o'}\},r'(s',o')\notin \Omega$ 的正确或错误的概率。

损失函数为，公式（10）是 DistMult with real embeddings；公式（11）是实值。当 $w_r$ 是实数，函数是对称的；当 $w_r$ 是纯虚数，函数是反对称的。我们可以获得关系矩阵 X_r = re()+Im()。可以准确地描述对称和反对称关系。

<img src=".\fig\complex\1.png" style="zoom:50%;" />

