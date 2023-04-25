## Few-Shot Knowledge Graph Completion

jtwang	2021.12.19

> AAAI 2020

### Preliminaries

**Few-shot knowledge graph completion**	给定关系r和few-shot entity pairs (reference set), 设计一个机器学习的模型, 使得对于一个新的头实体h, 排名靠前的候选尾实体t是h的true tail entity.

**Few-shot learning settings**	每个关系是一个训练任务

- $\mathcal{T}_{mtr}$, 训练集: $D_r=\{P_r^{train},P_r^{test}\}$, 关系r和它的训练/测试实体对, 

- $P_r^{train}$ 只包含few-shot $(h_k,t_k)\in R_r$

-  $P_r^{test}=\{(h_i,t_i,C_{h_i,r})|(h_i,r,t_i)\in G\}$

- $\mathcal{L}_\Theta(h_i,t_i|C_{h_i,r},P_r^{train})$, ranking loss of relation r

- $$
  min_\Theta\mathbb{E}_{\mathcal{T_{mtr}}}\left [\sum\frac{\mathcal{L}_\Theta(h_i,t_i|C_{h_i,r},P_r^{train})}{|P_r^{test}|}\right ]
  $$

- $\mathcal{T}_{mte}$, 测试集, $\{P_{r'}^{train},P_{r'}^{test}\}$, r'与训练集中的关系r不相交

- $\mathcal{T}_{mtv}$, 验证集, $\mathcal{T}_{mtr}$的子集

- $G'$, background KG, G的子集, 排除了$\mathcal{T}_{mtr},\mathcal{T}_{mtv},\mathcal{T}_{mte}$中的关系

### Model

FSRL由三部分组成

1. 为每个实体编码heterogeneous neighbors
2. 为每个关系aggregate few-shot reference entity pairs
3. 匹配query pairs和reference set进行预测

<img src=".\fig\fsrl\1.png" style="zoom:60%;" />

#### Encoding Heterogeneous Neighbors

GMatching利用one-hop邻居增强实体表示, 编码了图结构信息, 但是它忽略了邻居可能会有不同的impact. 我们设计了一个relation-aware heterogeneous neighbor encoder. 对于一个头实体h, 加入attention机制, 编码它的邻居, 得到h的特征表示. f(h)函数为不同的邻居赋予不同的权重, 考虑的不同邻居的不同的impact, 这个权重是通过关系和尾实体共同计算得到的.

- 给定头实体h, 它的邻居表示为 $\mathcal{N}_h=\{(r_i,t_i)|(h,r_i,t_i)\in G'\}$

- $$
  f_\theta(h)=\sigma(\sum_i\alpha_ie_{t_i})\\
  \alpha_i=\frac{exp\{u^T_{rt}(W_{rt}(e_{r_i}\oplus e_{t_i})+b_{rt}) \}}{\sum_jexp\{u^T_{rt}(W_{rt}(e_{r_j}\oplus e_{t_j})+b_{rt}) \}}
  $$

- $e_{r_i}, e_{t_i}\in \mathbb{R}^{d\times 1}$ , 预训练的关系, 尾实体向量

- $u_{rt}\in \mathbb{R}^{d\times 1},W_{rt}\in \mathbb{R}^{d\times 2d},b_{rt}\in \mathbb{R}^{d\times 1}$ , 要学习的参数

#### Aggregating Few-Shot Reference Set

GMatching只有一个reference pair, 我们需要设计一个模型建模reference set中的几个样本的interactions (注: 一个reference set中有多个样本, 每个样本都有一个表示, 聚合后得到整个reference set的表示). 

- $\varepsilon_{h_k,t_k}=[f_\theta(h_k)\oplus f_\theta(t_k)]$, 对每个$(h_K,t_k)\in R_r$, 应用$f_\theta(h)$得到

- $$
  f_\epsilon (R_r)=\mathcal{AG}_{(h_k,t_k)\in R_r}\{\varepsilon_{h_k,t_k}\}
  $$

- 使用RNN自编码器, reconstruction loss $\mathcal{L}_{re}$

<img src=".\fig\fsrl\2.png" style="zoom:60%;" />

<img src=".\fig\fsrl\3.png" style="zoom:60%;" />

<img src=".\fig\fsrl\4.png" style="zoom:60%;" />

- 所以, 用如下公式计算 $f_\epsilon (R_r)$, 如框架图所示, 利用hidden state+attention计算得到 (注:为什么要用隐状态???为什么要用自编码器???)

<img src=".\fig\fsrl\5.png" style="zoom:60%;" />

#### Matching Query and Reference Set

- $f_\theta$, neighbor encoder
- $f_\epsilon$, reference set aggregator
- $f_\mu$, matching
- $(h_l,t_l)\in Q_r$, query entity pair of relation r
- 计算得到 $\varepsilon_{h_l,t_l}=[f_\theta(h_l)\oplus f_\theta(t_l)], f_\epsilon (R_r)$, 需要对这两个向量进行匹配

采用recurrent processor进行多步匹配, 第t步匹配过程如下

<img src=".\fig\fsrl\6.png" style="zoom:60%;" />

- RNN_match 是一个LSTM cell
- 输入$\varepsilon_{h_l,t_l}$, hidden state $g_t$, cell state $c_t$
- $\varepsilon_{h_l,t_l}=g_T$, 最后一个hidden state

- $\varepsilon_{h_l,t_l}$ 与 $f_\epsilon (R_r)$ 的inner product是最终的similarity score

#### Objective and Model Training

对于一个关系r, 随机采样几个true entity pairs $\{(h_k,t_k)|(h_k,r,t_k)\in G \}$ 作为reference set $R_r$, 剩下的正例作为query. 破坏尾实体构建负例,

- $R_r=\{(h_k,t_k)|(h_k,r,t_k)\in G \}$, reference set for relation r
- $\mathcal{P\varepsilon_r}=\{(h_l,t_l)|(h_l,r,t_l)\in G \cap (h_l,t_l)\notin R_r \}$, 剩下的正例entity pairs作为query
- $\mathcal{P\varepsilon_r}=\{(h_l,t_l^-)|(h_l,r,t_l^-)\notin G \}$, 负采样得到的negative entity pairs
- ranking loss

$$
\mathcal{L}_{rank}=\sum_r\sum_{(h_l,t_l)\in \mathcal{P\varepsilon_r}}\sum_{(h_l,t_l^-)\in \mathcal{N\varepsilon_r}}\left [ \xi+s_{(h_l,t_l^-)}-s_{(h_l,t_l)} \right ]_+
$$

最终的损失函数为
$$
\mathcal{L}_{joint}=\mathcal{L}_{rank}+\gamma \mathcal{L}_{re}
$$
整个训练过程如下图

<img src=".\fig\fsrl\7.png" style="zoom:60%;" />

### Experiments

**数据集**	NELL和Wikidata, 只用有50~500个三元组的关系

**Baseline**	RESCAL, TransE, DistMult, ComplEx; GMatching(MaxP/MeanP/Max)

**Evaluation Metrics**	top-k ration Hit@k; mean reciprocal rank MRR; few-shot size K=3

#### Results

<img src=".\fig\fsrl\8.png" style="zoom:50%;" />

##### Overall results

- GMatching结果优于传统的关系嵌入方法, 说明加入局部图结构和matching network是有效的 (注: 为什么不在传统的方法上加入graph local structure????因为数据规模吗)
- FSRL表现最优, 说明 heterogeneous neighbor encoder和recurrent autoencoder aggregation network的有效性

##### Comparison over different relations

比较了GMatching和FSRL在不同关系上的表现

- 不同关系的差别很大, 这是因为不同关系的候选集大小不同, 候选集越小, 分数越高
- FSRL比GMatching表现更好, 说明我们的模型更robust for different relations

##### Ablation Study

- AS_1, 为了investigate heterogeneous neighbor encoder的作用, 使用mean-pooling layer替代(注: 也就是取邻居的均值)
- AS_2, 分析不同aggregation network的影响. AS_2a使用mean pooling(注: 取reference entity pair的均值); AS_2b使用mean pooling代替attention weight; AS_2c, 去掉decoder
- AS_3, matching network的作用, 去掉LSTM cell, 使用query embedding和reference embedding的inner-product作为相似度得分.

<img src=".\fig\fsrl\11.png" style="zoom:50%;" />

##### Analysis

**Impact of few-shot size**	K增大, 表现变好; FSRL>GMatching

<img src=".\fig\fsrl\10.png" style="zoom:50%;" />

**Embedding Visualization**	2D, positive/neg candidate entity paris

<img src=".\fig\fsrl\9.png" style="zoom:50%;" />

### Conclusion

定义了few-shot KG补全问题, 提出了few-shot关系学习模型FSRL. heterogeneous neighbor encoder+recurrent autoencoder aggregation network+matching network, 是一个joint 模型.