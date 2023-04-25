# Deep Co-Training for Semi-Supervised Image Recognition

jtwang  2022/11/9

> Qiao, S., Shen, W., Zhang, Z., Wang, B., Yuille, A. (2018). Deep Co-Training for Semi-Supervised Image Recognition. ECCV 2018.
> 论文地址：https://openaccess.thecvf.com/content_ECCV_2018/papers/Siyuan_Qiao_Deep_Co-Training_for_ECCV_2018_paper.pdf


---
## Abstract
本文研究半监督图像识别问题，即同时利用labeled和unlabeled数据学习图像分类器。我们提出了Deep Co- Training，一个基于Co- Training（协同训练）的深度学习方法。原始的Co- Training方法学习两个分类器基于2个views（描述相同实例的不同来源的数据）。为了将这个概念扩展到深度学习中，Deep Co- Training为不同的views训练多个深度神经网络，并利用adversarial examples to encourage view difference, in order to prevent the networks from collapsing into each other[即防止不同的网络趋于一致]。这样一来，这些co-trained网络提供不同且互补的数据信息，这是协同训练框架取得良好效果的必要条件。我们在SVHN, CIFAR-10/100和ImageNet数据集上进行测试，证明了Deep Co- Training在很大程度上优于之前的SOTA。


---
## Introduction
- 深度神经网络需要大量的标注数据，如果能利用未标注数据就能进行训练就太好啦
- 本文研究半监督图像识别问题，采用labeled+unlabeled images构建分类器
- 数据集 $D=S\cup U$，核心问题在于如何利用$U$帮助对$S$的学习
  - $S$: labeled images
  - $U$: unlabeled images
  - $C$: categories in $S$

- Co- Training framework[24]
  - award-winning method for semi-supervised learning
  - 假设每个数据$x\in D$都有两个views，即$x=(v_1,v_2)$
  - each view $v_i$ is sufficient for learning an effective model
  - the views can have different data sources or different representations
  - $\mathcal{X}$: the distribution that $D$ is drawn from
  - Co- Training假设$f_1$和$f_2$trained on view $v_1$ and $v_2$ respectively have consistent predictions [在两个视图上分别训练的模型具有一致的输出]
    ![图 4](fig/Depp%20Co-Training/Depp%20Co-Training_1.png)  

- 为了结合深度学习，一个naive的做法是在$D$同时训练2个神经网络
- 但这种方法有一个显著的缺点：there is no guarantee that the views provided by the two networks give different and complementary information about each data point. [没必要训练两个一样的网络] 协同训练只有在2个view不同，且 ideally conditionally independent given the category的情况下有用
- 为了解决这个问题，提出view difference constraint，使得两个网络得以区分
![图 5](fig/Depp%20Co-Training/Depp%20Co-Training_2.png)  

- 通过adversarial example构造$\mathcal{X}'$($\mathcal{X}\cap\mathcal{X}'=\empty$)
- 本文针对半监督图像识别问题，提出DCT（Deep Co- Training），将协同训练框架扩展到深度学习中。
- 通过最小化expected Jensen-Shannon divergence between the predictions of the two networks on $U$，满足协同训练的假设
- We impose the **view difference constraint** by training each network to be resistant to the adversarial examples [28, 29] of the other. The result of the training is that each network can keep its predictions unaffected on the examples that the other network fails on. In other words, the two networks provide different and complementary information about the data because they are trained not to make errors at the same time on the adversarial examples for them.
- 总结：贡献有2个：Co-Training assumption + view difference constraint


## Deep Co-Training

dual-view $\rightarrow$ multi-view

#### Co-Training Assumption in DCT

- 符号
  - 数据集：$D=S\cup U$
  - Views: $v_1(x)$, $v_2(x)$, convolutional representations of $x$
  - Fully-connected layer: $f_i(\cdot)$
  - 预测结果：$p_1(x)=f_1(v_1(x))$, $p_2(x)=f_2(v_2(x))$
- 在有监督数据集$S$，采用交叉熵损失
  ![图 6](fig/Depp%20Co-Training/Depp%20Co-Training_3.png)  
  - $(x,y)\in S$
- 协同训练的假设是两个网络在无监督数据集$U$的预测结果相近，采用二者间的Jensen-Shannon divergence衡量相似度
  ![图 7](fig/Depp%20Co-Training/Depp%20Co-Training_4.png)  
  - $x\in U$
  - 最小化$\mathbb{E}|L_{cot}|$

#### View Difference Constraint in DCT

- 构建$D'$，使得$p_1(x)\neq p_2(x), \forall x\in D', D'\cap D=\empty$
- 生成adversarial examples, $D'=\{g(x)|x\in D\}$, $g(x)-x$ to be small
- $p_1(x)\neq p_2(x), p_1(g(x))=p_1(x), p_2(g(x))\neq p_2(x)$，这意味着$g(x)$是$p_2$的adversarial example，fool $p_2$ but not $p_1$ 
- 因此，我们要使得$p_1$能抵抗$p_2$的adversarial examples $g_2(x)$，不会被扰动
  ![图 10](fig/Depp%20Co-Training/Depp%20Co-Training_6.png)  
- We want the models to have the same predictions on $D$ but make different errors when they are exposed to adversarial attack. [两个网络不同时犯错/不犯同样的错误]



#### Training DCT
- 目标函数
  ![图 11](fig/Depp%20Co-Training/Depp%20Co-Training_7.png)  
- 采用bundles of data streams送入数据
- 一次迭代
  ![图 12](fig/Depp%20Co-Training/Depp%20Co-Training_9.png)  


#### Multi-view DCT

