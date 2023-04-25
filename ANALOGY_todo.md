# Analogical Inference for Multi-relational Embeddings

jtwang	2022/03/31[^1]

[^1]: 【】里是我的注释

> Liu H, Wu Y, Yang Y. Analogical inference for multi-relational embeddings[C]//International conference on machine learning. PMLR, 2017: 2168-2178.

## Abstract

利用实体和关系嵌入的analogical properties【类比性质】，提出模型ANALOGY，我们的模型enjoy both theoretical power and computation scalability，并且超过了许多baseline方法。此外，我们将许多嵌入方法纳入到一个统一的框架中。



## Introduction

**Motivation**	从analogical inference的角度解决KG嵌入问题。如果system A（实体和关系的一个子集）与system B（实体和关系的另一个子集）类似，那么B中unobserved三元组可以通过mirroring their counterparts in A推断得到。下图是一个例子，红色是atom System，蓝色是Solar system。将原子系统看作一个微缩的太阳系（通过scale_down关系），我们可以complete missing facts (triplets) about the latter by mirroring the facts about the former。The analogy is built upon three basic analogical structures (三个平行四边形): “sun is to planets as nucleus is to electrons”, “sun is to mass as nucleus is to charge” and “planets are to mass as eletrons are to charge”.

<img src=".\fig\ANALOGY\1.png" style="zoom:50%;" />

虽然analogical reasoning（类比推理）was an active research topic in classic AI，早期的模型都集中在不可微的rule-based推理，不能应用到大规模的知识图谱上。How to leverage the intuition of analogical reasoning via statistical inference for automated embedding of very large knowledge graphs has not been studied so far, to our knowledge。

It is worth mentioning that analogical structures have been observed in the output of several word/entity embedding models . However, those observations stopped there as merely empirical observations. 【好难翻译。。】提出了几个问题：

- mathematically formulate the desirable analogical structures，并且在目标函数中借助其来提高嵌入效果
- develop new algorithms for tractable inference for the embedding of very large knowledge graphs

我们将这个open challenge命名为**analogical inference**，以便与rule-based analogical reasoning区分开来，

本文的贡献：

- a new framework，在多关系嵌入中建模analogical structure，and that improves the state-of-the-art performance on benchmark datasets
- The algorithmic solution for conducting analogical inference in a differentiable manner, whose implementation is as scalable as the fastest known relational embedding algorithms
- the theoretical insights on how our framework provides a unified view of several representative methods as its special (and restricted) cases, and why the generalization of such cases lead to the advantageous performance of our method as empirically observed.

