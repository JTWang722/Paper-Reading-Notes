# Differentiating Concepts and Instances for Knowledge Graph Embedding

jtwang	2022/03/29[^1]

[^1]: 【】里是我的注释

> Lv X, Hou L, Li J, et al. Differentiating concepts and instances for knowledge graph embedding[J]. arXiv preprint arXiv:1811.04588, 2018.

## Abstract

Concepts，表示一组具有相同属性的实例，在知识表示中是非常重要的信息。大多数传统的知识嵌入方法只编码实体（包含概念和实例）和关系，而忽略了概念和实例之间的差别。本文提出了TransC，将概念和实例区分开。将每个概念编码为球体， 每个实例编码为相同语义空间中的向量。使用相对位置建模概念和实例间的关系instanceOf以及概念和子概念间的关系subClassOf。在YAGO数据集上做了链接预测和三元组分类的实验。



## Introduction

- **Motivation**

  <img src=".\fig\TransC\1.png" style="zoom:50%;" />

  - Insufficient concept representation：现有的方法将概念和实例都当作实体，使用同样的方式建模为向量，但它们是有区别的
  - 不能保留isA（instanceOf & subClassOf）关系的传递属性

- **解决方法**：TransC将概念建模为球体，实例建模为相同语义空间中的向量，使用相对位置表示概念与实例间的关系。实例向量包含在概念球体里表示instanceOf关系，球体i包含在球体j表示subClassOf关系。



## TransC







