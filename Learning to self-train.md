# Learning to Self-Train for Semi-Supervised Few-Shot Classification

jtwang 2022/11/29

> Li X, Sun Q, Liu Y, et al. Learning to self-train for semi-supervised few-shot classification[J]. Advances in Neural Information Processing Systems, 2019, 32.
> 论文链接：https://aclanthology.org/2021.emnlp-main.836.pdf


## Abstract

- 问题：few-shot classification（FSC）
- Meta-learning，by learning to initialize a classification model for FSC.
- 本文提出一个半监督元学习方法learning to self-train (LST)
- cherry-pick
- 在每个任务上，训练一个小样本模型预测伪标签，利用labeled + pseudo labeled data迭代训练（self-train），每一次迭代后都进行fine-tune
- 学习一个soft weighting network（SWN），为pseudo labels分配权重so that better ones can contribute more to gradient descent optimization.


## Introduction

- 小样本学习方法
  1. meta-learning
  2. 半监督学习，使用unlabeled data

- 一个经典、直观、简单的半监督学习方法是self-training
- Self-training：It first trains a supervised model with labeled data, and then enlarges the labeled set based on the most confident predictions (called pseudo labels) on unlabeled data
- 本文关注的问题：半监督小样本分类任务
- few labeled data + much larger amount of unlabeled data for training classifiers
- 本文提出一个方法learning to self-train（LST），将自训练嵌入到meta gradient decent paradigm
- 难点： directly applying self-training recursively may result in gradual drifts and thus adding noisy pseudo-labels
- 为了解决上面这个问题，我们提出
  1.  meta-learn a soft weighting network (SWN) to automatically reduce the effect of noisy labels
  2.  fine-tune the model with only labeled data after every self-training step

- LST方法包含两部分
  1. inner-loop self-training (for one task)
  2. outer-loop meta-learning (over all tasks)

- LST meta-learns both
  -  initialize a self-training model
  -  how to cherry-pick from noisy labels

- 总体流程
  - An **inner loop** starts from the meta-learned initialization by which a task-specific model can be fast adapted with few labeled data
  - this model is used to predict pseudo labels, and labels are weighted by the meta-learned soft weighting network (SWN). 
  - Self-training consists of re-training using weighted pseudo-labeled data and fine-tuning on few labeled data.
  -  In the **outer loop**, the performance of these meta-learners are evaluated via an independent validation set, and parameters are optimized using the corresponding validation loss.



