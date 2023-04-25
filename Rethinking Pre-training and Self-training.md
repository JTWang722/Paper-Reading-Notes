# Rethinking Pre-training and Self-training

jtwang 2022/11/1

> Zoph B, Ghiasi G, Lin T Y, et al. Rethinking pre-training and self-training[J]. Advances in neural information processing systems, 2020, 33: 3833-3845.
> 论文链接：https://arxiv.org/pdf/2006.06882v1.pdf

这是一篇验证性的文章，通过实验比较预训练与自训练这两种方法在不同场景下的表现差异。并没有提出新的方法，贡献在于实验设计与实验发现。

## Abstract

在ImageNet做预训练/自训练，用于COCO目标检测，发现自训练表现更优。

## Introduction

**预训练**：Expected a model, pre-trained on one dataset, to help another。但是He et al. [1]证明such ImageNet pre-training does not improve accuracy on the COCO dataset

**自训练**：First discard the labels on ImageNet. We then train an object detection model on COCO, and use it to generate pseudo labels on ImageNet. The pseudo-labeled ImageNet and labeled COCO data are then combined to train a new model

**任务**：use ImageNet as additional data with the goal of improving COCO

做了一系列对比实验，发现：
1. Self-training works well exactly on the setup that pre-training fails（数据增强强度很高时）
2. Self-training is very flexible about unlabeled data sources, model architectures and computer vision tasks
3. 但是本文并没有否认预训练在CV中的作用

## Related Work
He et al. [1], however, demonstrate that ImageNet pre-training does not work well if we consider a much different task such as COCO object detection.
>


Compared to He et al. [1], our work takes a step further and studies the role of pre-training in computer vision in greater detail with stronger data augmentation, different pre-training methods (supervised and self-supervised), and different pre-trained checkpoint qualities.
与上一篇工作想比，我们的工作更深入，对比了data augmentation强度、监督或半监督的预训练方法、pre-trained checkpoint qualities

Our work argues for the scalability and generality of self-training。我们的工作验证了自训练的可扩展性与普适性


## Methodology

**数据增强**：horizontal flips and scale jittering
![图 1](fig/Rethinking%20Pre-training%20and%20Self-training/Rethinking%20Pre-training%20and%20Self-training_1.png)  



**自训练**基于Noisy Student training[10]，分为三步
1. a teacher model is trained on the labeled data (e.g., COCO dataset).
2. teacher model generates pseudo labels on unlabeled data (e.g., ImageNet dataset).
3. a student is trained to optimize the loss on human labels and pseudo labels jointly

##References
[1] Kaiming He, Ross Girshick, and Piotr Dollár. Rethinking imagenet pre-training. In ICCV, 2019.
[10] Qizhe Xie, Eduard Hovy, Minh-Thang Luong, and Quoc V Le. Self-training with noisy student improves imagenet classification. In CVPR, 2020.