# Self-training with Noisy Student improves ImageNet classification

jtwang  2022/11/1

> Xie, Qizhe, et al. "Self-training with noisy student improves imagenet classification." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.
> 论文链接：https://arxiv.org/abs/1911.04252v4
> 代码链接：https://github.com/google-research/noisystudent.


本文提出一个基于teacher-student framework的self-training方法。与之前工作的不同在于：为student的训练过程添加噪声（noise injection to the student）。
The main difference between our work and prior works is that we identify the importance of noise, and aggressively inject noise to make the student better.
本文实验结果表明：it is possible to use unlabeled images to significantly advance both <u>accuracy and robustness</u> of state-of-the-art ImageNet models.

## Abstract

We present a simple self-training method that achieves 88.4% top-1 accuracy on ImageNet, which is 2.0% better than the state-of-the-art model that requires 3.5B weakly labeled Instagram images. On robustness test sets, it improves ImageNet-A top-1 accuracy from 61.0% to 83.7%, reduces ImageNet-C mean corruption error from 45.7 to 28.3, and reduces ImageNet-P mean flip rate from 27.8 to 12.2.
我们提出了一个简单的自训练方法，达到了很高的准确率，鲁棒性也很好。

**方法**
1. Train an EfficientNet model on labeled ImageNet images and use it as a teacher to generate pseudo labels on 300M unlabeled images. Teacher在labeled data上训练，然后为unlabeled data生成伪标签
2. Train a larger EfficientNet as a student model on the combination of labeled and pseudo labeled images. Student结合labeled和peseudo labeled data进行训练
3. Iterate this process by putting back the student as the teacher. 在这个过程中，teacher的训练没有噪声，所以它生成的伪标签尽可能准确；但为student注入噪声，such as dropout, stochastic depth and data augmentation via RandAugment，这样一来，student比teacher的泛化性更强


## Introduction

**Motivation**：使用大量unlabeled data增强SOTA准确性与鲁棒性

之前的self-training框架 [70]
1. train a teacher model on labeled images
2. use the teacher to generate pseudo labels on unlabeled images
3. train a student model on the combination of labeled images and pseudo labeled images.

We iterate this algorithm a few times by treating the student as a teacher to relabel the unlabeled data and training a new student.

我们的实验表明一个关键因素在于：the student model should be noised during its training while the teacher should not be noised during the generation of pseudo labels. 因此这个方法被称为NoisyStudent

## NoisyStudent: Iterative Self-training with Noise

![图 2](fig/Self-training%20with%20Noisy%20Student%20improves%20ImageNet%20classification/Self-training%20with%20Noisy%20Student%20improves%20ImageNet%20classification_1.png)  

本文方法跟之前方法的区别在于：我们为student model添加噪声，并要求student相比teacher equal or larger。One can think of our method as Knowledge Expansion in which we want the student to be better than the teacher by giving the student model more capacity and difficult environments in terms of noise to learn through.