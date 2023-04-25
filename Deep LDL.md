# Deep label distribution learning with label ambiguity

jtwang  2023/1/3

> Gao B B, Xing C, Xie C W, et al. Deep label distribution learning with label ambiguity[J]. IEEE Transactions on Image Processing, 2017, 26(6): 2825-2838.
> Paper Link: https://arxiv.org/pdf/1611.01731.pdf



We convert the label of each image into a discrete label distribution, and learn the label distribution by minimizing a Kullback-Leibler divergence between the predicted and groundtruth label distributions using deep ConvNets

## Introduction

- Why is it difficult to collect a large and accurately labeled training set? age estimation, head pose estimation, multi-label classification and semantic segmentation
  - It is difficult to provide exact labels to some tasks
  - It is very hard to gather complete and sufficient data. 


- Thus, the publicly available age, head pose and semantic segmentation datasets are small scale compared to those in image classification tasks
- These aforementioned small datasets have a common characteristic, i.e., **label ambiguity**, which refers to the uncertainty among the ground-truth labels
    -  label ambiguity is unavoidable in some applications. 
    -  label ambiguity can also happen if we are not confident in the labels we provide for an image

- Label ambiguity will help improve recognition performance if it can be reasonably exploited.

- DLDL: Deep Label Distribution Learning
  - end-to-end learning framework which utilizes the label ambiguity in **both feature learning and classifier learning**
  - Relaxes the requirement for large amount of training images, e.g., a training face image with groundtruth label 25 is also useful for predicting faces at age 24 or 26


## DLDL

- The goal of DLDL is to directly learn a conditional probability mass function $\hat{y} = p(\mathbf{y}|X; \mathbf{\theta})$ from $D=\{(X^1,\mathbf{y}^1),..., (X^N,\mathbf{y}^N) \}$, where $θ$ is the parameters in the framework
- If the KullbackLeibler (KL) divergence is used as the measurement of the similarity between the ground-truth and predicted label distribution
![图 1](fig/Deep%20LDL/Deep%20LDL_1.png)  

- Loss function, gradient descent
![图 2](fig/Deep%20LDL/Deep%20LDL_2.png)  

- Label distribution construction
  - For age estimation, normal distribution
  - For head pose estimation, 2-d normal distribution
  - For multi-label classification, 
  - For semantic segmentation, We propose a mechanism to describe the label ambiguity in the boundaries.

