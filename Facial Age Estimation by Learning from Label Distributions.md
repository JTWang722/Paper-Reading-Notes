# Facial Age Estimation by Learning from Label Distributions

jtwang  2022/12/29

> Geng X, Yin C, Zhou Z H. Facial age estimation by learning from label distributions[J]. IEEE transactions on pattern analysis and machine intelligence, 2013, 35(10): 2401-2412


- **Motivation**: Faces at the close ages look quite similar, which results from the fact that aging is a slow and gradual progress. Inspired by this observation, the basic idea behind this paper is to **utilize the images at the neighboring ages while learning a particular age**.

- The utilization of adjacent ages is achieved by introducing a new labeling paradigm, i.e., **assigning a label distribution to each image rather than a single label of the real age**. A suitable label distribution will make a face image contribute to not only the learning of its real age, but also the learning of its neighboring ages.

- which is not the probability that class y correctly labels the instance, but the degree that y describes the instance, or in other words, the proportion of y in a full class description of the instance
- 在一个label distribution中，每个标签y有一个概率P(y)，representing the degree that the corresponding label describes the instance. 限制是$\sum_y y=1$
![图 1](fig/Facial%20Age%20Estimation%20by%20Learning%20from%20Label%20Distributions/Facial%20Age%20Estimation%20by%20Learning%20from%20Label%20Distributions_1.png)  


- The label distribution assigned to a face image with the real age $\alpha$ should satisfy the following two properties:
  - The probability of $\alpha$ in the distribution is the highest
  - The probability of other ages decreases with the distance away from $\alpha$
- 两种典型的分布是高斯分布和三角形分布
![图 2](fig/Facial%20Age%20Estimation%20by%20Learning%20from%20Label%20Distributions/Facial%20Age%20Estimation%20by%20Learning%20from%20Label%20Distributions_2.png)  


- The problem of learning from label distributions (LLD) can be formally described as
  -  Given a training set $S={(x_1,P_1(y)),  ..., (x_n,P_n(y))}$
  -  the goal is to learn a conditional $p(y|x)$

- Suppose $p(y|x)$ is a parametric model $p(y|x; θ)$, 
  - where $\theta$ is the vector of the model parameters.
  - the goal of LLD is to find the $θ$ that can generate a distribution similar to $P_i(y)$ given the instance $x_i$
  - 使用KL散度度量两个分布间的相似度
  ![图 3](fig/Facial%20Age%20Estimation%20by%20Learning%20from%20Label%20Distributions/Facial%20Age%20Estimation%20by%20Learning%20from%20Label%20Distributions_3.png)  

- 公式推导没看，最后推出一个梯度公式，算法如下

![图 4](fig/Facial%20Age%20Estimation%20by%20Learning%20from%20Label%20Distributions/Facial%20Age%20Estimation%20by%20Learning%20from%20Label%20Distributions_4.png)  



## 总结
使用LLD进行年龄估计，出发点是认为邻近的年龄对当前图片的估计有帮助，因为年龄增长是一个缓慢的过程。输入不再是标签，而是一个label distribution，这个分布是生成的高斯分布/三角形分布。训练的目标是使得输出的标签$p(y|x; θ)$与输入的标签分布尽可能相似。