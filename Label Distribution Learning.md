# Label Distribution Learning

jtwang  2022/12/29

> Geng X. Label distribution learning[J]. IEEE Transactions on Knowledge and Data Engineering, 2016, 28(7): 1734-1748.
> 论文链接：https://arxiv.org/pdf/1408.6027.pdf


## Introduction

-  **Single-label learning (SLL)** assumes that all the instances in the training set are labeled in the first way
-  **Multi-label learning (MLL)** allows the training instances to be labeled in the second way
-  MLL can deal with the ambiguous case where one instance belongs to more than one classes (labels).
-  current MLL algorithms have been developed with two strategies
   -  problem transformation, transform the MLL task into one or more SLL tasks
   -  algorithm adaptation, extend specific SLL algorithms to handle multi-label data
-  Both SLL and MLL actually aim to answer the essential question “**which label can describe the instance?**”, 
-  “**how much does each label describe the instance?**”
-   it is necessary to extend MLL to LDL (label distribution learning)
    -   While MLL usually assumes indiscriminate importance within the relevant label set (e.g., all represented by ‘1’s) as well as within the irrelevant label set (e.g., all represented by ‘0’s), LDL allows direct modeling of different importance of each label to the instance, and thus can better match the nature of many real applications.
    -   LDL相对MLL的优势

  
## Related Work

-  LDL is different from these learning methods in mainly three aspects:
   -  Each training instance of LDL is explicitly associated with a label distribution, rather than a single label or a relevant (positive) label set
   -  The purpose of most numerical label indicators used in previous learning algorithms is to rank the labels, and then decide the positive label(s) through, say, thresholding over the ranking. In most cases, it only cares about the partition between positive and negative labels. On the other hand, what LDL cares about is the overall label distribution. **The value of each label’s description degree is important**.
   -  The performance evaluation measures of previous learning algorithms with numerical label indicators are still those commonly used for SLL (classification accuracy, error rate) or MLL (Hamming loss, one-error, coverage, ranking loss). On the other hand, the performance of LDL should be evaluated by the **similarity or distance between the predicted label distribution and the real label distribution**


- label embedding and attribute learning
  - the supervision signal is transformed from a label into a vector
  - instance is still associated with one class label, and the final aim is still standard classification


- multi-target learning (MTL)


- description degree vs. membership used in fuzzy classification


- Note also that $d^y_x$ is not the probability that $y$ correctly labels $x$, but the proportion that $y$ accounts for in a full description of $x$


## Formulation of LDL
- 符号
  - $x$, instance variable
  - $x_i$, i-th instant
  - $y$, label variabel
  - $y_j$, j-th label value
  - $d_x^y$, description degree of y to x
  - $D_i=\{d_{x_i}^{y_1}, d_{x_i}^{y_2}, ..., d_{x_i}^{y_c} \}$, label distribution of $x_i$
  - $c$, number of possible label values
  - $d_x^y\in[0,1]$, $\sum_y d_x^y=1$


- Both single-label annotation and multi-label annotation can be viewed as special cases of label distribution.
![图 1](fig/Label%20Distribution%20Learning/Label%20Distribution%20Learning_1.png)  


- From single-label annotation to multi-label annotation, and then to label distribution, the size of the **output space of the learning process becomes increasingly larger**.



- $d_x^y=P(y|x)$, Label distribution is a probability distribution.


- The problem of LDL can be formulated as follows:
  - $\mathcal{X}=\mathbb{R}^q$, input space
  - $\mathcal{Y}=\{y_1, y_2, ..., y_c \}$, complete set of labels
  - $S=\{(x_1,D_1), (x_2,D_2), ..., (x_n,D_n) \}$, training set
  - The goal of LDL is to learn a conditional probability mass function $p(y|x)$ from $S$
  - Suppose $p(y|x)$ is a parametric model $p(y|x; θ)$, $\theta$ is the parameter vector.
  - The goal of LDL is to find the $\theta$ that can generate a distribution similar to $D_i|x_i$


- 有不同的criteria衡量两个分布间的距离或者相似度。例如，KL散度，最优参数$\theta^*$
![图 2](fig/Label%20Distribution%20Learning/Label%20Distribution%20Learning_2.png)  

- Examine the traditional learning paradigms under the optimization criterion shown in Eq. (1).
  - For SLL, Eq. (1) can be simplified as 
  ![图 3](fig/Label%20Distribution%20Learning/Label%20Distribution%20Learning_3.png)  
  This is actually the maximum likelihood (ML) estimation of $\theta$
  - For MLL, Eq.(1) can be changed into
  ![图 4](fig/Label%20Distribution%20Learning/Label%20Distribution%20Learning_4.png)  
  Eq. (3) can be viewed as a ML criterion weighted by the reciprocal cardinality of the label set associated with each instance.

-  LDL may be viewed as a more general learning framework which includes both SLL and MLL as its special cases.



## 4 LDL Algorithms
  
- 3 strategies
  - problem transformation, transform the LDL problem into existing learning paradigms
  - algorithm adaptation, extend existing learning algorithms to deal with label distributions
  - design specialized algorithms according to the characteristics of LDL

### 4.1 Problem Transformation

- One straightforward way to transform an LDL problem into an SLL problem is to change the training examples into weighted single-label examples
  - $(x_i,D_i)$ -> $c$ single-label examples $(x_i,y_j)$ with the weight $d_{x_i}^{y_j}$
  - The resampled training set includes $c\times n$ examples
  - Any SLL algorithms can be applied to the training set.


- Two representative algorithms are adopted here for this purpose. **Bayes classifier and SVM**.
  - **Bayes classifier** assumes Gaussian distribution for each class, and the posterior probability computed by the Bayes rule is regarded as the description degree of the corresponding label
  - As to **SVM**, the probability estimates are obtained by a pairwise coupling multi-class method [47], where the probability of each binary SVM is calculated by an improved implementation of Platt’s posterior probabilities [28], and the class probability estimates are obtained by solving a linear system whose solution is guaranteed by the theory in finite Markov Chains
  - **PT-Bayes and PT-SVM**， ‘PT’ is the abbreviation of ‘Problem Transformation’


### 4.2 Algorithm Adaptation

- k-NN -> AA-kNN
  -  Given a new instance $x$, its $k$ nearest neighbors are first found in the training set. 
  -  Then, the mean of the label distributions of all the $k$ nearest neighbors is calculated as the label distribution of $x$
  ![图 5](fig/Label%20Distribution%20Learning/Label%20Distribution%20Learning_5.png)  
  
- Backpropagation (BP) neural network -> AA-BP
  - The target of the BP algorithm is to minimize the sum-squared error of the output of the neural network compared with the real label distributions


### 4.3 Specialized Algorithms

- SA-IIS
  - our previous work on facial age estimation [19], [20]
  - the key step was to solve an optimization problem similar to Eq. (1).

-  SA-BFGS


## 5 Experiments

- Evaluation Measures
  - Chebyshev distance (Cheb), 
  - Clark distance (Clark),
  - Canberra metric (Canber), 
  - Kullback-Leibler divergence (KLdiv), 
  - cosine coefficient (Cosine), 
  - intersection similarity (Intersec)

- Datasets
  - 16, an artificial toy dataset and 15 real-world datasets
