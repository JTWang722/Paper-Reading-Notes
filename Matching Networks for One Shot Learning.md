# Matching Networks for One Shot Learning

jtwang  2023/1/4

> Vinyals O, Blundell C, Lillicrap T, et al. Matching networks for one shot learning[J]. Advances in neural information processing systems, 2016, 29.
> Paper link: https://proceedings.neurips.cc/paper/2016/file/90e1357833654983612fb05e3ec9148c-Paper.pdf


## Abstract

In this work, we employ ideas from **metric learning** based on deep neural features and from recent advances that augment neural networks with **external memories**.

## Introduction

- One-shot learning
- We aim to incorporate the best characteristics from both parametric and non-parametric models – namely, rapid acquisition of new examples while providing excellent generalisation from common examples.
- The novelty of our work is twofold: at the **modeling** level, and at the **training** procedure
  - Matching Nets,  a neural network which uses recent advances in attention and memory that enable rapid learning
  - test and train conditions must match


## Model

### Model Architecture

- We draw inspiration from models such as sequence to sequence (seq2seq) with attention [2], memory networks [29] and pointer networks 