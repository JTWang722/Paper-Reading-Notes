# Matching Networks for One Shot Learning

jtwang 2022/11

> Vinyals O, Blundell C, Lillicrap T, et al. Matching networks for one shot learning[J]. Advances in neural information processing systems, 2016, 29.
> 论文链接：https://proceedings.neurips.cc/paper/2016/file/90e1357833654983612fb05e3ec9148c-Paper.pdf


划分小样本训练任务，episode。我现在的理解是
- N-way K-shot
- 每个任务包含N个类别，其中每个类别包含K个样本
- 每个任务采样support set和query set
- 训练任务和测试任务标签不重合，也就是说在meta-testing阶段，所有测试任务的标签都是模型没见过的，只能利用每个测试任务的support set去预测query set的标签
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        