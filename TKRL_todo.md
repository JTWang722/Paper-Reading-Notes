## Representation Learning of Knowledge Graphs with Hierarchical Types

> Xie R, Liu Z, Sun M. Representation Learning of Knowledge Graphs with Hierarchical Types[C]//IJCAI. 2016: 2965-2971.



- Hierarchical Type Encoders
  - Recursive Hierarchy Encoder
  - Weighted Hierarchy Encoder

<img src=".\fig\TKRL\1.png" style="zoom:50%;" />

- Type Information as Constraints
  - 训练时, 同类型的实体更可能被负采样到
  - 评估时, 对于关系 r, 只保留 domain/range 类型的实体