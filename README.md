# ANN
实现一些基础的神经网络算法

## Features

- 除了依赖的 Eigen 矩阵运算库以外，只包含一个头文件 `/include/net2.h`
- 接口简单，易于使用
- 实现了 线性层，BN层，sigmoid，relu，hardswish，softmax 等
- 支持 mini-batch 和 adam
- 可扩展性（可以自行添加神经网络层）

## Todos
- [x] 网络导入导出
- [x] 封装优化器
- [ ] 3d 卷积层（tensor -> matrix）
- [ ] max pooling 层
