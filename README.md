# 数据集处理

## 切分验证集

`train_db, val_db = torch.utils.data.random_split(train_db, [50000, 10000])`

从训练集中分割出验证集，[代码参考]()