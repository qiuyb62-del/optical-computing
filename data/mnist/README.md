# MNIST 数据来源

MNIST 手写数字由 Yann LeCun、Corinna Cortes、Christopher J. C. Burges 整理。引用：Y. LeCun, L. Bottou, Y. Bengio, P. Haffner, “Gradient-Based Learning Applied to Document Recognition,” Proceedings of the IEEE, 86(11), 2278–2324, 1998。

- [原始数据介绍](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow 官方数据目录](https://www.tensorflow.org/datasets/catalog/mnist)
- [torchvision v0.24.1 MNIST loader](https://github.com/pytorch/vision/blob/v0.24.1/torchvision/datasets/mnist.py)

本项目使用官方 IDX 格式的 60,000 张训练、10,000 张测试图片。数据来自 torchvision 使用的 HTTPS 镜像 https://ossci-datasets.s3.amazonaws.com/mnist/，下载器按其固定 MD5 校验完整性，再记录 SHA-256。来源归属保留，不把图片声明为本项目原创。

| 压缩文件 | 固定 MD5 |
|---|---|
| train-images-idx3-ubyte.gz | f68b3c2dcbeaaa9fbdd348bbdeb94873 |
| train-labels-idx1-ubyte.gz | d53e105ee54ea40749a09fcbcd1e9432 |
| t10k-images-idx3-ubyte.gz | 9fb629c4189551a2d022fa330f9573f3 |
| t10k-labels-idx1-ubyte.gz | ec29112dd5afa0611ce80d1b7f02629c |

首次完整实验运行 `python verify_week06.py --download`，数据缓存到本目录的 `raw/`（不提交 Git）。完整文件来源、字节数、校验值在第六阶段的 data_provenance.json 中。仓库只附带官方测试集中每类最先出现的两张图片供离线演示，原始索引一同保存。
