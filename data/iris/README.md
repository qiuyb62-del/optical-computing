# Iris 数据来源

来源：R. A. Fisher (1936), Iris [Dataset], UCI Machine Learning Repository。

- 数据页面：https://archive.ics.uci.edu/dataset/53/iris
- DOI：https://doi.org/10.24432/C56C76
- 下载地址：https://archive.ics.uci.edu/static/public/53/iris.zip
- 获取日期：2026-11-09。
- 使用压缩包中的 `bezdekIris.data`，逐字节保留，无清洗、打乱或手工修改。运行时离线读取。
- SHA-256：`0fed2a99db77ec533a62dc66894d3ec6df3b58b6a8f3cf4a6b47e4086b7f97dc`。
- 许可：[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。共享时保留来源和许可说明。

共 150 行有效数据，每类 50 行；4 个特征依次为花萼长、花萼宽、花瓣长、花瓣宽，单位 cm。三个标签为 Iris-setosa、Iris-versicolor、Iris-virginica。

特意固定为 bezdekIris 版本：UCI 同时提供的旧 iris.data 在第 35、38 行存在差异，UCI 数据说明列出了相应更正。不要与其他库的 Iris 版本混用或替换文件而不更新实验。

标准化在划分之后仅用训练集拟合；原始数据文件不含标准化值。报告、模型和演示都记录此文件的哈希值。
