# 第一阶段 单个 MZI

状态：已完成

实现双光场输入输出、矩阵接口与相邻通道操作。19 项测试通过。

## 复现

```sh
python demo.py
python -m pytest -q
```

代码见 ../../src/mzi.py，测试见 ../../tests/test_mzi.py。
