# 第三阶段：一般方阵的光学矩阵乘法

目标：输入实数或复数方阵 W 和复数列向量 x，自动配置两套 MZI 网格及中间衰减器，逐器件传播得到 y≈Wx。支持零矩阵、不满秩矩阵和 N=1。当前不接收长方形矩阵、批量张量或神经网络层。

## 数学与物理约定

NumPy 的 `U, sigma, Vh = np.linalg.svd(W, full_matrices=False)` 满足

```text
W = U @ diag(sigma) @ Vh
Vh = V†（已经是共轭转置，不能再转置）
```

对 U 和 Vh 分别调用现有 Clements 分解，完整保留各自输出端相位。MZI 矩阵约定、通道编号、complex128 和 rad 均沿用统一标准。

被动衰减器只允许幅度透过率处于 [0,1]。本项目规定：

```text
s = max(1, sigma_max)
t_i = sigma_i / s

x → V† 网格 → diag(t) 幅度衰减 → U 网格 → y_optical
y_optical ≈ (W/s) @ x
y = s * y_optical ≈ W @ x
```

最后乘 s 是数字恢复数值，不是光学放大。s 必须随器件参数保存，不得省略。零矩阵的 s=1、所有 t_i=0，不会除以零；小的非零奇异值不人为截断。

这里 t_i 乘的是复数光场。通道功率乘 t_i²，例如 t=0.5 表示场幅度减半、功率变为 1/4。两套理想网格各自保持总功率，衰减器可以降低总功率，整条被动光路输出功率不能超过输入。数字恢复后的数值模平方不作为实际输出光功率。

这是理想传输矩阵仿真；不建模激光器、调制器、探测器、硬件增益或噪声。真实硬件读取完整复数场还需要相应的相干读出，普通强度探测不能直接得到复数 y。

## 接口与复用

```python
import numpy as np
from src.svd_matmul import compile_svd, propagate_svd, trace_svd

W = np.array([[1, 2], [3, 4]], dtype=complex)
circuit = compile_svd(W)       # W 不变时只需做一次
x = np.array([1, -1], dtype=complex)
y = propagate_svd(x, circuit)  # 约为 [-1, -1]
raw = propagate_svd(x, circuit, restore_scale=False)
steps = trace_svd(x, circuit)   # 查看每一步的复数光场
```

- `compile_svd(W)`：验证有限非空方阵，做 SVD，将 U/V† 编译为 MZI 参数，返回 `SVDCircuit`。每套网格有 N(N−1)/2 个 MZI，两套共 N(N−1) 个。
- `SVDCircuit`：只读配置，使用 tuple 保存参数；包含 `input_ops`、`input_phases`、`singular_values`、`amplitude_transmissions`、`recovery_scale`、`output_ops`、`output_phases`。通过 `to_dict()` 导出 JSON 数据。请用 `compile_svd` 创建配置。
- `propagate_svd(x, circuit, restore_scale=True)`：逐器件计算，默认返回数字恢复后的 y；`False` 返回原始光学输出。
- `trace_svd(x, circuit)`：返回 `after_input_mesh`、`after_attenuation`、`optical_output` 和 `restored_output` 四个独立数组。第一套网格的输出相位在衰减前施加，第二套网格的输出相位在数字恢复前施加。
- `reconstruct_svd(circuit, restore_scale=True)`：仅供验收，从器件参数重建 W；`False` 重建 W/s。传播不会调用它，也不会使用 `W @ x` 或缓存稠密 U/V† 代替网格。

所有输入不被修改，不自动归一化 x。矩阵尺寸与输入长度必须一致。配置只依赖 W，可重复用于不同 x。有限精度可能让不同平台返回不同但等价的 SVD 因子，验收比较最终矩阵与光场，不比较参数逐项相同。

## 验收定义

1. 矩阵重建：`||W_hat-W||_F / ||W||_F <= 1e-10`。W=0 时检查绝对误差 ≤1e-10。
2. 输出：普通结果检查 `||y-Wx||_2 / ||Wx||_2 <= 1e-10`。完整比较复数，不对齐全局相位。
3. 零或近零输出：令 B=`||W||_F * ||x||_2`。当 `||Wx||_2=0` 或 `||Wx||_2 <= 1e-12*B`，检查 `||y-Wx||_2 <= 1e-10*max(1,B)`。记录实际绝对误差与采用的限值；零分母的相对误差记为 JSON `null`。这避免在零空间附近用极小分母放大浮点残差。整体很小但没有相消的结果仍用相对误差。
4. 原始光学输出：同样比较 `y_optical` 与 `(W/s)@x`，其中参考尺度改为 B/s。
5. 功率：两套网格各自守恒，衰减后不增加，完整光路不增加；比较容差为输入功率的 1e-10。零输入要求各阶段严格为零。

`src/svd_validation.py` 统一上述定义；检查函数只用于验收，不参与光学计算。批量报告的最大相对误差只统计使用相对判据的样本，近零样本另外报告数量及最大绝对误差，不能混称为全部样本的相对精度。

## 运行与文件

在 PyCharm 打开根目录中的 `demo_week03.py` 并点击 Run，即可查看 4×4 复数示例的每一步。也可在项目终端运行：

```sh
python demo_week03.py --size 4 --seed 42 --report progress/03_svd_matmul/example_4x4.json
python demo_week03.py --size 2 --real
python verify_week03.py
python -m pytest -q
```

示例 seed=42；批量验证 seed=20261102。种子用于复现输入数据，不改变算法。每个尺寸 100 个随机矩阵（实数/复数交替；整体系数 0.01、1、100 轮换）及 7 个特殊矩阵，每个矩阵复用配置验证 N 个基向量、一个随机复向量和零向量。目标由 NumPy 随机数、手工矩阵生成，不由 MZI 反向生成。

结果保存在 `progress/03_svd_matmul/`；`results.json` 保存汇总与失败定位，`example_4x4.json` 保存完整参数、输入、中间光场和参考答案。报告只说明理想模型在浮点运算中的正确性，不代表真实器件测量精度或硬件速度。

## 资料

- [NumPy SVD 文档](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)：确认返回值 U、sigma、Vh 及重建顺序。
- [Clements 等，2016](https://doi.org/10.1364/OPTICA.3.001460)：通用酉变换的矩形干涉仪网格。
- [Shen 等，Deep Learning with Coherent Nanophotonic Circuits](https://arxiv.org/abs/1610.02365)：相干光子线性运算背景。
