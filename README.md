# 光计算仿真项目

项目名称：**Simulation of Optical Computing to Support Vector/Tensor Computation**。

当前完成第三阶段：输入一般实数/复数方阵 W，自动用 SVD 与 Clements 配置两套 MZI 网格及幅度衰减器，逐器件传播复数光场 x，恢复比例后验证输出等于 W @ x。前两阶段的代码与示例仍保留。

**本周入口：打开 `demo_week03.py`，在 PyCharm 点击 Run。** 示例显示中间光场、衰减系数、实际光学输出、数字恢复后的结果与误差。

## 环境与运行

验证环境：Python 3.12.4、NumPy 2.4.1、pytest 8.3.5。PyCharm 是编辑器，Python 是运行环境，NumPy 负责复数矩阵运算。本阶段没有新增依赖，不需要 PyTorch 或电磁场仿真软件。

组员拿到代码后自行建立虚拟环境，不复制其他人的 `.venv`：

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python demo_week03.py
python -m pytest -q
python verify_week03.py
```

Windows 激活命令为 `.venv\Scripts\activate`。完全固定所有测试依赖版本时使用 `requirements-lock.txt`。

在 PyCharm 用 Open 打开包含本 README 的项目根目录，在 Python Interpreter 中选择该项目的虚拟环境（本机为 `.venv/bin/python`），打开 `demo_week03.py` 点击 Run，运行目录设为项目根目录。之前的示例仍为 `demo.py` 和 `demo_week02.py`。

## 第三阶段最小调用

```python
import numpy as np
from src.svd_matmul import compile_svd, propagate_svd

W = np.array([[1, 2], [3, 4]], dtype=complex)
x = np.array([1, -1], dtype=complex)
circuit = compile_svd(W)       # 同一个 W 可重复用于不同 x
print(propagate_svd(x, circuit), W @ x)  # 都约为 [-1, -1]
```

W 可以是非酉、零或不满秩方阵。`compile_svd` 内部把 W 分成 UΣV†，仅把其中的酉矩阵交给 `decompose_clements`；后者仍拒绝非酉输入。

光路顺序为 **V† 网格 → diag(sigma/s) 幅度衰减 → U 网格**，s=max(1,sigma_max)。原始光学输出是 Wx/s，默认接口再数字乘 s 恢复结果。`propagate_svd(x, circuit, restore_scale=False)` 返回原始光场。s 不是光学放大器，衰减系数作用于幅度，功率按其平方变化。

完整数学约定见 [统一标准](docs/technical-standards.md)，运行方法、接口与验收判据见 [SVD 矩阵乘法说明](docs/svd-matmul.md)。

## 代码入口

| 文件 | 用途 |
|---|---|
| src/mzi.py | 固定约定的单个 MZI 和相邻通道操作 |
| src/clements.py、src/mesh.py | 酉矩阵自动分解、逐器件传播与重建 |
| src/svd_matmul.py | 一般方阵编译、两套网格与衰减器传播、比例恢复 |
| src/svd_validation.py | 统一误差、零或近零结果与功率验收 |
| demo_week03.py | 4×4 复数示例；`--size 2 --real` 切换为 2×2 实数 |
| verify_week03.py | 验证 2、4、8 通道一般方阵并生成报告 |
| tests/ | 96 项测试，覆盖前三阶段和配置复用 |
| demo.py、demo_week02.py、verify_week02.py | 前两阶段的示例与批量验证 |
| docs/technical-standards.md | 全组交给 AI 的统一数学规则 |

## 进度与范围

| 目录 | 内容 | 状态 |
|---|---|---|
| progress/01_single_mzi/ | 单个 MZI 与数学约定 | 已完成 |
| progress/02_clements_mesh/ | Clements 酉矩阵网格与报告 | 已完成 |
| progress/03_svd_matmul/ | 一般方阵乘法、完整示例与报告 | 已完成 |
| progress/04_nonidealities/ | 损耗、相位误差、量化 | 待开始 |
| progress/05_ai_extension/ | 光学线性层与 AI 对照 | 待开始 |

第三阶段共 321 个矩阵、2,140 组输入通过验收：最大矩阵重建相对误差约 4.24×10⁻¹⁵；按相对判据检查的输出最大误差约 9.25×10⁻¹⁵，零或近零输出另按绝对误差检查。详见 [第三阶段记录](progress/03_svd_matmul/README.md)。这些是浮点数值误差，不是真实光器件测量误差；不同平台末位可能变化。

当前为理想传输矩阵仿真：MZI 无损，中间衰减器按目标矩阵进行理想幅度缩放。额外器件损耗、相位噪声、量化与 AI 扩展尚未实现；长方形矩阵和张量接口尚未实现。当前模型不计算波导内部的空间电磁场，也不测量硬件速度或能耗。
