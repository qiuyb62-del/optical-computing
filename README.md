# 光计算仿真项目

项目名称：**Simulation of Optical Computing to Support Vector/Tensor Computation**。

当前完成第四阶段：在 SVD 与 Clements 光学矩阵乘法上加入静态相位误差、器件损耗、物理相位量化和耦合器偏差，比较同一个 W、x 的理想与非理想输出。前三阶段的代码与示例仍保留。

**本周入口：打开 `demo_week04.py`，在 PyCharm 点击 Run。** 看完示例后，查看[本周结果与曲线](progress/04_nonidealities/README.md)；`verify_week04.py` 可重跑完整扫描。

## 环境与运行

验证环境：Python 3.12.4、NumPy 2.4.1、pytest 8.3.5。PyCharm 是编辑器，Python 是运行环境，NumPy 负责复数矩阵运算。本周没有新增依赖。图表由脚本直接生成 SVG，不要求安装 Matplotlib；PyTorch 和 Lumerical 留在后续阶段。

组员拿到代码后自行建立虚拟环境，不复制其他人的 `.venv`：

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python demo_week04.py
python -m pytest -q
python verify_week04.py
```

Windows 激活命令为 `.venv\Scripts\activate`。完全固定所有测试依赖版本时使用 `requirements-lock.txt`。

在 PyCharm 用 Open 打开包含本 README 的项目根目录，在 Python Interpreter 中选择该项目的虚拟环境，打开 `demo_week04.py` 点击 Run，运行目录设为项目根目录。之前的示例是 `demo.py`、`demo_week02.py` 和 `demo_week03.py`。

快速检查批量脚本可用 `python verify_week04.py --samples 2 --trials 2 --output-dir scratch_week04`。使用单独输出目录，避免覆盖已保存的完整实验。

## 第四阶段最小调用

```python
import numpy as np
from src.svd_matmul import compile_svd
from src.nonideal import NonidealConfig, realize_nonideal, propagate_nonideal

W = np.array([[1, 2j], [3, 4]], dtype=complex)
x = np.array([1, -1j])
ideal = compile_svd(W)
config = NonidealConfig(phase_std_rad=0.01, loss_db_per_mzi=0.1, phase_bits=8)
device = realize_nonideal(ideal, config, seed=20261109)
print(propagate_nonideal(x, device), W @ x)
```

`realize_nonideal` 采样一次静态器件；重复调用传播不会重新产生随机误差。所有默认误差为零时恢复前三阶段的理想结果。控制相位的准确映射、误差含义、接口和验收见[非理想模型说明](docs/nonidealities.md)。

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
| src/nonideal.py | 物理控制相位映射、静态器件误差、量化、损耗、局部传播 |
| demo_week04.py | 理想与非理想 4×4 对照；保存输入、器件快照和输出 |
| verify_week04.py | 配对扫描、理想回归、被动性检查及 JSON/CSV/SVG 结果 |
| demo_week03.py | 4×4 复数示例；`--size 2 --real` 切换为 2×2 实数 |
| verify_week03.py | 验证 2、4、8 通道一般方阵并生成报告 |
| tests/ | 152 项测试通过，覆盖前四阶段、完整复相位、解析损耗、量化边界、静态复用和复现 |
| demo.py、demo_week02.py、verify_week02.py | 前两阶段的示例与批量验证 |
| docs/technical-standards.md | 全组交给 AI 的统一数学规则 |

## 进度与范围

| 目录 | 内容 | 状态 |
|---|---|---|
| progress/01_single_mzi/ | 单个 MZI 与数学约定 | 已完成 |
| progress/02_clements_mesh/ | Clements 酉矩阵网格与报告 | 已完成 |
| progress/03_svd_matmul/ | 一般方阵乘法、完整示例与报告 | 已完成 |
| progress/04_nonidealities/ | 静态相位误差、损耗、量化、耦合比与完整扫描 | Python 部分已完成 |
| progress/05_ai_extension/ | 光学线性层与 AI 对照 | 待开始 |

第三阶段共 321 个矩阵、2,140 组输入通过验收：最大矩阵重建相对误差约 4.24×10⁻¹⁵；按相对判据检查的输出最大误差约 9.25×10⁻¹⁵，零或近零输出另按绝对误差检查。详见 [第三阶段记录](progress/03_svd_matmul/README.md)。这些是浮点数值误差，不是真实光器件测量误差；不同平台末位可能变化。

第四阶段在 60 个独立目标矩阵上扫描 16 组条件，每个条件/矩阵有 30 次器件实现，共 28,800 个条件内器件配置和 192,000 组输入；另外验证 21 个结构化理想矩阵。保存的实验通过零误差回归和被动性检查。非理想输出超过理想数值阈值是实验结果，不表示软件出错，也不等于 AI 分类错误率。

当前仍是 Python 传输矩阵仿真，器件误差是人为指定的模型参数，不是真实测量。Lumerical、动态噪声、标定优化、光电读出、AI、长方形矩阵和张量接口尚未实现。按目前安排，Lumerical 仿真留到后面统一进行。
