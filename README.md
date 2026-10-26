# 光计算仿真项目

项目名称：**Simulation of Optical Computing to Support Vector/Tensor Computation**。

当前完成第二阶段：输入目标酉矩阵 U，自动用 Clements 算出 MZI 网格参数，逐器件传播复数光场 x，验证输出等于 U @ x。第一阶段的单个 MZI 与示例仍保留。

## 统一环境

本机验证环境：Python 3.12.4、NumPy 2.4.1、pytest 8.3.5。
PyCharm 是编辑器，Python 是运行环境，NumPy 负责复数矩阵运算。
本周不依赖 PyTorch，神经网络扩展时再确定其版本。

本机已经建立 `.venv`；组员拿到代码后需要自己建立环境，不复制 `.venv`。

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python demo.py
python demo_week02.py
python -m pytest -q
python verify_week02.py
```

Windows 激活命令为 `.venv\Scripts\activate`。

## 在 PyCharm 运行

1. 用 Open 打开下载或克隆后的项目根文件夹（包含 README.md 的目录）。
2. 在项目 Python Interpreter 中选择该文件夹的 `.venv/bin/python`。
3. 打开 `demo_week02.py`，点击 Run。运行目录应为项目根目录；第一阶段示例为 `demo.py`。
4. 在项目终端运行 `python -m pytest -q` 检查测试。

## 代码入口

- `src/mzi.py`：唯一 MZI 矩阵实现、双光场接口、相邻通道接口。
- `demo.py`：固定输入与混合角，改变相位，观察光场和功率。
- `src/clements.py`：自动把酉矩阵分解成相邻通道 MZI 和输出相位。
- `src/mesh.py`：逐器件传播、矩阵重建和矩形网格分层。
- `demo_week02.py`：显示 4×4 完整示例；可用 `--size 8 --seed 42` 改尺寸和种子。
- `verify_week02.py`：批量验证 2、4、8 通道，写入第二阶段结果报告。
- `tests/test_mzi.py`：端口、符号、相位、干涉、酉性、功率、逆变换与输入检查。
- `tests/test_mesh.py`、`tests/test_clements.py`：传播顺序、完整相位、自动分解、边界条件与矩形布局。
- `docs/technical-standards.md`：交给 AI 的统一数学规则。
- `docs/clements-decomposition.md`：第二阶段接口、算法推导和验收说明。

输入 `[1,1]/sqrt(2)`，theta=pi/4：phi=0 的输出约为 `[0,1]`，phi=pi 的输出约为 `[-1,0]`。微小的 1e-16 量级残差属于浮点误差。每种情况总功率均为 1。

## 第二阶段最小调用

```python
import numpy as np
from src.clements import decompose_clements
from src.mesh import propagate_mesh

U = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
x = np.array([1, 1j], dtype=complex)
ops, phases = decompose_clements(U)
y = propagate_mesh(x, ops, phases)
print(y, U @ x)
```

U 必须是酉矩阵。分解结果的列表按光的传播顺序保存，输出端相位最后施加；theta 允许为负。详见 [统一标准](docs/technical-standards.md)。

## 实施顺序

1. 已完成：单个 MZI、复数光场表示和器件测试。
2. 本周已完成：连接相邻通道单元，实现 Clements 分解与输出端相位；验证 2、4、8 维酉矩阵。
3. 再下一步：SVD 实现 W=UΣV†，传播顺序为 V† → Σ → U，验证 Wx。
4. 最后：引入损耗、相位误差和量化；理想结果正确后才做 AI 扩展。

当前为理想无损传输矩阵仿真；SVD、非理想因素与 AI 扩展尚未实现。无需另外安装电磁场仿真软件即可运行本阶段，但当前模型不计算波导内部的空间电磁场。

## 项目目录与进度

| 目录 | 内容 | 状态 |
|---|---|---|
| src/ | 可复用的仿真源码 | MZI、网格与 Clements 已完成 |
| tests/ | 自动测试 | 61 项通过 |
| docs/ | 全组与 AI 使用的技术标准 | 已确定 |
| progress/01_single_mzi/ | 第一阶段验收记录 | 已完成 |
| progress/02_clements_mesh/ | Clements 网格记录与实验报告 | 已完成 |
| progress/03_svd_matmul/ | SVD 与 Wx 验证计划 | 待开始 |
| progress/04_nonidealities/ | 器件误差研究计划 | 待开始 |
| progress/05_ai_extension/ | AI 扩展计划 | 待开始 |

目录编号表示实施顺序，不表示已承诺的周数。统一标准集中维护在 docs/technical-standards.md，进度记录与实际代码分开存放。第二阶段的 312 个矩阵均通过验收，最大矩阵重建相对误差约为 1.46×10⁻¹⁵（不同平台的末位可能变化）。
