# 光计算仿真项目

本周交付：单个理想 MZI，输入两个复数光场，输出两个复数光场。

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
python -m pytest -q
```

Windows 激活命令为 `.venv\Scripts\activate`。

## 在 PyCharm 运行

1. 用 Open 打开下载或克隆后的项目根文件夹（包含 demo.py 的目录）。
2. 在项目 Python Interpreter 中选择该文件夹的 `.venv/bin/python`。
3. 打开 `demo.py`，点击 Run。运行目录应为项目根目录。
4. 在项目终端运行 `python -m pytest -q` 检查测试。

## 本周代码

- `src/mzi.py`：唯一 MZI 矩阵实现、双光场接口、相邻通道接口。
- `demo.py`：固定输入与混合角，改变相位，观察光场和功率。
- `tests/test_mzi.py`：端口、符号、相位、干涉、酉性、功率、逆变换与输入检查。
- `docs/technical-standards.md`：交给 AI 的统一数学规则。

输入 `[1,1]/sqrt(2)`，theta=pi/4：phi=0 的输出约为 `[0,1]`，phi=pi 的输出约为 `[-1,0]`。微小的 1e-16 量级残差属于浮点误差。每种情况总功率均为 1。

## 后续顺序

1. 本周：确认单个 MZI 和测试正确，理解振幅与功率的区别。
2. 下一步：连接相邻通道单元，实现 Clements 分解与输出端相位；验证 2、4、8 维酉矩阵。
3. 再下一步：SVD 实现 W=UΣV†，传播顺序为 V† → Σ → U，验证 Wx。
4. 最后：引入损耗、相位误差和量化；理想结果正确后才做 AI 扩展。

Clements 与 SVD 尚未实现，不能把单个 MZI 测试通过当作整个光计算系统已完成。

## 项目目录与进度

| 目录 | 内容 | 状态 |
|---|---|---|
| src/ | 可复用的仿真源码 | 单个 MZI 已完成 |
| tests/ | 自动测试 | 19 项通过 |
| docs/ | 全组与 AI 使用的技术标准 | 已确定 |
| progress/01_single_mzi/ | 第一阶段验收记录 | 已完成 |
| progress/02_clements_mesh/ | Clements 网格计划 | 待开始 |
| progress/03_svd_matmul/ | SVD 与 Wx 验证计划 | 待开始 |
| progress/04_nonidealities/ | 器件误差研究计划 | 待开始 |
| progress/05_ai_extension/ | AI 扩展计划 | 待开始 |

目录编号表示实施顺序，不表示已承诺的周数。统一标准集中维护在 docs/technical-standards.md，进度记录与实际代码分开存放。
