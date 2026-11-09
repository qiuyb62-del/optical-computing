# 光计算仿真项目

项目名称：**Simulation of Optical Computing to Support Vector/Tensor Computation**。

当前完成第五阶段 AI 验证：训练 4→4→3 的 Iris 分类器，将第一层 4×4 权重编译为 SVD/Clements 光路，对比同一模型的数字推理、理想光学推理与静态器件误差。前四阶段的代码、示例和结果仍保留。

**最新入口：打开 `demo_week05.py`，在 PyCharm 点击 Run。** 示例直接使用已保存的模型，无需训练或联网；查看 [AI 结果与曲线](progress/05_ai_extension/README.md)。5 组固定种子的数字与理想光路平均准确率均为 97.33%；组合误差条件平均为 95.91%。这些是 Iris 小样本任务的 Python 仿真结果。

## 环境与运行

验证环境：Python 3.12.4、NumPy 2.4.1、pytest 8.3.5。第五阶段训练使用 PyTorch 2.9.1，绘图使用 Matplotlib 3.10.7；CPU 即可，不要求显卡。PyCharm 是编辑器，Python 是运行环境。前四阶段和第五阶段保存模型的快速演示只需要原有依赖；Lumerical 尚未执行。

组员拿到代码后自行建立虚拟环境，不复制其他人的 `.venv`：

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python demo_week05.py
python -m pytest -q
```

完整重跑 AI 训练和 9 组误差实验：

```sh
python -m pip install -r requirements-ai.txt
python verify_week05.py
python -m pytest -q
```

`requirements-ai-lock.txt` 保存本次 macOS CPU 环境的依赖版本。Windows/Linux 如果只使用 CPU，可先按照 PyTorch 官方方式安装 CPU 版 torch 2.9.1，再安装上述 requirements。训练和绘图只需安装 Python 包，无需下载单独的 PyTorch 应用。

Windows 激活命令为 `.venv\Scripts\activate`。固定基础依赖版本使用 `requirements-lock.txt`；完整 AI 环境见 `requirements-ai-lock.txt`。未安装 PyTorch 时，训练相关测试会明确显示跳过。

在 PyCharm 用 Open 打开包含本 README 的项目根目录，在 Python Interpreter 中选择该项目的虚拟环境，打开 `demo_week05.py` 点击 Run。代码也支持从其他工作目录启动。之前的示例是 `demo.py`、`demo_week02.py`、`demo_week03.py`、`demo_week04.py`。

快速检查批量脚本可用 `python verify_week04.py --samples 2 --trials 2 --output-dir scratch_week04`。使用单独输出目录，避免覆盖已保存的完整实验。

第五阶段快速开发检查：`python verify_week05.py --seeds 7 --trials 2 --max-epochs 10 --patience 5 --output-dir scratch_week05`。这是冒烟检查，不是正式结果。

## AI 实际计算流程

4 个花卉特征 → 仅由训练集拟合的标准化 → 光学 W1x → 数字偏置与 ReLU → 数字分类头 → 三类预测。

光学 W1x 内部仍是 V† 网格 → 被动衰减 → U 网格。保存的权重只编译一次，同一静态器件处理全部测试样本。使用共同相位参考下的复光场实部，而非强度；残余虚部单独记录。此次没有校准或噪声训练，不声称全光神经网络。

完整协议和边界见 [AI 验证说明](docs/ai-validation.md)。

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
| src/ai_data.py | 固定 Iris 数据、分层划分、训练集标准化 |
| src/ai_training.py | PyTorch CPU 训练、验证集早停、可读取的 JSON 权重 |
| src/ai_inference.py | 数字/光学第一层、数字分类头、完整复场与分类指标 |
| demo_week05.py | 离线重放保存模型，打印三组分类结果和具体样本 |
| verify_week05.py、src/ai_plots.py | 五组训练、配对误差扫描、逐器件记录及图表 |
| demo_week04.py | 理想与非理想 4×4 对照；保存输入、器件快照和输出 |
| verify_week04.py | 配对扫描、理想回归、被动性检查及 JSON/CSV/SVG 结果 |
| demo_week03.py | 4×4 复数示例；`--size 2 --real` 切换为 2×2 实数 |
| verify_week03.py | 验证 2、4、8 通道一般方阵并生成报告 |
| tests/ | 安装 AI 依赖后 170 项测试通过；含数据划分、训练集标准化、权重方向、光路复用与模型重放 |
| demo.py、demo_week02.py、verify_week02.py | 前两阶段的示例与批量验证 |
| docs/technical-standards.md | 全组交给 AI 的统一数学规则 |

## 进度与范围

| 目录 | 内容 | 状态 |
|---|---|---|
| progress/01_single_mzi/ | 单个 MZI 与数学约定 | 已完成 |
| progress/02_clements_mesh/ | Clements 酉矩阵网格与报告 | 已完成 |
| progress/03_svd_matmul/ | 一般方阵乘法、完整示例与报告 | 已完成 |
| progress/04_nonidealities/ | 静态相位误差、损耗、量化、耦合比与完整扫描 | Python 部分已完成 |
| progress/05_ai_extension/ | Iris 分类、光学第一层、固定误差对照、模型与图表 | Python 部分已完成 |

第三阶段共 321 个矩阵、2,140 组输入通过验收：最大矩阵重建相对误差约 4.24×10⁻¹⁵；按相对判据检查的输出最大误差约 9.25×10⁻¹⁵，零或近零输出另按绝对误差检查。详见 [第三阶段记录](progress/03_svd_matmul/README.md)。这些是浮点数值误差，不是真实光器件测量误差；不同平台末位可能变化。

第四阶段在 60 个独立目标矩阵上扫描 16 组条件，每个条件/矩阵有 30 次器件实现，共 28,800 个条件内器件配置和 192,000 组输入；另外验证 21 个结构化理想矩阵。保存的实验通过零误差回归和被动性检查。非理想输出超过理想数值阈值是实验结果，不表示软件出错，也不等于 AI 分类错误率。

第五阶段使用 5 个预设模型/划分种子，各自训练/验证/测试为 90/30/30。9 个非理想条件各运行 30 个器件 seed，总计 1,350 个条件内器件配置、40,500 次非理想测试预测。理想与数字的全部测试预测一致；组合误差平均准确率下降约 1.42 个百分点。纯量化、损耗和耦合器条件中的重复 seed 是确定性重复，不是独立制造样本。

当前仍是 Python 传输矩阵仿真，器件误差是人为指定的模型参数，不是真实测量。Lumerical、动态噪声、校准优化、实际光电读出、长方形矩阵和通用张量接口尚未实现。按目前安排，暂不加入校准，Lumerical 仿真留到后面统一进行。
