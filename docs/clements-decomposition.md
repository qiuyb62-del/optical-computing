# 第二周：自动分解与网格传播

给定一个 N×N 酉矩阵 U，程序自动算出每个 MZI 的位置、theta、phi，以及最后 N 个输出端相位。然后把输入光场逐个送过这些器件，与 NumPy 的 U @ x 比较。这里的“自动”是确定性的数值算法，不需要 AI 猜相位，也不需要训练。

## 调用方法

```python
import numpy as np
from src.clements import decompose_clements
from src.mesh import propagate_mesh, reconstruct_mesh, mesh_layers

U = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
x = np.array([1, 1j], dtype=complex)
ops, output_phases = decompose_clements(U)
y = propagate_mesh(x, ops, output_phases)
Uhat = reconstruct_mesh(ops, output_phases)
print(y, U @ x)
```

- `decompose_clements(U)`：返回 `(ops, output_phases)`。U 必须为非空方阵且满足 `||U†U−I||_F ≤ 1e−10`；不满足时抛出 ValueError，不偷偷修改目标矩阵。
- `ops`：Python 列表，元素为 `(m, theta, phi)`，依次作用于相邻通道 `(m,m+1)`。长度为 N(N−1)/2，包含零混合角单元。
- `output_phases`：shape=(N,) 的实数数组。所有分解结果的 phi、输出端相位在 [0,2π)；theta 允许为负，不按 π 取模。
- `propagate_mesh(x, ops, output_phases)`：按列表顺序逐器件传播，最后乘输出相位，返回新的 complex128 光场数组，不修改输入、不归一化。
- `reconstruct_mesh(...)`：重建完整矩阵，主要用于验证。
- `mesh_layers(...)`：把互不占用同一通道的单元排到同一层，用于查看矩形布局。输出端相位 D 始终在所有层之后。

四通道示例的各层通道对为 `[(0,1),(2,3)] → [(1,2)] → [(0,1),(2,3)] → [(1,2)]`，共 6 个 MZI。

## 数学约定及顺序

沿用唯一器件矩阵：

```text
T(theta,phi) = [ exp(i phi) cos(theta)   −sin(theta) ]
               [ exp(i phi) sin(theta)    cos(theta) ]
```

因此 phi 先作用于第一输入端口。theta 是有效混合角，不能直接解释为物理两臂相位差。数学上使用列向量，右边的矩阵先作用：

`Uhat = D T_k … T_2 T_1`，其中 `D=diag(exp(i output_phases))`。

单元都嵌入 N 通道，其余通道保持不变。输入与输出都是复数光场；功率另算为 abs(E)**2。比较时保留全部复相位，不做全局相位对齐。

## 实现原理

沿反对角线交替从右侧与左侧消去元素，直到剩下对角相位矩阵。右消元用 `U ← U T†`，左消元用 `U ← T U`。对待消元素 a 和参考元素 b，使用 `atan2(abs(a),abs(b))` 求混合角，相位来自 `arg(a)−arg(b)`；左消元对下方目标元素取负。避免计算 a/b，因而可处理零元素。恰为零时选合法的任意相位 0；极小非零值不主动截断。

左侧逆变换需要搬到输出相位之前才能得到统一的传播顺序。对一对通道，有精确恒等式：

`T(theta,phi)† diag(exp(ia),exp(ib))`

`= diag(exp(i(b−phi)),exp(ib)) T(−theta,a−b)`。

依次处理逆序的左消元记录，将新的 MZI 接在右消元记录之后，再保留最终 D。这也是实现中允许负 theta 的原因。

算法结构依据 [Clements 等，Optimal design for universal multiport interferometers，Optica 3 (2016)](https://doi.org/10.1364/OPTICA.3.001460)。本项目代码按已确定的 T 矩阵推导相位搬移关系；不能直接混用其他软件的参数约定。Reck 仅作背景，本项目没有采用其三角网格。

## 验收和范围

`python -m pytest -q` 验证手算串联顺序、所有基向量、功率、完整相位、非法输入、奇偶通道数、全部四通道置换，以及极小元素。随机目标由独立的复高斯 QR 分解产生，测试尺寸包含 1、2、3、4、5、8、16。

`python verify_week02.py` 对 2、4、8 通道分别测试 100 个随机矩阵及 4 个特殊矩阵，保存重建、光场、功率与酉性误差。重建误差定义为 `||Uhat−U||_F/||U||_F`，光场误差为 `||y−U@x||₂/||U@x||₂`。零光场在单元测试中单独按绝对误差验证，不做除零计算。

这是理想无损器件的传输矩阵仿真。当前输入 U 必须是酉矩阵，尚不实现任意 W 的乘法；下一阶段通过 `W=UΣV†` 的两套网格和幅度缩放扩展。器件损耗、热漂移、电压到相位的标定，以及电磁场空间分布也尚未建模。
