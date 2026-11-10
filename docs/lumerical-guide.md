# Lumerical 光路实现与操作指南

本指南对应第七阶段（`progress/07_lumerical/`）。目标是让组员能在本机重新打开工程、重建电路、运行频域/时域实验，并准确理解哪些是实际运行结果、哪些只是生成脚本。

## 1. 安装、版本与许可证

本机实测：

```text
Windows 10 19045
Lumerical 2025 R2, build 4175
D:\Program Files\ANSYS Inc\v252\Lumerical
官方 Python API: ...\api\python\lumapi.py
随装 Python: ...\python-3.13.1\python.exe (3.13.1, NumPy 2.2.2)
```

`lumapi` 不要复制到项目里，直接从安装目录导入：

```python
import sys
sys.path.insert(0, r"D:\Program Files\ANSYS Inc\v252\Lumerical\api\python")
import lumapi
```

许可证是否有效必须通过实际启动判断，不能只看安装目录。本机已验证：

```python
fdtd = lumapi.FDTD(hide=True)            # 成功
interconnect = lumapi.INTERCONNECT(hide=True)  # 成功
mode = lumapi.MODE(hide=True)            # 成功
```

代码入口 `lumerical/interconnect.py` 会从显式 `root`、`LUMERICAL_API`、`LUMERICAL_ROOT` 或本机默认路径依次解析安装目录，并校验 `api/python/lumapi.py` 存在。换机器时不需要改源码：

```powershell
$env:LUMERICAL_ROOT = "C:\Program Files\Lumerical\v252\Lumerical"
# 或只指定 API
$env:LUMERICAL_API = "C:\Program Files\Lumerical\v252\Lumerical\api\python"
```

项目 Python 测试环境与 Lumerical Python 可以分开：项目用 `.venv` 跑 pytest，Lumerical 脚本用随装 Python 跑；两者只通过 JSON/NPZ/CSV 交换数据。

## 2. 文件结构

```text
lumerical/
  interconnect.py       # 元件、MZI、Clements、SVD、ONA 建模与矩阵提取
  time_domain.py        # 时域编码、90° 混频、I/Q 读出
  experiments.py        # 单 MZI / Clements / SVD / 非理想实验
  run_mzi.py
  run_clements4.py
  run_svd4.py
  run_svd4_td.py
  run_nonideal4.py
  run_iris4.py
  run_mnist16.py
  run_device_fde.py
  run_device_fdtd.py
  make_report.py
progress/07_lumerical/  # 原生 .icp/.lms、JSON、日志、图表
```

## 3. 元件映射（2025 R2 实测）

### 3.1 耦合器

使用 `Waveguide Coupler`，四个端口：左侧 1/2，右侧 3/4。以左侧为输入、右侧为输出，实测：

```text
conjugate=0:  B  = [[1,  i], [i,  1]] / sqrt(2)
conjugate=1:  B† = [[1, -i], [-i, 1]] / sqrt(2)
```

`coupling coefficient` 是**交叉耦合功率**，不是直通功率。实测 coefficient=0.49 时输出幅度为 `[sqrt(0.51), i sqrt(0.49)]`。因此项目定义的直通功率 `η` 要写成：

```python
setnamed(coupler, "coupling coefficient 1", 1 - eta)
setnamed(coupler, "coupling coefficient 2", 1 - eta)
```

理想 η=0.5 时 coefficient=0.5；非理想 η=0.49 时 coefficient=0.51。

### 3.2 相移器

`Optical Phase Shift` 的复传输为 `exp(+i phase)`，单位 rad。实测 φ=0.7 时传输为 `0.764842+0.644218i`。

### 3.3 MZI

唯一约定仍是：

```text
T(θ,φ) = [[exp(iφ)cosθ, -sinθ],
          [exp(iφ)sinθ,  cosθ]]
```

实际连接：

```text
输入上通道 → P(φ) → C1(port 1)
输入下通道 →        C1(port 2)
C1(port 3) → P(2θ) → C2(port 3)
C1(port 4) →         C2(port 4)
C2(port 1) → P(−θ) → 输出上通道
C2(port 2) → P(−θ) → 输出下通道
C1: conjugate=0 (B)
C2: conjugate=1 (B†)
```

这正好实现：

```text
Dχ B† diag(exp(iδ),1) B diag(exp(iφ),1)
δ = 2θ, χ_upper = χ_lower = −θ
```

不能用同一个 `B` 做两个耦合器；否则会多出固定相位，不能靠逐样本相位对齐掩盖。

### 3.4 网格

每个 MZI 还保留两个输出参考相位。`N` 通道 Clements 网格共有 `N(N−1)/2` 个 MZI，最后再有 `N` 个网格输出相位。4×4 SVD 两套网格共 12 个 MZI、8 个输出相位、4 个中间幅度衰减器。16 通道两套网格共 240 个 MZI、32 个输出相位、16 个衰减器。

通道编号从 0 开始，输入是列向量；ONA 的 `input 1..N` 对应输出通道 0..N−1，激励源接到输入通道 0..N−1。

### 3.5 衰减器

`Optical Attenuator` 的 `attenuation` 是功率 dB。项目 `t=σ/s` 是幅度透过率，设置：

```python
attenuation_db = -20 * log10(t)
```

t=0 时用 300 dB 表示关断；不要把 t 直接填成 dB。

## 4. 频域 SPS/ONA 测试台

### 4.1 官方要求

SPS（S-Parameter Simulator）频域测试只能使用 ONA，`analysis type=scattering data`，不要同时放 CW Laser。ONA 的 `output` 是激励源，`input 1..N` 是接收端口。

一个 N 端口前向网络：ONA 设 `number of input ports=N`，`output` 接到待测网络第 j 个输入，`input 1..N` 接到 N 个输出。对 j=0..N−1 各跑一次，得到 H 的第 j 列。

### 4.2 本项目的设置

```text
start frequency = 193.09 THz
stop frequency  = 193.11 THz
number of points = 3
analysis type = scattering data
minimum amplitude = 1e-30
minimum loss = 300 dB
sensitivity = -300 dBm
```

只读取 `input i/mode 1/transmission` 的中间点，不遍历所有结果。

**频率陷阱：** ONA 的离散频率是 `f_n = start + n*(stop-start)/N`，最后一个点不等于 `stop`。3 点时中间点是 `start + (stop-start)/3`，不是 `(start+stop)/2`。因此从 ONA 结果取复 S21 做相位比较时，必须读取结果里的 `frequency` 数组，用实际频率计算，而不是假定中心频率。

**零输出陷阱：** ONA 对精确零透射不返回 `transmission` 结果，即使把 `minimum amplitude` 设为 0。代码现在把 API 错误、空数组、缺字段和非有限值都视为无效数据并中止；只有明确“结果缺失”且独立功率低于阈值时，才允许记为检测限以下的补零项，并记录功率、阈值、幅度上界和通道索引。

对小规模电路，先用独立时域 CW + 功率监视器测一张功率矩阵（3–5 ns 窗口，取 3 ns 后稳态），功率计灵敏度显式设为 `1e-13 W`（约 `-100 dBm`）。在 1 W 输入下，`1e-12 W` 阈值对应幅度上界 `1e-6`；当前实测 `1e-13 W` 对应约 `3.16×10⁻⁷`。这只是功率上界，不是 `1e-10` 复振幅精度，也不是精确的复数零。16 通道 W1 是稠密矩阵，不构造这个额外证明，若 ONA 缺失则直接失败。

### 4.3 运行

```powershell
$py = "D:\Program Files\ANSYS Inc\v252\Lumerical\python-3.13.1\python.exe"
& $py lumerical\run_mzi.py
& $py lumerical\run_clements4.py
& $py lumerical\run_svd4.py
& $py lumerical\run_nonideal4.py
& $py lumerical\run_iris4.py
& $py lumerical\run_mnist16.py          # 首次无 raw 数据时加 --download
& $py lumerical\run_device_fdtd_slab.py
& $py lumerical\run_device_eme_minimal.py   # 只建布局；--run 尚未在本机完成
```

`run_mnist16.py` 会在打开 Lumerical 之前检查冻结模型、PCA、参考预测和 10,000 张官方测试数据；缺数据会立即报错，不会先花时间建 240 个 MZI。首次运行可用：

```powershell
& $py lumerical\run_mnist16.py --download
```

每个脚本在 `progress/07_lumerical/...` 下保存 `.icp` 原生工程和 JSON。`.icp` 可直接用 INTERCONNECT 打开。

## 5. 时域 I/Q 测试台

### 5.1 为什么不能直接用功率

普通光电探测器输出 `|E|²`，丢掉符号和相位。要得到带符号/复数的 `Wx`，必须有共同相位参考的相干接收。本项目用：

```text
一个 CW Laser
  → 理想 Fork 扇出
  → 4 路幅度衰减 + 相移（编码 x/g）
  → SVD 光路
  → 每个输出分到 0 与 π/2 两个 2×2 混频器
  → 每个混频器 2 个 PIN PD
  → 平衡相减得到 I/Q
```

`Fork 1xN` 在 INTERCONNECT 中是**理想场复制**，每个输出复制同一复场，不按 1/√N 分功率。它只是测试台的相干扇出，不是被验证的 DUT；文档和 JSON 都明确记录这一点。

### 5.2 编码与恢复

```text
g = max(1, max|x|)
每个输入幅度 = |x_j|/g
每个输入相位 = arg(x_j/g)
零幅度用 300 dB 关断
```

固定记录：CW 功率 1 W、频率 193.1 THz、线宽 0；采样率 200 GSa/s；时间窗 5 ns；稳定后取最后 200 点（1 ns）；PD responsivity=1 A/W；LO 相位 0/π2。

平衡公式（单位 LO 幅度）：

```text
I = -(P_Q+ - P_Q-)/2
Q =  (P_I+ - P_I-)/2
E_meas = I + iQ
```

恢复 `Wx`：

```text
y_optical = H (x/g)
Wx ≈ s * g * E_meas
```

`s` 是 SVD 的数字恢复比例，不能描述为光学放大。没有逐样本相位拟合，也没有把幅度/强度当复场。

### 5.3 运行与结果

```powershell
& $py lumerical\run_svd4_td.py
```

4×4 SVD 的 3 组输入（含正、负、复数、零，第三组 g=2）直接时域复场误差约 `4–6×10⁻¹⁶`，恢复 `Wx` 约 `1.2–1.6×10⁻¹⁵`。

总体验收现在同时要求：频域 `H` 通过、每条记录非空且有限、时域对 `Hx` 通过、恢复尺度后的 `Wx` 通过。只有“频域和时域同时错成同一个半值”会失败；不会因为两者相互一致就给 PASS。

## 6. 静态非理想映射

`src.nonideal.realize_nonideal` 先量化物理控制相位，再加入静态高斯偏差，并对 `(φ,δ,χ_upper,χ_lower)` 与网格输出相位都生效。Lumerical 端 `add_mesh_nonideal` 只写入这些**已固定的 actual phases**，不重新采样。

- 均匀 MZI 损耗：Python 对完整 2×2 乘一个幅度因子；Lumerical 在两个输出各放一个相同衰减器。
- 上臂损耗：放在两个耦合器之间。
- 耦合比：Python `coupler_through=η`；Lumerical `coefficient=1−η`。
- 量化：Lumerical 的相移器本身连续，量化已在 Python 快照中完成；不能把“设置连续相位”冒充量化。

5 个条件（相位 0.01 rad、0.1 dB/MZI、6-bit、耦合比 0.49、组合）与 Python 快照的 H 相对误差均约 `5–7×10⁻¹⁶`。

## 7. MODE FDE 器件级检查

```powershell
& $py lumerical\run_device_fde.py
& $py lumerical\run_device_fde_local.py
```

- 几何：Si 0.5 µm × 0.22 µm，SiO₂ 背景，2D X-normal FDE，fundamental TE；1530–1570 nm。
- 全局网格 60/120/480；120→480 最大 `Δneff=2.259×10⁻³`，只通过 neff 的初查判据 `3×10⁻³`。
- **相位没有收敛**：1550 nm、100 µm 波导两档网格的 `Δφ=0.913 rad`，远大于 `0.01 rad`；复 S21 差约 `0.882`。JSON 中 `phase_converged=false`、`complete=false`，状态是 `preliminary_phase_not_converged`。
- 新增局部核心网格（20/10/7.5/5 nm）：`Δφ` 分别为 `1.92134`、`0.01279`、`0.14992 rad`。10→7.5 nm 接近但未过，7.5→5 nm 的差值变大；没有独立真值，不能据此断言 5 nm 一定更不准确。`mode_identity_valid=null`（只检查了 TE 占比，未做模场重叠）。
- 终止判据已改为最近连续两档，避免最早的 20→10 nm 粗差把后续结果永久判 false；本次最近两档仍未通过，所以 `phase_mesh_converged=false`。
- `neff(1550)=2.445848`；TE 占比约 0.984。
- 100 µm 波导复 S21 用 `exp(+i2πneffL/λ)`，与配置了同一 neff 的 INTERCONNECT `Straight Waveguide` 的 ONA 复 S21 一致。这是参数映射自洽检查，不是独立证明器件相位准确。

2D X-normal FDE 在 y-z 截面求全矢量模式，**可以表示沿 x 均匀的 500×220 nm 条波导**；它与 z 方向不变的 2D FDTD slab 不是一回事。FDE 不处理沿 x 变化的耦合器/过渡，但均匀直波导的 neff 和传播相位不需要为了“维度”强行改跑 3D EME。EME/FDTD 用于耦合器、弯曲、端面等非均匀结构。

第四轮进一步做了显式 Metal 边界、实际网格/材料导出和 `bestoverlap`/`overlap` 模式跟踪，见 `progress/07_lumerical/round4/README.md`。10/5/2.5/1.25 nm 的 Re(neff) 为 `2.44525209/2.44565704/2.44574809/2.44576715`，相邻模式重叠均 ≥0.9994；但最后两个连续区间的 `5→2.5 nm` 仍为 `0.03691 rad`，所以 `phase_mesh_converged=false`。0.625 nm 在约 5.16 GB、超过 10 分钟未完成，记录为资源限制。

域 3×2/4×3/5×4（2.5 nm）的 neff 为 `2.44574809/2.44574836/2.44574836`，相位差约 `1.09×10⁻⁴` 和 `3.17×10⁻⁸ rad`；Metal/PML16/PML24 差约 `10⁻¹² rad`。这些是单 2.5 nm 对照，不替代网格收敛；PML 的微小负 Im(neff) 被保留报告。

这份结果只用于“单个代表性波导”的初步核查，不能称为已收敛的相位模型；也没有把传播相位/损耗重复加到 240-MZI 理想电路。

## 8. 2D FDTD 诊断与“卡住”原因

2D FDTD 求解本身约 4 秒/次。之前看起来 10 分钟不返回，实际是运行结束后遍历所有结果时读取了 `farfield` 巨型数据集。排查步骤：

1. 最小 2D 均匀介质 + 偶极源 + 功率监视器：2.4 秒完成。
2. 加波导 + 模式源 + 功率监视器：4.2 秒完成。
3. 运行后只读 `T` 和局部 `E`：正常退出。
4. 遍历 `getresult` 的所有条目并读 `farfield`：进程长时间不返回。

因此代码里明确注释：**不要枚举全部 FDTD 结果，只读所需字段**。

短波导端面在计算域内，T 有纹波；mesh 2/3 最大差 0.278，`max T=1.074`。所以 `run_device_fdtd.py` 只生成诊断 JSON，不作为定量 S 参数。要得到定量 FDTD 结果，需要把波导延伸进 PML、加长稳定时间并做更严格的网格/反射收敛；本轮没有把未收敛结果用于主电路。

补充的 `run_device_fdtd_slab.py` 只做 2D slab 修补：Si 延伸进两侧 PML，源在 -3.5 µm、参考面在 ±3 µm，T 用 P_out/P_in。mesh 2/3/4 的相邻最大差为 `8.19×10⁻⁶`、`5.82×10⁻⁶`，但 `max T=1.00000497`，严格被动性仍未通过；`geometry_valid_for_strip=false`，不是目标 0.5×0.22 µm 条波导。

3D EME 早期尝试的 `user s matrix` 是 4×4 全零，原因是 `addeme()` 默认已有两个端口，旧脚本又添加两个；端口模式更新也超过 10 分钟。第三轮改为直接配置默认 `EME::Ports::port_1/port_2`、5 个 cell 模式、四侧 PML，实际得到候选 2×2：

```text
S21 = -0.19668884 - 0.98046596i
max|S| = 1.0000000000000189
status = nonzero_s_observed
valid_s_parameters = false
complete = false
```

这只是 EME 流程打通，不是器件验收；模式身份、被动性/幺正性、网格/域/PML/模式数收敛和相位都还没有验证。详见 `progress/07_lumerical/device_eme_round3/README.md`。

第四轮又跑了 10/10.1/10.2 µm：由 S21 相位差反推 `neff=2.436366`，而 FDE 为 `2.445748`；100 µm 相位差约 `3.803 rad`，不一致。原始 `max|S|` 和最大奇异值约 `1.0000000000000189 > 1`，`strict_raw_passive=false`。因此 EME 长度相位检查没有通过。

## 9. 结果如何解读

- “脚本已生成”：代码存在，但还没运行。
- “工程已运行”：`.icp/.lms` 已由 `lumapi` 实际保存并完成 `run()`。
- “验证通过”：完整复数矩阵/复场与独立 NumPy 参考比较，满足预先设定的误差判据。

本阶段三种状态分别记录在 `progress/07_lumerical/README.md`。不要把 Python 参考、频域 H 回灌和时域直接求解混为一谈：

| 验证 | 做了什么 | 不能声称什么 |
|---|---|---|
| 频域 H | ONA 逐输入提取完整复矩阵 | 不能声称每张图片都做了瞬态仿真 |
| 时域 I/Q | 直接编码 3 组 4 通道输入，PD 平衡读出 | 不能声称 10,000 张 MNIST 都逐张时域仿真 |
| 混合推理 | Lumerical H 导回 NumPy，跑完整测试集 | 必须标注“传输矩阵提取后的混合推理” |

## 10. 已知限制

- 16 通道网络只做频域 H 和混合推理；没有 16 路时域 I/Q。
- 非理想静态误差只在 4×4 上验证。
- 2D FDTD 未得到收敛 S 参数；没有 3D FDTD 或全 240-MZI 的 FDTD。
- 没有校准、噪声训练、动态噪声、真实接收机标定、速度/能耗结论。
- 所有误差是模型/数值结果，不是芯片实测。
