# 第七阶段：Lumerical 光路实现

状态：**理想电路、时域 I/Q、静态误差和 AI 混合推理已实际运行；算术验收通过，但含结构性零的矩阵有功率检测限补零项，`complete=false`。MODE FDE 的 neff 初查通过，相位精度未收敛；2D FDTD 仅为诊断。** 本阶段保留前六阶段全部 Python 代码、模型和历史结果；只把数学光路落地为 Lumerical 2025 R2 的 INTERCONNECT/MODE 工程，并加入与仓库定义一致的静态器件误差。

## 1. 环境与许可证

| 项目 | 实测值 |
|---|---|
| 操作系统 | Windows 10 19045，64 位 |
| Lumerical | 2025 R2，build 4175，`D:\Program Files\ANSYS Inc\v252\Lumerical` |
| 官方 Python API | `...\api\python\lumapi.py`，随装 Python 3.13.1，NumPy 2.2.2 |
| 项目测试环境 | 独立 `.venv`，Python 3.13.0、NumPy 2.4.1、pytest 8.3.5 |
| Git | VS 2022 自带 Git 2.46.2 |
| 许可证 | FDTD、INTERCONNECT、MODE 均实际启动成功；不是只看安装文件 |

FDTD、INTERCONNECT、MODE 都通过 `lumapi` 实例化并关闭；INTERCONNECT 还完成了全部电路求解。

## 2. 实际完成的光路

| 光路 | 规模 | 状态 |
|---|---|---|
| 单 MZI | 2 耦合器 + 4 相移器 | 已运行，4 组相位验证 |
| 4 通道 Clements | 6 MZI | 已运行，4 类酉矩阵验证 |
| 4×4 SVD | 12 MZI + 4 衰减器 + 8 网格输出相位 | 已运行，4 类矩阵验证 |
| 4×4 SVD 时域 I/Q | 4 输入编码 + 8 混频器 + 16 PD | 已运行，3 组 4 通道输入 |
| 4×4 静态非理想 | 相位/损耗/量化/耦合比 5 条件 | 已运行，与 Python 快照对照 |
| Iris 4×4 | 第一层 4×4 权重 | 已运行，30 个测试样本 |
| MNIST 16×16 | 2×120=240 MZI + 16 衰减器 + 32 网格输出相位 | 已运行，16×16 复数 H 提取 |
| MODE FDE 波导 | Si 0.5×0.22 µm / SiO₂，1530–1570 nm | 已运行；neff 初查通过，相位未收敛 |
| 2D FDTD 波导 | 诊断性短波导 | 求解已运行，结果仅作诊断，未用于定量 S 参数 |

所有 MZI 都由两个 `Waveguide Coupler` 和四个 `Optical Phase Shift` 组成；没有用稠密 W 黑盒，也没有用 Python `W@x` 代替 Lumerical 求解。

## 3. 关键端口与相位约定（实测，不是假定）

- `Waveguide Coupler` 端口 1/2 在左、3/4 在右；`conjugate=0` 时前向矩阵为
  `B = [[1, i], [i, 1]]/√2`。
- `conjugate=1` 时同一耦合器给出 `B† = [[1, -i], [-i, 1]]/√2`。这是固定的端口参考面选择。
- Lumerical 的 `coupling coefficient` 是**交叉耦合功率**。实测 coefficient=0.49 时输出为 `[√0.51, i√0.49]`，所以项目直通功率 `η` 对应 `coefficient=1−η`。
- `Optical Phase Shift` 的传输为 `exp(+i phase)`。
- 单 MZI 的连接与设置为：
  `P(φ) → B → 上臂 P(2θ) → B† → 上/下输出 P(−θ)`。
  该映射对 θ=0、π/4、π/2 和多组 φ 均与 `T(θ,φ)` 逐元素一致。

## 4. 实际结果

### 4.1 单 MZI

4 组相位设置，完整 2×2 复数矩阵最大相对误差 **2.22×10⁻¹⁶**。零混合角与 π/2 也包含在内。θ=0 和 θ=π/2 两组各有 2 个结构性零项，使用功率检测限补零；算术误差通过，但 `complete=false`，其他两组 `complete=true`。

### 4.2 4 通道 Clements

| 目标 | 相对误差 |
|---|---:|
| 随机复酉矩阵 | 8.42×10⁻¹⁶ |
| 单位矩阵 | 3.85×10⁻¹⁶ |
| 反序置换 | 5.53×10⁻¹⁶ |
| 对角相位 | 4.10×10⁻¹⁶ |

随机复酉矩阵无回退项，`complete=true`；identity、反序置换、对角相位各有 12 个结构性零项，`complete=false`，零项只作为功率上界。

### 4.3 4×4 SVD

| 目标 W | H 对 W/s 相对误差 | sH 对 W 相对误差 |
|---|---:|---:|
| 随机复矩阵 | 1.60×10⁻¹⁵ | 1.58×10⁻¹⁵ |
| 单位矩阵 | 9.87×10⁻¹⁶ | 9.87×10⁻¹⁶ |
| 秩一矩阵 | 1.52×10⁻¹⁵ | 1.48×10⁻¹⁵ |
| 零矩阵 | 不适用（复 H 未测） | 功率泄漏上界，见下 |

`s=max(1,σmax)`；衰减器按幅度 `t=σ/s` 设置，Lumerical 中用功率 dB `−20log10(t)` 实现。

零矩阵不能靠“ONA 缺结果”判定：ONA 对精确零会省略 transmission。零矩阵改用独立的时域 CW + 四个功率监视器，取 3–5 ns 稳态窗口；在 1 W 输入、功率计灵敏度 `1e-13 W`（约 `-100 dBm`）下，记录输出功率、`1e-12 W` 验收阈值和幅度上界。当前最大输出功率 `1e-13 W`，对应幅度上界约 `3.16×10⁻⁷`；`complex_H_measured=false`，误差字段为 null，不能写成“精确零”或 `1e-10` 复振幅精度。

单位矩阵、反序置换和对角相位矩阵中的结构性零输出也使用同一套独立功率图；JSON 里保存 `fallback_count`、`fallback_entries`、`zero_amplitude_upper_bound`、`power_meter_sensitivity_w` 和 `complex_fully_measured`。这些矩阵的算术误差只描述已测/未测混合结果，`complete` 会明确为 false。单 MZI 的 θ=0、π/2 两组也有 2 个检测限零项，`complete=false`；随机复矩阵、秩一矩阵、Iris 和 MNIST 的稠密 H 没有回退项，`complete=true`。单通道/矩阵缺失、API 错误、空数组、缺字段或非有限值都会中止，不再被当作 0。

### 4.4 时域相干 I/Q（4×4 SVD）

- 共同 CW 光源；四个输入幅度/相位编码 `x/g`，`g=max(1,max|x|)`。
- 每个输出用 0 与 π/2 两个 2×2 混频器，16 个 PIN 光电探测器，平衡相减得到 I/Q。
- 5 ns 窗口、200 GSa/s，取最后 200 点（1 ns）平均；PD responsivity=1 A/W。
- 三组输入：`[0.8,−0.5,0.3+0.4i,−0.2i]`、`[i,−1,0.5,0]`、`[2,−1,0.5i,0]`（第三组 g=2）。
- 复场对 `H(x/g)` 最大相对误差 **6.04×10⁻¹⁶**；恢复 `Wx` 最大相对误差 **1.59×10⁻¹⁵**。
- 总体验收要求频域 H 通过、每条记录有限且非空、Hx 通过、恢复后的 Wx 通过；不再只看 Hx。零/近零参考用尺度化绝对容差。

这是**直接时域 Lumerical 求解**，不是把频域 H 乘上 x 冒充时域结果。

### 4.5 静态非理想

同一 W、同一固定器件 seed=20261110，逐项写入 Python `RealizedMesh` 的实际相位；不同输入复用同一器件。

| 条件 | H 对 Python 快照相对误差 |
|---|---:|
| 相位 σ=0.01 rad | 5.50×10⁻¹⁶ |
| 0.1 dB/MZI | 5.29×10⁻¹⁶ |
| 6-bit 相位量化 | 5.50×10⁻¹⁶ |
| 耦合器直通功率 0.49 | 5.48×10⁻¹⁶ |
| 组合误差 | 7.17×10⁻¹⁶ |

非理想误差是人为设定的静态参数，不是硬件测量；这里验证的是 Lumerical 与 Python 定义一致。

### 4.6 AI 复验

只把第一层权重乘法交给 Lumerical 传输矩阵，偏置、ReLU 和分类头仍为数字计算。

| 任务 | 数字准确率 | Lumerical 传输矩阵混合推理 | 预测一致率 |
|---|---:|---:|---:|
| Iris 4→4→3，30 个测试样本 | 100.00% | 100.00% | 100% |
| MNIST PCA16→16→10，10,000 个测试样本 | 91.99% | 91.99% | 100% |

MNIST 这里使用 seed=20261109 的单个冻结模型；前六阶段三个模型的 Python 平均 92.41% 仍保留。**不是所有 10,000 张图片都在 Lumerical 中逐张瞬态仿真**：H 提取一次，随后在 NumPy 中应用 H 做完整测试集混合推理；另有 4×4 时域测试台直接验证了选定输入的 Hx 一致性。

### 4.7 MODE FDE 器件级检查（相位尚未收敛）

- 几何：Si 0.5 µm × 0.22 µm，SiO₂ 背景，2D X-normal FDE，fundamental TE。
- 1530–1570 nm，网格 60/120/480；120→480 的最大 `Δneff=2.259×10⁻³`，仅通过 neff 的初查判据 `3×10⁻³`。
- **相位判据没有通过**：在 1550 nm、100 µm 波导上，两档网格的 `Δφ=0.913 rad`，远大于预设 `0.01 rad`；两档网格的复 S21 差约 `0.882`。因此 `phase_converged=false`、`complete=false`，结果只标为 `preliminary_phase_not_converged`。
- `neff(1550 nm)=2.445848`；TE 占比约 0.984。
- 由同一 `neff` 配置 INTERCONNECT `Straight Waveguide` 得到的 S21 与解析式一致（差约 `1.1×10⁻¹³`），相位约定为 `exp(+i2πneffL/λ)`。这是**参数映射自洽检查**，不是独立证明 FDE 相位准确。

也就是说，这套 FDE 结果不能作为“已收敛的相位模型”；要证明相位收敛，`Δneff` 需要降到约 `2.47×10⁻⁵` 量级，还需要更细网格、区域/模式一致性检查，或改用 3D EME。

新增 `device_fde_local` 局部网格入口，只做 1550 nm，核心步长 20/10/7.5/5 nm：

```text
neff = [2.45000335, 2.44526360, 2.44529515, 2.44566499]
20→10 nm: Δφ = 1.92134 rad
10→7.5 nm: Δφ = 0.01279 rad
7.5→5 nm: Δφ = 0.14992 rad
phase_converged = false
mode_identity_valid = null（只检查了 TE 占比，未做模场重叠）
complete = false
```

相位仍没有稳定到 `0.01 rad`；`10→7.5 nm` 接近但仍超，`7.5→5 nm` 的差值变大。没有独立真值，不能据此断言 5 nm 一定更不准确，也不能挑选 10→7.5 nm 这一对当作成功依据；需要先检查模式身份/界面/边界。终止判据已改为最近连续两档，不再被最早的 20→10 nm 永久判 false。

2D X-normal FDE 在 y-z 截面求全矢量模式，可以表示沿 x 均匀的 500×220 nm 条波导；不能因为名字里有“2D”就把它等同于 z 方向不变的 2D FDTD slab，也不能说均匀条波导必须再跑 3D EME 才算有截面。

### 4.8 2D FDTD 诊断（旧模型与 slab 修补）

**旧 `device_fdtd` 结果保留，不覆盖。** 它是 2D z-invariant Si slab（不是 500×220 nm 条波导），端面在计算域内，`T>1`，并且只比较 mesh 2/3。`diagnostic_status.json` 明确标记：

```text
diagnostic_only=true
geometry_valid_for_strip=false
max_T_all=1.07541
max_abs_T_change_mesh2_to_mesh3=0.27789
complete=false
```

新增的 `device_fdtd_slab` 只修一项：Si 的 x span 从 10 µm 改为 16 µm，穿过两侧 PML；源在 x=-3.5 µm，功率参考面在 ±3 µm。监视器保存的是**源归一化、无量纲净通量**，`T_net_ratio = output_flux_normalized / input_flux_normalized`，不是瓦特数；跑 mesh 2/3/4。

实测：

```text
mesh 2→3 max |ΔT| = 8.19×10⁻⁶
mesh 3→4 max |ΔT| = 5.82×10⁻⁶
max T_all = 1.00000497
power_ratio_mesh_converged = true
raw_passive = false（严格逐点 T<=1）
power_mesh_checks_passed = false
geometry_valid_for_strip = false
complete = false
```

所以 slab 的网格已经稳定，但严格 `T≤1` 仍差约 `5×10⁻⁶`，没有通过被动性门槛；而且 slab 不是目标条波导，不能据此声称条波导验证完成。

### 4.9 3D EME 尝试（未完成）

早期交互式 EME 尝试（`device_eme/`）是 4×4 全零：`addeme()` 默认已有两个端口，旧脚本又添加两个，形成四个端口；`updateportmodes(1)` 和显式 `x=±5 µm` 都超过 10 分钟。旧记录保留，作为失败历史。

第三轮修正了端口构建（`run_device_eme_minimal.py`）：

```text
直接配置默认 EME::Ports::port_1 / port_2，不再 addemeport()
显式 1550 nm、x min=-5 µm、5 个 cell 模式、四侧 PML
Si x span=14 µm，覆盖整个 EME 段
```

实际 `--run` 结果：

```text
status = nonzero_s_observed
user_s_matrix_shape = [2, 2]
S21 = -0.19668884 - 0.98046596i
max|S| = 1.0000000000000189
unitarity_error = 3.774758283725532e-14
candidate_lossless = true
nonzero_s_observed = true
valid_s_parameters = false
complete = false
```

这是均匀直波导的合理候选，但只表示 EME 流程打通；模式身份、S 矩阵被动性/幺正性、网格/域/PML/模式数收敛和传播相位都还没有验证。详见 `device_eme_round3/README.md`。不能因为 S 非零就写“器件验证完成”。

### 4.10 第四轮：FDE 模式跟踪、域/边界与 EME 长度检查

本轮按 `D:\Lumerical_第四轮_给Windows电脑的具体修改任务.md` 执行，完整说明在 `round4/README.md`。

FDE 10/5/2.5/1.25 nm（3×2 µm，Metal）的结果：

```text
Re(neff) = [2.4452520854, 2.4456570442, 2.4457480915, 2.4457671489]
overlap  = [—, 0.999408, 0.999851, 0.999962]
10→5 nm:     0.16416 rad  fail
5→2.5 nm:    0.03691 rad  fail
2.5→1.25 nm: 0.00773 rad  pass
```

最近两个连续细化区间是 5→2.5 和 2.5→1.25，前者超 0.01，所以 `phase_mesh_converged=false`；0.625 nm 补充运行达到约 5.16 GB、超过 10 分钟，已终止并记录为资源限制，未进入收敛结论。

域大小（2.5 nm，Metal）：

```text
3×2: 2.4457480915
4×3: 2.4457483599
5×4: 2.4457483600
3×2→4×3: 1.09×10⁻⁴ rad
4×3→5×4: 3.17×10⁻⁸ rad
```

仅该 2.5 nm 单步对照的域变化低于 0.01 rad；边界 Metal/PML16/PML24 的相位差约 `10⁻¹² rad`，PML 微小负 Im(neff) 被保留报告。两者都不替代网格收敛。

EME 长度 10/10.1/10.2 µm 的 S21 相位反推：

```text
neff_from_length = 2.436366
FDE neff(2.5 nm) = 2.445748
100 µm 相位差 = 3.803 rad
```

长度相位与 FDE 不一致；`max|S|` 和最大奇异值约 `1.0000000000000189 > 1`，`strict_raw_passive=false`。所以整体仍是：

```text
complete = false
```

不是“第四轮已经收敛”，而是“模式身份和域/边界已有真实数据，网格相位/EME 长度一致性仍未通过”。

当前也没有把 4.7 的初步 FDE 相位当作已收敛的定量器件 S 参数。

## 5. 文件

| 文件 | 内容 |
|---|---|
| `single_mzi/single_mzi.icp`、`.json` | 单 MZI 原生工程与 4 组矩阵 |
| `clements4/*.icp`、`clements4.json` | 4 通道 Clements 工程与结果 |
| `svd4/*.icp`、`svd4.json` | 4×4 SVD 原生工程与结果 |
| `svd4_td/*.icp`、`time_domain.json` | 时域 I/Q 工程与原始复场 |
| `nonideal4/*.icp`、`nonideal4.json` | 5 个静态非理想条件 |
| `iris4/*.icp`、`hybrid_inference.json` | Iris 混合推理 |
| `mnist16/seed_20261109.icp`（约 30.7 MB）、`*_matrix.json`、`hybrid_*` | 16 通道 240-MZI 工程与全测试集混合推理 |
| `device_fde/soi_waveguide_fde.lms`、`device_fde.json` | MODE FDE 几何工程与初步 neff/映射检查；相位未收敛 |
| `device_fde_local/soi_waveguide_fde_local.lms`、`device_fde_local.json` | 局部核心网格 20/10/7.5/5 nm；相位仍未收敛 |
| `device_fdtd/device_fdtd.json`、`device_fdtd/diagnostic_status.json`、`device_fdtd/fdtd_diag_mesh*.fsp`、`*.log` | 旧 2D slab 诊断，保留历史；明确不是条波导 |
| `device_fdtd_slab/device_fdtd_slab.json`、`fdtd_slab_mesh2/3/4.fsp`、`*.log`、`slab_T.png` | 新 2D slab 修补：3 档网格、严格逐点 T<=1 未通过 |
| `device_eme/README.md`、`attempt.json`、`eme_layout_attempt.lms` | 早期 3D EME 尝试；4×4 全零/端口模式更新超时，保留失败记录 |
| `device_eme_round3/`、`device_eme_round3_run/` | 修正默认端口后的 2×2 候选 S；`valid_s_parameters=false`、`complete=false` |
| `round4/` | 第四轮 FDE 10/5/2.5/1.25 nm、域/边界、EME 10/10.1/10.2 µm；完整说明与图 |
| `summary_errors.png`、`time_domain_iq.png`、`ai_accuracy.png`、`fde_neff.png`、`fdtd_diagnostic.png` | 同批 JSON 生成图表 |

## 6. 复现

在仓库根目录，使用 Lumerical 随装 Python：

```powershell
$py = "D:\Program Files\ANSYS Inc\v252\Lumerical\python-3.13.1\python.exe"
& $py lumerical\run_mzi.py
& $py lumerical\run_clements4.py
& $py lumerical\run_svd4.py
& $py lumerical\run_svd4_td.py
& $py lumerical\run_nonideal4.py
& $py lumerical\run_iris4.py
& $py lumerical\run_mnist16.py
& $py lumerical\run_device_fde.py
& $py lumerical\run_device_fde_local.py
& $py lumerical\run_device_eme_minimal.py --run --length-scan --length-um 10 --out progress\07_lumerical\round4\eme_length_scan
& $py lumerical\run_device_fdtd.py
& $py lumerical\run_device_fdtd_slab.py
& $py lumerical\make_report.py
```

Python 回归：`.\.venv\Scripts\python.exe -m pytest -q`，当前 **201 passed, 4 skipped**。

## 7. 明确未完成/边界

- 16 通道网络只做了频域复数 H 提取和 H 回灌混合推理；没有为 16 通道搭建 16 路 I/Q 时域接收器。
- 时域 I/Q 只对 4×4 SVD 的 3 组输入做了直接验证，没有逐张 MNIST 图片做瞬态仿真。
- 非理想静态误差只在 4×4 上验证；16 通道没有重复全部误差扫描。
- 没有加入校准优化、噪声训练或动态噪声；没有声称硬件测量、速度或能耗优势。
- 旧 2D FDTD 是 z-invariant slab，不是 0.5×0.22 µm 条波导，`T>1`，只作历史诊断。
- 新 slab 诊断的 mesh 2/3/4 已稳定，但 `max T=1.00000497`，严格被动性仍未通过；参考面位置、源距离、PML 层数和时间窗的逐项对照还没有做，不能把 4.97 ppm 残差唯一归因。
- 3D EME 已得到 2×2 非零候选 S，但 `max|S|`/最大奇异值约 `1.0000000000000189 > 1`，`strict_raw_passive=false`；长度 10/10.1/10.2 µm 反推 `neff=2.436366`，与 FDE `2.445748` 的 100 µm 相位差 `3.803 rad`，不一致。没有有效器件验收。
- FDE 第四轮 10/5/2.5/1.25 nm 的模式重叠 ≥0.9994，但 5→2.5 nm 仍为 `0.03691 rad`；0.625 nm 因约 5.16 GB、超过 10 分钟未完成。`phase_mesh_converged=false`，没有把传播相位回灌 240-MZI 主电路。
- 没有做 3D FDTD 或整个 240-MZI 网络的 FDTD。
