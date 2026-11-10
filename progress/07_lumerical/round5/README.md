# 第五轮：参数生效、checkpoint 与匹配 EME

日期：2026-11-16  
状态：**两个代码遗漏已修并实际验证；匹配 EME 的 neff 与同配置 FDE 一致，但第三长度和实际网格导出未完成，整体 `complete=false`。**

## 1. 本轮修复

### EME 导出重复键与 PML 对象

已修两个会破坏证据/边界设置的代码问题：

- cell `mode fields` 导出先构造唯一键字典，再 `update(y,z,index)`；不会再出现 `multiple values for 'y'`。
- 设置 PML 前显式 `app.select("EME")`，回读实际层数；本机 16→16。`--boundary metal` 在该版本不支持，启动求解前直接拒绝。
- cell 网格、cell neff、各端口场/重叠分开记录状态：单端口导出失败不会覆盖已成功的 mesh 状态。

### EME 的 `--core-step-nm` 真正生效

旧代码只把 `core_step_nm` 写进 JSON，没有传给 `build()`，也没有建立 mesh override。现在：

- `main()` → `run_case()` → `build()` 传递 `core_step_nm`
- 外围最大 dy/dz = 50 nm
- 核心 y×z = 1.2×0.8 µm，使用 `addmesh()` 和 `set maximum mesh step`
- 保存 `core_step_nm`、`mesh_matching_valid`
- 尝试导出实际 y/z 网格和 `index_x`

EME 2025 R2 的 `material` 结果对象在当前 `3D: X Prop` 会话中取不到，实际导出记录为：

```text
mesh_export.status = failed
error = Can not find object 'material'
```

后来已改为在 `run()`/`emepropagate()` 后读取 EME 自己的：

```text
EME::Cells::cell_1  -> mode fields (y/z/index), neff
EME::Ports::port_1/2 -> neff, mode profiles, overlap_forward/backward
```

在已求解工程上的 API 验证拿到了实际 `81×81` 坐标和 `index`，说明读取路径已修正。但新的 2.5 nm、10 µm 核心网格验证运行超过 10 分钟未完成，所以旧的匹配 JSON 仍保留 `mesh_export=failed`，**没有新的核心实际网格证据**。参数已传入、建立了 override，不能等同于实际网格已核对。

### FDE 的 `--trial-modes` 真正生效并回读

旧代码只限制首档筛选模式数，没有设置求解器。现在每档在 `findmodes()` 前：

```python
app.setanalysis("number of trial modes", trial_modes)
trial_modes_actual = app.getanalysis("number of trial modes")
```

4 模式实测：

```text
trial_modes_requested = 4
trial_modes_actual = 4
modes_found = 4
```

2.5 nm 的 neff 与旧 20 模式结果一致（2.4457480915）。

### 收敛助手与输入检查

- 拒绝空、NaN/Inf、非正、重复或非严格递减的步长序列。
- `recent_phase_converged()` 不再只信 `phase_converged` 布尔值：检查相位数值、限值、步长链、检查数量与 run 数一致、限值为有限正数。
- 模式重叠/neff/TE fraction 必须有限；单档模式身份为 `null`，不自动当成功。
- FDE 逐档写 `checkpoint.json`，严格 JSON（`allow_nan=False`），再导出大场文件。
- EME 请求用例数与完成数、长度检查数分开记录；不再 `any` 成功即返回 0。

## 2. FDE 低模式数试跑

命令：

```powershell
& $py lumerical\run_device_fde_local.py --steps-nm 2.5 1.25 0.625 `
  --domain-um 3 2 --boundary metal --trial-modes 4 `
  --out progress\07_lumerical\round5\fde_low_modes
```

已完成并落盘：

| 步长 nm | Re(neff) | 与上一档重叠 | trial/actual |
|---:|---:|---:|---:|
| 2.5 | 2.4457480915 | — | 4/4 |
| 1.25 | 2.4457671489 | 0.999962 | 4/4 |

2.5→1.25 的 100 µm 相位差：

```text
0.00772524 rad  < 0.01
```

但只有两档、一个区间，不能形成“最后两个连续细化区间”的判断：

```text
phase_mesh_converged = false
complete = false
```

0.625 nm 在约 4 GB、超过 10 分钟仍未完成，已终止；2.5/1.25 的 checkpoint 保留，0.625 没有 checkpoint。没有用插值、挑选较好区间或放宽阈值。

## 3. 匹配 EME 2.5 nm

命令（5×4 µm、PML16、5 个 cell 模式、2.5 nm 核心网格）：

```powershell
& $py lumerical\run_device_eme_minimal.py --run --length-scan --length-um 10 `
  --domain-um 5 4 --core-step-nm 2.5 --cell-modes 5 `
  --boundary pml --pml-layers 16 `
  --out progress\07_lumerical\round5\eme_matched_2p5
```

已完成的长度：

```text
L = 10.0 µm
L = 10.1 µm
```

由 S21 相位差反推：

```text
neff_from_length = 2.4457483599911676
FDE 5×4/PML16 neff = 2.4457483599911685
100 µm 相位差 = 3.60×10⁻¹³ rad
```

这与同配置 FDE 在数值精度内一致，明显优于旧 EME 的 `2.436366`；说明补上核心网格后，传播相位/FDE 的差异基本消失。

原始候选 S：

```text
max|S| = 0.9999999999999222
max singular value ≈ 0.9999999999999222
strict_raw_passive = true
```

这次严格原始被动性为 true，且没有用 `candidate_lossless` 替代严格判据。

但：

- 第三长度 10.2 µm 超过 10 分钟未返回，已终止；
- 预期 3 个长度、2 个相邻区间，实际只有 2 个长度、1 个区间；
- `mesh_export=failed`，实际网格坐标未核对；
- 端口/cell 模式身份和场尚未导出；
- `flow_complete=false`、`complete=false`。

## 4. 结果表

| 项目 | 结果 | 状态 |
|---|---|---|
| FDE trial-modes 4 实际生效/回读 | 4/4 | 通过 |
| FDE 2.5→1.25 nm 相位 | 0.007725 rad | 单区间通过，不足以判两区间 |
| FDE 0.625 nm | 约 4 GB、>10 min | 未完成 |
| EME core-step 2.5 nm 参数传递 | 已传入并建立 override | 代码路径通过 |
| EME PML 设置/回读 | 请求 16，实际 16 | 对象选择通过 |
| EME 实际网格导出 | 已改为 cell mode fields；旧运行 `material` 失败，新核心运行超时 | API 已修，核心证据未生成 |
| EME 10→10.1 µm neff | 2.4457483599911676 | 与 FDE 同配置一致 |
| EME 10.2 µm | >10 min | 未完成 |
| EME 严格原始被动性 | max|S|=0.9999999999999222 | true |
| 整体 complete | — | false |

## 5. 文件

```text
lumerical/run_device_fde_local.py
lumerical/run_device_eme_minimal.py
lumerical/validation.py
lumerical/summarize_device_validation.py
tests/test_lumerical_mapping.py

progress/07_lumerical/round5/README.md
progress/07_lumerical/round5/summary.json
progress/07_lumerical/round5/manifest.json
progress/07_lumerical/round5/summary_from_manifest.json
progress/07_lumerical/round5/fde_trial4/
progress/07_lumerical/round5/fde_low_modes/
progress/07_lumerical/round5/eme_matched_2p5/
progress/07_lumerical/round5/eme_export_api_test/
progress/07_lumerical/round5/eme_layout_check2/
```

## 6. 结论边界

- 电路级 INTERCONNECT 结果继续有效，不能改称实物芯片验证。
- 均匀直波导的 FDE 与匹配 EME 在 10→10.1 µm 上已经相互一致；这是本轮最重要的正面结果。
- 但第三长度、实际网格证据、端口/cell 模式身份未齐全，不能写“器件验收完成”。
- 0.625 nm 资源失败、10.2 µm 超时都是未完成项，不是通过。
