# 8 GB 补验：代码修正与实际状态

日期：2026-11-18  
基线提交：`0802c1dab8082d988628e8aeb41fb6c91f4fb649`

## 1. 本轮修改

### EME 重复 cell 导出键

`run_device_eme_minimal.py` 现在先构造唯一键字典，再更新 `y/z/index`：

```python
payload = {
    k: np.asarray(v)
    for k, v in cell.items()
    if k != "Lumerical_dataset"
}
payload.update(y=y, z=z, index=index)
np.savez_compressed(..., **payload)
```

不会再出现 `got multiple values for keyword argument 'y'`。单测 `test_eme_cell_export_does_not_duplicate_y_z_index` 覆盖。

### PML 对象与回读

PML 设置在 `app.select("EME")` 后执行，并回读：

```text
pml_layers_requested = 16
pml_layers_actual = 16
boundary_applied = PML
```

`--boundary metal` 在当前 EME 3D:X 版本不支持，启动求解前直接拒绝（退出码 2），不再静默当作 PML。

### EME propagation sweep

新增 `--propagation-sweep`：

```python
app.setemeanalysis("propagation sweep", 1)
app.setemeanalysis("parameter", "group span 1")
app.setemeanalysis("start", 10e-6)
app.setemeanalysis("stop", 10.2e-6)
app.setemeanalysis("number of points", 3)
app.emesweep()
scan = app.getemesweep("S")
```

已在已求解 EME 工程上验证 `getemesweep("S")` 返回：

```text
group_span_1 = [10.0, 10.1, 10.2] µm
s11/s12/s21/s22 各 3 点
```

但目标 2.5 nm 核心网格的 `run()` 在本机超过 10 分钟仍未完成，已终止；没有生成新的 10.2 µm 原始 S。这是求解资源/时间未完成，不是 sweep 解析失败。

### 汇总

`summarize_device_validation.py` 现在：

- 计算全部相邻长度区间，不只第一个；
- 检查 manifest 期望长度、顺序、S 形状、S21 相位可判定性；
- `_fde_neff()` 按 `fde_reference_step_nm` 取，不再默认 `runs[0]`；
- 域对照要求 3×2、4×3、5×4 三档且相邻相位 ≤0.01；
- 边界对照要求三档且相邻相位 ≤0.01；
- `eme_evidence_complete` 要求实际 mesh flag、cell neff、两个端口 profile 都保存；
- 分开报告 `eme_flow_complete`、`eme_phase_checks_complete`、`eme_phase_mapping_passed`、`eme_device_complete`；
- 总 `complete` 由这些证据计算，不再硬编码 false。

## 2. 当前结果

### FDE

本轮没有重新完成 0.625 nm。已有可核查数据：

```text
2.5 nm:  2.4457480915
1.25 nm: 2.4457671489
2.5→1.25 nm: 0.00772524 rad
```

只有一个区间；0.625 nm 未完成。`phase_mesh_converged=false`。

### EME

`round5` 的 10/10.1 µm 结果仍是：

```text
neff_from_length = 2.4457483599911676
FDE 同配置 neff = 2.4457483599911685
100 µm 相位差 ≈ 3.6×10⁻¹³ rad
```

但 10.2 µm 未完成；旧 JSON 的 `mesh_export` 仍为 failed。新 sweep 代码路径已验证，目标核心运行未完成。

### 汇总状态

```text
fde_mesh_phase_passed = false
fde_domain_checked = true（3×2/4×3/5×4，2.5 nm）
fde_boundary_checked = true（Metal/PML16/PML24）
fde_mode_identity_valid = true（已有重叠 ≥0.9994）
uniform_strip_fde_complete = false
eme_flow_complete = false
eme_phase_checks_complete = false
eme_evidence_complete = false
eme_device_complete = false
complete = false
```

## 3. 未完成项

1. FDE 0.625 nm：约 4 GB、>10 min；仍缺第二个连续区间。
2. EME 2.5 nm core propagation sweep：一次 `run()` 在本机 >10 min；缺 10.2 µm 实际 S。
3. EME 实际核心网格/端口场：新导出路径可用，但目标运行未完成，旧匹配 JSON 仍缺核心证据。

## 4. 结论

代码层的重复导出、PML 选择、Metal 拒绝、全区间/域/边界/端口证据检查已修；实际电磁验收仍因 0.625 nm 和 10.2 µm 未完成而保持 `complete=false`。没有裁剪 T、放宽阈值、挑选区间或用单元测试替代求解。
