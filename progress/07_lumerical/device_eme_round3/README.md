# EME 最小入口：第三轮修正结果

日期：2026-11-12  
状态：**得到候选 2×2 非零 S；仍未完成器件验收，`valid_s_parameters=false`、`complete=false`。**

## 修正内容

第三轮补丁修正了 `run_device_eme_minimal.py` 的端口构建：

- `addeme()` 默认已经创建两个端口；
- 旧脚本又调用两次 `addemeport()`，实际形成四个端口，解释了 4×4；
- 现在直接配置默认的 `EME::Ports::port_1` 和 `port_2`，不再额外添加；
- 显式设置 1550 nm、`x min=-5 µm`、5 个 cell 模式、四侧 PML；
- Si 波导 x span 改为 14 µm，使 EME 的 [-5,+5] µm 全段位于同一均匀 Si 波导中；
- 非零 S 只记为 `nonzero_s_observed`，不再直接设 `valid_s_parameters=true` 或 `complete=true`。

## 实际运行

命令：

```powershell
& $py lumerical\run_device_eme_minimal.py --run --out progress\07_lumerical\device_eme_round3_run
```

结果：

```text
status = nonzero_s_observed
user_s_matrix_shape = [2, 2]
user_s_matrix_max_abs = 1.0000000000000189
unitarity_error = 3.774758283725532e-14
candidate_lossless = true
nonzero_s_observed = true
valid_s_parameters = false
complete = false
```

原始候选 S：

```text
S11 = 0
S21 = -0.19668883962640776 - 0.9804659608402807i
S12 = -0.19668883962640776 - 0.9804659608402807i
S22 = 0
```

这是均匀直波导的合理候选：`|S21| ≈ 1`，左右对称，没有增益；`S†S` 的误差约 `3.77×10⁻¹⁴`，在数值精度内接近无损。但还没有验证：

- 端口模式身份和 mode ID；
- S 矩阵的完整被动性/幺正性；
- 网格、PML、横向域、模式数收敛；
- 长度/参考面与传播相位；
- 与 FDE 的 neff/相位交叉验证。

所以它只是**流程打通**，不是器件验收通过。旧 4×4 全零结果保留在 `device_eme/`，作为历史失败记录。

## 文件

```text
lumerical/run_device_eme_minimal.py
progress/07_lumerical/device_eme_round3/eme_minimal.json
progress/07_lumerical/device_eme_round3/eme_minimal_layout.lms

progress/07_lumerical/device_eme_round3_run/eme_minimal.json
progress/07_lumerical/device_eme_round3_run/eme_minimal_layout.lms
progress/07_lumerical/device_eme_round3_run/eme_minimal_layout_p0.log
```

## 下一步

1. 导出两个端口的实际模式场、mode ID、复 neff、与 cell 导模的重叠。
2. 对候选 S 计算 `S†S` 与被动性；确认 `|S21|` 在数值误差内为 1。
3. 扫描 cell 模式数、PML 层数、横向域，做真正的 EME 收敛。
4. 通过后再与 FDE 的均匀条波导 neff/相位比较，最后才考虑耦合器。
