# 3D EME 直波导尝试（未完成）

日期：2026-11-12  
状态：**未完成，未通过验收；不能作为 S 参数或器件验证结果。**

## 已尝试的模型

```text
求解器：MODE EME，3D: X Prop
材料：Si (Silicon) - Palik / SiO2 (Glass) - Palik
Si 截面：y = 0.5 µm，z = 0.22 µm
传播长度：10 µm
横向域：y = 4 µm，z = 3 µm
cell：1 个均匀 cell，group span = 10 µm
内部模式数：10
PML：16 层
energy conservation：none
端口：左右各一个，y/z span = 2.5/2.5 µm，fundamental mode
```

模型布局文件：

```text
eme_layout_attempt.lms
eme_minimal_layout.lms
```

最小入口：

```text
lumerical/run_device_eme_minimal.py
```

默认只建布局并保存 `eme_minimal_layout.lms`；`--run` 会尝试求解，但本轮没有在这台 8 GB 机器上完成，不能把它当作已验证结果。

## 实际运行结果

1. 第一次 EME `run()` 完成，但 `user s matrix` 是 **4×4 全零**（只有 ~1e-16 的数值噪声），不能作为透射结果。
2. 之后尝试对两个端口执行 `updateportmodes(1)`，命令超过 10 分钟未返回，已终止进程。
3. 另一次把左右端口显式放到 `x=±5 µm`、`use full simulation span=true` 的尝试也超过 10 分钟未返回，已终止。
4. 因此没有获得有效的 3D EME S 参数，也没有把全零 S 矩阵回灌主电路。

## 为什么没有继续

- 端口模式没有成功建立/确认，`user s matrix` 全零说明端口之间的模式重叠或传播路径没有正确计算。
- 需要先检查：端口模式选择、`updateportmodes`、端口方向/参考面、cell 长度、EME 模式数、PML 和能量修正。
- 在 8 GB RAM 的机器上，端口模式更新超过 10 分钟，不能把“进程未崩溃”当作成功。
- 报告保留为未完成，不用零矩阵 S 冒充结果。

## 下一步（接续者）

1. 先在 GUI 或脚本中确认左右端口各自找到 quasi-TE0，并保存 `neff`、模式场、重叠。
2. 只用 1550 nm 单点、1 个均匀 cell、较少模式数（如 5–10）做最小 EME，先得到非零 `S21`。
3. 再增加模式数、横向域、PML、网格和 cell 过渡，按交接说明的收敛规则验证。
4. 通过前不要导入 INTERCONNECT，也不要把 EME 的 `user s matrix` 当最终器件 S 参数。

当前结论：**3D EME 路线已尝试，但没有完成；这是未运行通过的限制，不是已验证的器件结果。**
