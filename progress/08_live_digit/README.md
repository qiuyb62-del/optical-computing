# 单张手写数字：16 通道 INTERCONNECT 现场闭环

日期：2026-11-18  
状态：**最小闭环已完成并实际运行一次；不是任意手写图片精度保证，也不是完整 10,000 张逐张仿真。**

## 1. 做了什么

新增：

```text
lumerical/live_digit.py
lumerical/live_digit_ui.py
tests/test_live_digit.py
```

流程：

```text
画板/上传图片
→ 28×28 归一化、裁剪、居中
→ 冻结 PCA（16 维）
→ 数字基线 W1x + b1 + ReLU + W2 + b2
→ 16 通道时域 INTERCONNECT（实际编码 + 相干 I/Q 读出）
→ s*g*y 恢复复线性输出
→ 同一个数字分类头
```

`live` 模式不是 `H@x`：每次识别都建立 16×16 SVD 电路、设置输入幅度/相位、运行 INTERCONNECT、读取 16 路复输出。

另有两种明确标注的对照：

- `cached`：读取已保存的 16×16 H，`s*H@x`，标记 `cached_H_demo`，不调用求解器；
- `digital`：只算数字基线，不调用求解器。

## 2. 实际运行证据

用一张程序生成的“1”跑 `live`，两次实测分别约 124.4 s 和 97.1 s；最新一次：

```text
status = live_ok
live_interconnect_called = true
elapsed_seconds = 97.10
build_seconds = 46.51
run_seconds = 45.39
digital prediction = 0
optical prediction = 0
linear_relative_error = 5.03×10⁻¹⁵
imaginary_leakage_ratio = 2.27×10⁻¹⁵
circuit_size = 16
mzi_count = 240
```

保存的 session：

[progress/07_lumerical/live_digit_proof/session_5a8b4f9112.json](../07_lumerical/live_digit_proof/session_5a8b4f9112.json)

更早一次 124.4 s、预测 1/1、误差 `4.12×10⁻¹⁵` 的 session：

[progress/07_lumerical/live_digit_test/session_f317b6ad74.json](../07_lumerical/live_digit_test/session_f317b6ad74.json)

session 包含：

- 输入 28×28 图像、哈希；
- 冻结模型/PCA 的 SHA-256；
- 16 维 PCA 特征 `x`；
- 数字线性输出、logits、概率、预测；
- `g`、`s`、原始复光输出、恢复的复线性输出；
- 数字/光路预测与误差；
- 是否实际调用 INTERCONNECT（`live_interconnect_called`）；
- 运行耗时。

## 3. 启动界面

使用项目的普通 Python（界面本身只需要 NumPy 和标准库）：

```powershell
.\.venv\Scripts\python.exe lumerical\live_digit_ui.py
```

浏览器打开：

```text
http://127.0.0.1:8765/
```

界面支持：

- 鼠标/触控画板；
- 上传图片并缩放到画板；
- 清空；
- 三种模式：实际 INTERCONNECT / 缓存 H 演示 / 仅数字；
- 28×28 预处理预览；
- 16 维特征、数字/光路预测、线性误差、复泄漏、耗时；
- **光路证据面板**：LIVE/CACHED/DIGITAL 徽标、是否调用求解器、16 通道/MZI 数、build/run 耗时、`g/s`、16 路原始复输出、编码幅度/相位、数字 vs 光路逐通道条形图、session JSON 下载。
- 每次识别保存 session JSON 和 PGM 预览。

LIVE 徽标只在 `live_interconnect_called=true` 时出现；缓存 H 和仅数字分别是灰色 CACHED/DIGITAL，不会伪装成现场求解。

无 Lumerical 时可先快速看界面：

```powershell
.\.venv\Scripts\python.exe lumerical\live_digit_ui.py --digital-only
```

此时界面会禁用 live/cached，只显示数字基线，避免把未调用求解器的结果伪装成光路结果。

## 4. 命令行单张入口

```powershell
.\.venv\Scripts\python.exe lumerical\live_digit.py `
  --image path\to\image_28x28.pgm --mode live `
  --out progress\07_lumerical\live_digit
```

`--mode cached` / `--mode digital` 分别对应缓存 H 和仅数字。

## 5. 已验证与未验证

已验证：

- 空白画布会拒绝；
- 28×28 裁剪/缩放/居中；
- 模型/PCA 哈希校验；
- 16 通道 live 路径实际跑通一次；
- 数字/光路预测一致，线性相对误差约 `4×10⁻¹⁵`；
- 缓存 H 模式明确标记为非现场。

未验证/边界：

- 不是任意手写图片都能正确识别；91.99% 是官方测试集上的冻结模型结果，不是现场精度；
- 界面每次 live 识别都会重新建立 16 通道电路，首张约 124 秒；尚未实现会话级电路复用；
- 没有逐张跑 10,000 张 MNIST 的 INTERCONNECT 瞬态；
- 没有将 FDTD/EME 器件精度与这个应用绑定；
- 上传图片的预处理是黑底白字假设（画板默认白底黑字），复杂照片需要先手动调整。

## 6. 运行环境

- 界面：`.\.venv\Scripts\python.exe`，仅 NumPy + 标准库；
- live 模式：需要 Lumerical 2025 R2 的 `lumapi`，由 `lumerical.interconnect.load_lumapi()` 定位；
- 模型：`progress/06_mnist/models/seed_20261109.json`；
- PCA：`progress/06_mnist/preprocessor.npz`。
