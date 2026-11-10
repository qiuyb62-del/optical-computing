# 统一标准

- 信号为复数光场 E=A exp(iφ)，A≥0，功率为 |E|²。传播 E，不传播功率；不自动归一化输入。
- 角度统一 rad，数据 complex128，输入 shape=(N,)，列向量约定，使用 U @ E。通道自上而下从 0 编号。
- MZI 唯一约定：T(theta,phi)=[[exp(i*phi)*cos(theta),-sin(theta)],[exp(i*phi)*sin(theta),cos(theta)]]。
- φ 先作用于第一输入端口；θ 是有效混合角，不等同于物理两臂相位差。不按 π 对 θ 取模，不忽略相位。
- mzi(E1,E2,theta,phi) 返回两个复数光场；mzi_matrix 返回 2×2 矩阵；apply_mzi(E,m,theta,phi) 返回新数组，更新相邻通道 (m,m+1)。
- 酉分解统一 Clements 矩形网格；Reck 仅为背景。decompose_clements(U) 返回 (ops, output_phases)。ops 按传播顺序保存 (m,theta,phi)，另存 N 个输出端相位 α，不能省略。
- 分解输入 U 必须满足 ||U†U−I||_F≤1e-10；不满足时拒绝，不自动投影。支持 N=1；保留 N(N−1)/2 个 MZI。分解输出 phi 和 α 位于 [0,2π)，theta 可以为负。
- 网格重建为 Uhat=D T_k ... T_1，D=diag(exp(iα))；实际传播 T_1 到 T_k 再 D。必须完整保留相位。
- propagate_mesh(E,ops,output_phases) 逐器件传播；reconstruct_mesh(ops,output_phases) 重建矩阵；mesh_layers(ops,output_phases) 给出互不共享通道的层。所有接口不修改输入数组。
- 局部测试 atol=rtol=1e-12；酉矩阵重建相对 Frobenius 误差≤1e-10。必须比较完整复数结果，不仅比较功率，也不预先对齐全局相位。
- 一般矩阵阶段只接收有限、非空的实数/复数方阵 W；SVD 返回 U、sigma、Vh，其中 Vh 已是 V†，不得再次转置。长方形矩阵与张量暂不支持。
- `compile_svd(W)` 分别对 V†、U 做 Clements 分解，返回只读 SVDCircuit。两套网格共 N(N−1) 个 MZI，配置可复用于多个 x。不得改变已确定的 MZI convention。
- 被动衰减统一 s=max(1,sigma_max)、t_i=sigma_i/s。实际传播顺序为 V† 网格（含输出相位）→ diag(t) → U 网格（含输出相位），得到 y_optical≈(W/s)x；随后数字恢复 y=s*y_optical。s 必须保存，数字恢复不能描述为光学放大。
- t_i 是幅度透过率，功率透过率为 t_i²。零矩阵使用 s=1、t=0；不截断非零小奇异值，不自动归一化输入。各网格守恒功率，被动光路整体不增加功率；恢复后的数值不作为实际光功率。
- `propagate_svd(x,circuit)` 默认返回恢复后的结果；`restore_scale=False` 返回原始光学输出。`trace_svd` 返回各阶段光场；`reconstruct_svd` 仅供验证。实际传播必须逐器件计算，不使用稠密 W@x 或 U@x 替代光路。
- 一般矩阵重建相对 Frobenius 误差、普通复数输出相对误差均≤1e-10。W=0 时查绝对误差≤1e-10。令 B=||W||_F||x||_2，若 ||Wx||=0 或 ≤1e-12*B，改查绝对输出误差≤1e-10*max(1,B)，单独报告；零分母相对误差记 null。原始光学输出按 W/s 和 B/s 同样检查，不对齐全局相位。
- 第四阶段在 `src/nonideal.py` 中加入静态相位偏差、损耗、物理控制相位量化和耦合器偏差。原理想接口不变；不得把数值舍入误差或人为设定的器件误差当成硬件测量结果。第五阶段已加入 AI 推理，Lumerical 尚未实现。
- 非理想 MZI 映射固定为 `Dχ B† diag(exp(iδ),1) B diag(exp(iφ),1)`；理想 `B=[[1,i],[i,1]]/sqrt(2)`，`δ=2θ, χ_upper=χ_lower=−θ`。补偿相位不能省略，输出耦合器相位约定不能混用。
- 量化作用于物理相位 `(φ,δ,χ_upper,χ_lower)` 和网格输出相位；步长 `2π/2^b`，最近格点，中点 round-to-even。先量化命令再加入独立静态高斯控制相位偏差。有效 θ 不直接量化。
- `realize_nonideal(ideal,config,seed=...)` 只采样一次；`propagate_nonideal` 按固定 2×2 器件逐个传播，输入多个向量时不重新采样。不同条件采用相同 seed 做配对对照，不使用 NumPy 全局随机状态。
- 额外功率损耗 L dB 对应场幅度乘 `10^(-L/20)`；coupler_through 是直通功率比，非幅度。相位误差和量化不应破坏无损单元的酉性，有损光路只检查功率不增；数字恢复不计入物理功率检查。
- 非理想误差超过理想阈值是实验结果，不能自动判成代码失败。必须另验零误差恢复、被动性、固定器件线性和随机种子复现。分位范围不是置信区间。

- 第五阶段 AI 验证使用固定的 UCI bezdekIris 数据和 4→4→3 网络，只将第一层 W1 编译成 4×4 光路；完整保留为基础示例。偏置、ReLU 和输出分类头保持数字计算。PyTorch 权重为 W[out,in]；批量数字参考 X@W1.T，光路逐输入 W1@x，不得错误转置。
- 标准化参数仅从训练集拟合，早停和权重选择只使用验证集；测试集仅评估。保存数据哈希、划分行号、权重、训练种子及器件 seed。不能按测试准确率挑选 seed 或报告最好的一次。
- AI 读出使用共同相位参考下恢复复光场的实部，单独报告虚部；不使用幅度或强度替代，也不逐样本对齐相位。当前假设可读复场，不声称实现真实相干接收器。
- 每个训练模型只编译一次；每台静态非理想器件处理整个测试集，期间不重新采样。比较完整复线性输出、ReLU 后激活、logits、预测一致率与分类准确率。准确率变化用百分点，不等同于数值相对误差。
- 第五阶段 AI 实验当时不含校准、噪声训练或 Lumerical；第七阶段已把 Iris/MNIST 第一层 H 放进 Lumerical，但仍不包含校准和噪声训练。图表的经验分位数不是置信区间，确定性重复 seed 不算独立制造样本。固定环境提高复现性但不保证不同平台训练逐位一致。

- 第六阶段新增 MNIST，不替换 Iris。官方 60,000/10,000 图片；前者划分 50,000 训练、10,000 验证。图片除以 255，仅训练集拟合 PCA16，不白化。16→16→10 网络只将第一层 16×16 权重编译为光路，两套网格共 240 个 MZI。具体种子、早停、误差条件和数据文件见 mnist-validation.md。
- 批量光场 shape=(samples,N)，每一行代表一个独立列向量输入的转置；每个 2×2 MZI 作用于相邻两列，依次应用所有单元及输出相位。批量接口必须与逐样本传播相符，不能通过重建稠密矩阵走捷径。
- MNIST 使用整个官方测试集评估；20 张固定展示图片只用于解释流程，不能把其准确率当作全测试集性能。所有结果分阶段保存到 progress/05_ai_extension 和 progress/06_mnist。

- 第七阶段 Lumerical 实现沿用上述全部约定。实测 `Waveguide Coupler` 的 `conjugate=0` 为 `B`、`conjugate=1` 为 `B†`；`coupling coefficient` 是交叉耦合功率，项目直通功率 η 对应 `coefficient=1−η`；`Optical Phase Shift` 为 `exp(+i phase)`。MZI 固定为 `P(φ)→B→P(2θ)→B†→P(−θ)`，不得用同一个 B 连接两个耦合器。
- Lumerical 频域测试只使用 ONA/SPS；ONA 的实际频点为 `start+n(stop−start)/N`，读取复 S21 做相位比较时必须使用结果中的实际频率。时域读出使用共同 CW、0/π2 LO 和平衡 PD；`Fork 1xN` 是理想场复制，只作测试台扇出。16 通道 H 回灌属于“Lumerical 传输矩阵提取后的混合推理”，不能声称每个样本都在 Lumerical 中做了瞬态仿真。
- Lumerical 非理想实验必须从 Python `RealizedMesh` 的固定 actual phases 写入，不能按输入重新采样；耦合比、损耗、量化、耦合器相位的映射见 docs/lumerical-guide.md。

接口与推导见 clements-decomposition.md、svd-matmul.md、nonidealities.md、ai-validation.md、mnist-validation.md 和 lumerical-guide.md。修改代码前先使用本标准；不得因参考论文或其他库的公式不同而悄悄替换矩阵 convention。
