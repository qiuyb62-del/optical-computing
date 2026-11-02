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
- 理想模型通过后再引入非理想因素。损耗、相位噪声、量化及 AI 扩展尚未实现，不得把数值舍入误差当成硬件实验误差。

接口与推导见 clements-decomposition.md 和 svd-matmul.md。修改代码前先使用本标准；不得因参考论文或其他库的公式不同而悄悄替换矩阵 convention。
