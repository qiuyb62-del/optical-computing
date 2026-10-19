# 交给 AI 的统一标准

- 信号为复数光场 E=A exp(iφ)，A≥0，功率为 |E|²。传播 E，不传播功率；不自动归一化输入。
- 角度统一 rad，数据 complex128，输入 shape=(N,)，列向量约定，使用 U @ E。通道自上而下从 0 编号。
- MZI 唯一约定：T(theta,phi)=[[exp(i*phi)*cos(theta),-sin(theta)],[exp(i*phi)*sin(theta),cos(theta)]]。
- φ 先作用于第一输入端口；θ 是有效混合角，不等同于物理两臂相位差。不按 π 对 θ 取模，不忽略相位。
- mzi(E1,E2,theta,phi) 返回两个复数光场；mzi_matrix 返回 2×2 矩阵；apply_mzi(E,m,theta,phi) 返回新数组，更新相邻通道 (m,m+1)。
- 后续酉分解统一 Clements 矩形网格；Reck 仅为背景。ops 按传播顺序保存 (m,theta,phi)，另存 N 个输出端相位 α。
- 网格重建为 Uhat=D T_k ... T_1，D=diag(exp(iα))；实际传播 T_1 到 T_k 再 D。必须完整保留相位。
- 局部测试 atol=rtol=1e-12；酉矩阵重建相对 Frobenius 误差≤1e-10。必须比较完整复数结果，不仅比较功率，也不预先对齐全局相位。
- 理想模型通过后再引入非理想因素。后续 W=UΣV† 的传播顺序为 V† → Σ → U。
