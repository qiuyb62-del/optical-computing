# 0.625 nm FDE 尝试：资源限制未完成

日期：2026-11-12

命令：

```powershell
& $py lumerical\run_device_fde_local.py --steps-nm 10 5 2.5 1.25 0.625 `
  --domain-um 3 2 --boundary metal --trial-modes 10 `
  --out progress\07_lumerical\round4\fde_mesh_0p625
```

现象：

- `mode-solutions` 工作集约 **5.16 GB**；
- 运行超过 10 分钟仍未返回；
- 已终止，未生成有效 `device_fde_local.json`；
- 该档没有进入任何收敛结论。

这不是隐含通过；只是本机资源/时间不足。要完成最后两个连续细化区间，需要在内存更充足的机器上重跑 10/5/2.5/1.25/0.625 nm 全系列，或先优化网格/模式数。
