"""第四阶段：同一 W 和 x，在理想光路与固定非理想器件上比较。"""
import argparse
import json
from pathlib import Path
import numpy as np
from src.svd_matmul import compile_svd, trace_svd
from src.nonideal import NonidealConfig, realize_nonideal, trace_nonideal
from src.svd_validation import complex_data, comparison


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=20261109)
    parser.add_argument('--phase-std',type=float,default=.01,help='静态控制相位误差标准差 rad')
    parser.add_argument('--loss-db',type=float,default=.1,help='每个 MZI 插入功率损耗 dB')
    parser.add_argument('--bits',type=int,default=8,help='相位控制位数，0 表示不量化')
    parser.add_argument('--coupler',type=float,default=.49,help='耦合器直通功率比')
    parser.add_argument('--upper-arm-loss-db',type=float,default=0.)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'progress/04_nonidealities/example_4x4.json')
    args=parser.parse_args()
    try:
        if args.seed<0: raise ValueError('seed 必须非负')
        config=NonidealConfig(args.phase_std,args.loss_db,None if args.bits==0 else args.bits,
                              args.coupler,args.upper_arm_loss_db)
    except ValueError as exc: parser.error(str(exc))
    rng=np.random.default_rng(args.seed)
    W=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4))
    x=rng.normal(size=4)+1j*rng.normal(size=4)
    ideal=compile_svd(W)
    device=realize_nonideal(ideal,config,seed=args.seed+1)
    good,bad=trace_svd(x,ideal),trace_nonideal(x,device)
    target=W@x
    scale=float(np.linalg.norm(W)*np.linalg.norm(x))
    def data(trace):
        names=['after_input_mesh','after_attenuation','optical_output','restored_output']
        return {name:complex_data(getattr(trace,name)) for name in names}
    def powers(trace):
        return {name:float(np.linalg.norm(v)**2) for name,v in
                [('input',x),('after_input_mesh',trace.after_input_mesh),
                 ('after_attenuation',trace.after_attenuation),('optical_output',trace.optical_output)]}
    report={'seed':args.seed,'W':complex_data(W),'x':complex_data(x),
            'reference':complex_data(target),'ideal_circuit':ideal.to_dict(),
            'realized_device':device.to_dict(),'ideal_fields':data(good),'nonideal_fields':data(bad),
            'ideal_powers':powers(good),'nonideal_powers':powers(bad),
            'ideal_comparison':comparison(good.restored_output,target,reference_scale=scale),
            'nonideal_comparison':comparison(bad.restored_output,target,reference_scale=scale),
            'interpretation':'Nonideal error above the ideal tolerance is an experimental outcome, not a failed implementation.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    np.set_printoptions(precision=5,suppress=True)
    print('第四阶段：固定器件误差；以下 y 已数字恢复比例。')
    print('目标 NumPy Wx:',target)
    print('理想光路输出:',good.restored_output)
    print('非理想光路输出:',bad.restored_output)
    print('理想相对误差:',f"{report['ideal_comparison']['relative_error']:.3e}")
    print('非理想相对误差:',f"{report['nonideal_comparison']['relative_error']:.3e}")
    print('原始光学功率（输入/输出）:',report['nonideal_powers']['input'],report['nonideal_powers']['optical_output'])
    print('数字恢复比例 s:',ideal.recovery_scale,'；不表示光学放大。')
    print('完整参数与结果:',args.output)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
