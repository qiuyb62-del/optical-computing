"""第四阶段配对实验：单因素误差扫描、被动性检查、理想回归及可复现汇总。"""
import argparse
import csv
from dataclasses import asdict
import json
import platform
from pathlib import Path
import numpy as np
from src.svd_matmul import compile_svd
from src.nonideal import NonidealConfig, realize_nonideal, trace_nonideal, reconstruct_nonideal
from src.svd_validation import comparison, TOLERANCE
from verify_week03 import special_matrices


def settings():
    return [('ideal','baseline',0.,NonidealConfig())]+[
        (f'phase_{v:g}','phase',v,NonidealConfig(phase_std_rad=v)) for v in [.001,.01,.03,.1]]+[
        (f'loss_{v:g}','loss',v,NonidealConfig(loss_db_per_mzi=v)) for v in [.1,.3,.5]]+[
        (f'bits_{v}','quantization',v,NonidealConfig(phase_bits=v)) for v in [4,6,8]]+[
        (f'coupler_{v:g}','coupler',v,NonidealConfig(coupler_through=v)) for v in [.49,.47,.45]]+[
        ('upper_arm_loss_0.3','arm_loss',.3,NonidealConfig(upper_arm_loss_db=.3)),
        ('combined','combined',None,NonidealConfig(phase_std_rad=.01,loss_db_per_mzi=.1,
                                                 phase_bits=8,coupler_through=.49))]


def stats(values):
    values=np.asarray(values,float)
    if values.size==0 or not np.all(np.isfinite(values)):
        raise ValueError('statistics require finite nonempty data')
    return {'count':int(values.size),'p05':float(np.percentile(values,5)),
            'median':float(np.median(values)),'p95':float(np.percentile(values,95)),
            'max':float(np.max(values))}


def power_check(x,trace):
    powers=np.asarray([np.vdot(v,v).real for v in
                       [x,trace.after_input_mesh,trace.after_attenuation,trace.optical_output]],float)
    passed=bool(np.all(np.isfinite(powers)) and np.all(np.diff(powers)<=TOLERANCE*powers[0]))
    throughput=float(powers[-1]/powers[0]) if powers[0]>0 else None
    return passed,throughput


def targets(n,count,seed):
    """目标和输入独立于器件参数，所有条件共享同一组样本。"""
    rng=np.random.default_rng(np.random.SeedSequence([seed,n,0]))
    result=[]
    for k in range(count):
        W=rng.normal(size=(n,n)).astype(complex)
        if k%2: W+=1j*rng.normal(size=(n,n))
        W*= [.1,1.,10.][k%3]
        inputs=list(np.eye(n,dtype=complex))
        inputs += [rng.normal(size=n)+1j*rng.normal(size=n) for _ in range(2)]
        result.append((W,compile_svd(W),inputs))
    return result


def ideal_special_check(n,seed):
    failures=[]
    matrix_max=0.
    for name,W in special_matrices(n):
        device=realize_nonideal(compile_svd(W),seed=seed)
        H=reconstruct_nonideal(device)
        metric=comparison(H,W)
        matrix_max=max(matrix_max,metric['relative_error'] or 0.)
        for x in [*np.eye(n,dtype=complex),np.zeros(n,complex),np.ones(n,complex)*(1+1j)]:
            trace=trace_nonideal(x,device)
            scale=float(np.linalg.norm(W)*np.linalg.norm(x))
            vector=comparison(trace.restored_output,W@x,reference_scale=scale)
            passive,_=power_check(x,trace)
            if not (metric['passed'] and vector['passed'] and passive):
                failures.append({'case':name,'matrix':metric,'vector':vector,'passive':passive})
    return {'size':n,'cases':7,'max_matrix_relative_error':matrix_max,'passed':not failures,'failures':failures}


def run_experiments(*,sizes=(2,4,8),samples=20,trials=30,seed=20261109,verbose=False):
    if (samples<1 or trials<1 or seed<0 or not sizes or
            len(set(sizes))!=len(sizes) or any(n<1 for n in sizes)):
        raise ValueError('sizes must be unique positive integers; samples/trials positive; seed nonnegative')
    rows=[]
    special=[]
    for n in sizes:
        cases=targets(n,samples,seed)
        special.append(ideal_special_check(n,seed))
        for name,factor,value,config in settings():
            matrix_errors,vector_errors,throughputs=[],[],[]
            failures=0
            passive=True
            worst=None
            for k,(W,ideal,inputs) in enumerate(cases):
                for trial in range(trials):
                    device_seed=int(np.random.SeedSequence([seed,n,k,trial,1]).generate_state(1,dtype=np.uint64)[0])
                    device=realize_nonideal(ideal,config,seed=device_seed)
                    H=reconstruct_nonideal(device)
                    matrix=comparison(H,W)
                    matrix_errors.append(matrix['relative_error'])
                    if worst is None or matrix['relative_error']>worst['matrix_relative_error']:
                        worst={'matrix_index':k,'trial':trial,'device_seed':device_seed,
                               'matrix_relative_error':matrix['relative_error']}
                    for x in inputs:
                        trace=trace_nonideal(x,device)
                        vector=comparison(trace.restored_output,W@x,
                                          reference_scale=float(np.linalg.norm(W)*np.linalg.norm(x)))
                        vector_errors.append(vector['relative_error'])
                        failures += int(not vector['passed'])
                        ok,throughput=power_check(x,trace)
                        passive=passive and ok
                        throughputs.append(throughput)
            row={'size':n,'condition':name,'factor':factor,'setting':value,'config':asdict(config),
                 'matrix_count':samples,'realizations':samples*trials,
                 'vector_cases':len(vector_errors),'matrix_relative_error':stats(matrix_errors),
                 'vector_relative_error':stats(vector_errors),'power_throughput':stats(throughputs),
                 'fraction_exceeding_ideal_vector_tolerance':failures/len(vector_errors),
                 'passivity_passed':bool(passive),'worst_matrix_case':worst}
            row['validation_passed']=bool(passive and
                (name!='ideal' or (row['matrix_relative_error']['max']<=TOLERANCE and failures==0)))
            rows.append(row)
            if verbose:
                print(f"N={n} {name:20s} median matrix error={row['matrix_relative_error']['median']:.3e}; "
                      f"passive={passive}",flush=True)
    return {'model':'passive_physical_phase_v1','python':platform.python_version(),'numpy':np.__version__,
            'seed':seed,'samples_per_size':samples,'realizations_per_matrix':trials,'sizes':list(sizes),
            'phase_noise':'independent static physical control offsets, added after quantization',
            'paired_conditions':True,'vector_inputs':'N basis inputs plus two independent random complex vectors',
            'ideal_tolerance':TOLERANCE,'percentiles':'empirical 5th/95th percentiles, not confidence intervals',
            'nonideal_interpretation':'Exceeding ideal numerical tolerance is expected; validation checks passivity and ideal recovery.',
            'special_ideal_checks':special,'conditions':rows,
            'passed':bool(all(v['validation_passed'] for v in rows) and all(v['passed'] for v in special))}


def save_csv(report,path):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f,lineterminator='\n')
        writer.writerow(['size','condition','realizations','vectors','matrix_error_p05','matrix_error_median',
                         'matrix_error_p95','matrix_error_max','vector_error_median','vector_error_p95',
                         'throughput_median','fraction_exceeding_ideal_tolerance','passivity_passed'])
        for r in report['conditions']:
            m,v=r['matrix_relative_error'],r['vector_relative_error']
            writer.writerow([r['size'],r['condition'],r['realizations'],r['vector_cases'],m['p05'],m['median'],m['p95'],
                             m['max'],v['median'],v['p95'],r['power_throughput']['median'],
                             r['fraction_exceeding_ideal_vector_tolerance'],r['passivity_passed']])


def save_svg(report,path):
    """无绘图库依赖的静态科学曲线；实际汇总数据驱动，离散横坐标明确标注。"""
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="850" viewBox="0 0 1200 850">',
           '<rect width="1200" height="850" fill="white"/>',
           '<g font-family="Arial, sans-serif" fill="#111">',
           '<text x="600" y="32" text-anchor="middle" font-size="23">Week 4 — static device errors</text>',
           '<text x="600" y="57" text-anchor="middle" font-size="13">Median relative Frobenius matrix error; vertical ranges show empirical 5th–95th percentiles</text>']
    panels=[('phase','Phase standard deviation (rad)',['phase_0.001','phase_0.01','phase_0.03','phase_0.1'],['.001','.01','.03','.1']),
            ('loss','Insertion power loss per MZI (dB)',['loss_0.1','loss_0.3','loss_0.5'],['.1','.3','.5']),
            ('quantization','Phase control resolution (bits)',['bits_8','bits_6','bits_4'],['8','6','4']),
            ('coupler','Coupler through power fraction',['coupler_0.49','coupler_0.47','coupler_0.45'],['.49','.47','.45'])]
    colors=['#0072B2','#D55E00','#009E73','#CC79A7']
    rows={(r['size'],r['condition']):r for r in report['conditions']}
    for idx,(_,title,conditions,labels) in enumerate(panels):
        left=75+(idx%2)*590; top=105+(idx//2)*340; width=485; height=240
        def y(v): return top+height*(1-np.clip((np.log10(max(v,1e-4))+4)/5,0,1))
        parts.append(f'<text x="{left+width/2}" y="{top-18}" text-anchor="middle" font-size="17">{title}</text>')
        for exponent in [-4,-3,-2,-1,0,1]:
            yy=y(10.**exponent)
            parts.append(f'<line x1="{left}" y1="{yy}" x2="{left+width}" y2="{yy}" stroke="#ddd"/>')
            parts.append(f'<text x="{left-8}" y="{yy+4}" text-anchor="end" font-size="12">1e{exponent}</text>')
        for j,label in enumerate(labels):
            x=left+width*j/(len(labels)-1)
            parts.append(f'<text x="{x}" y="{top+height+22}" text-anchor="middle" font-size="13">{label}</text>')
        for ni,n in enumerate(report['sizes']):
            color=colors[ni%len(colors)]
            points=[]
            for j,c in enumerate(conditions):
                r=rows[n,c]['matrix_relative_error']; xx=left+width*j/(len(labels)-1); yy=y(r['median'])
                points.append(f'{xx},{yy}')
                parts.append(f'<line x1="{xx}" y1="{y(r["p05"])}" x2="{xx}" y2="{y(r["p95"])}" stroke="{color}" stroke-width="2" opacity="0.5"/>')
                parts.append(f'<circle cx="{xx}" cy="{yy}" r="4" fill="{color}"/>')
            points=' '.join(points)
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    baseline=max(r['matrix_relative_error']['max'] for r in report['conditions'] if r['condition']=='ideal')
    parts.append(f'<text x="600" y="752" text-anchor="middle" font-size="13">Ideal baseline max error = {baseline:.2e}; shown separately to keep device-error differences readable</text>')
    for ni,n in enumerate(report['sizes']):
        xx=440+ni*130
        parts.append(f'<circle cx="{xx}" cy="786" r="5" fill="{colors[ni%len(colors)]}"/><text x="{xx+12}" y="791" font-size="15">N = {n}</text>')
    parts.append('<text x="600" y="820" text-anchor="middle" font-size="13">Tested settings are equally spaced on each x axis. y axes are logarithmic. No hardware measurements.</text></g></svg>')
    path.write_text('\n'.join(parts),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=int,default=20)
    parser.add_argument('--trials',type=int,default=30)
    parser.add_argument('--seed',type=int,default=20261109)
    parser.add_argument('--sizes',type=int,nargs='+',default=[2,4,8])
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'progress/04_nonidealities')
    args=parser.parse_args()
    try:
        report=run_experiments(sizes=args.sizes,samples=args.samples,trials=args.trials,seed=args.seed,verbose=True)
    except ValueError as exc: parser.error(str(exc))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    save_csv(report,args.output_dir/'summary.csv')
    save_svg(report,args.output_dir/'sensitivity.svg')
    print('验证:', 'PASS' if report['passed'] else 'FAIL','；输出目录:',args.output_dir)
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
