"""Independent saved-array metrics and 2x2 four-group comparisons; no inference."""
import argparse
import json
from pathlib import Path
import numpy as np
from worker import bind

def measure(path):
    p=np.load(path/'predictions_f32.npy',mmap_mode='r')
    y=np.load(path/'targets_f32.npy',mmap_mode='r')
    assert p.shape==y.shape==(2817,64,7) and np.isfinite(p).all() and np.isfinite(y).all()
    e=p.astype(np.float64)-y
    mse=(e*e).mean((0,1)); mae=np.abs(e).mean((0,1))
    recorded=json.loads((path/'metrics.json').read_text())
    np.testing.assert_allclose(mse,recorded['variable_mse'],rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(mae,recorded['variable_mae'],rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose([mse.mean(),mae.mean()],[recorded['mse'],recorded['mae']],rtol=1e-10,atol=1e-12)
    return dict(mse=float(mse.mean()),mae=float(mae.mean()),variable_mse=mse.tolist(),variable_mae=mae.tolist(),
                ot_mse=float(mse[-1]),ot_mae=float(mae[-1]),source=str(path))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--project',type=Path,required=True); p.add_argument('--run',type=Path,required=True)
    a=p.parse_args(); t=bind(a.project)
    old=a.project/'runs/20261015_five_multivariate_oracle_continue_v1/ETTh1/jobs'
    refs=json.loads((a.run/'historical_references.json').read_text())
    reference=np.load(old/'dlinear_base_zero/targets_f32.npy')
    starts=np.load(old/'dlinear_base_zero/window_start.npy')
    expected=np.load(a.run/'llm_cache/test.npz')
    assert np.array_equal(starts,expected['starts'])
    measurements={}
    # Exactly 35 historical artifacts + 40 new artifacts, no duplicate scoring.
    for name,r in refs.items():
        path=Path(r['path'])
        assert t.sha(path/'best.pth')==r['checkpoint_sha256']
        assert t.sha(path/'config.json')==r['config_sha256'] and t.sha(path/'metrics.json')==r['metrics_sha256']
        assert np.array_equal(np.load(path/'targets_f32.npy'),reference)
        assert np.array_equal(np.load(path/'window_start.npy'),starts)
        measurements[name]=measure(path)
    pairs=[]; rows=[]
    for method in t.METHODS:
        for strategy in ('A','B'):
            for label in ('l1','l2'):
                suffix=f'{method}_{strategy}_{label}'
                row=dict(method=method,strategy=strategy,label=label,G3=measurements[suffix],
                         G4=measurements[method+'_base_zero'],zero=measurements[f'{method}_{strategy}_zero'])
                for group in ('G1','G2'):
                    path=a.run/'jobs'/f'{group}__{method}__{strategy}__{label}'
                    assert json.loads((path/'status.json').read_text())['status']=='complete'
                    assert np.array_equal(np.load(path/'targets_f32.npy'),reference)
                    assert np.array_equal(np.load(path/'window_start.npy'),starts)
                    assert np.array_equal(np.load(path/'labels.npy'),expected[label])
                    row[group]=measure(path)
                    if group=='G1':
                        ck=json.loads((path/'checkpoint_reference.json').read_text())
                        assert ck['sha256']==refs[suffix]['checkpoint_sha256'] and ck['weights_unchanged']
                    else:
                        check=json.loads((path/'paired_check.json').read_text())
                        assert check['initialization'] and check['common_epoch_order']
                        assert json.loads((path/'initial_hash.json').read_text())==json.loads((old/suffix/'initial_hash.json').read_text())
                        if strategy=='B':
                            assert check['frozen_backbone_verified']
                            assert json.loads((path/'config_alignment.json').read_text())['base_checkpoint_sha256']==refs[method+'_base_zero']['checkpoint_sha256']
                        pairs.append(dict(method=method,strategy=strategy,label=label,**check))
                row['relative_gains_pct']={f'{candidate}_vs_{ref}':{
                    metric:100*(row[ref][metric]-row[candidate][metric])/row[ref][metric]
                    for metric in ('mse','mae','ot_mse','ot_mae')}
                    for candidate,ref in [('G1','G4'),('G2','G4'),('G2','G1'),('G1','G3'),('G2','G3'),('G1','zero'),('G2','zero')]}
                rows.append(row)
    assert len(rows)==20 and len(pairs)==20
    t.dump(a.run/'comparison.json',dict(dataset='ETTh1',test_windows=2817,variables=['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT'],
           cpu_seed=2021,actual_cuda_seed=2022,rows=rows))
    t.dump(a.run/'paired_checks.json',dict(passed=True,pairs=pairs,count=20))
    t.dump(a.run/'verification.json',dict(passed=True,new_jobs=40,reused_jobs=35,core_unique_results=65,
           exact_targets=True,exact_windows=True,exact_LLM_conditions=True,all_metrics_recomputed=True,
           historical_artifacts_unchanged=True,paired_count=20))
    lines=['# ETTh1 四组比较：冻结 LLM 标签', '',
        '所有结果为原始测试集 2817 个窗口、7变量、512→64，标准化空间；MSE/MAE 越低越好。',
        'G1=旧 oracle 模型+LLM测试标签；G2=LLM标签训练/验证/测试；G3=旧oracle；G4=旧基线。',
        'G3 使用真实未来离散标签，非可部署结果。单 seed；CPU2021/CUDA2022；每32窗口共享标签。',
        '未调参、未新增API调用、未用测试选模型；G2以LLM条件下验证数值MSE选择。', '']
    for strategy in ('A','B'):
        for label in ('l1','l2'):
            lines += [f'## {strategy}-{label.upper()}', '',
                '|模型|G1 MSE/MAE|G2 MSE/MAE|G3 MSE/MAE|G4 MSE/MAE|G1 vs G4 MSE改善|G2 vs G4 MSE改善|G2 vs G1 MSE改善|',
                '|---|---|---|---|---|---|---|---|']
            for row in rows:
                if row['strategy']!=strategy or row['label']!=label: continue
                cells=[row['method']]+[f"{row[g]['mse']:.6f}/{row[g]['mae']:.6f}" for g in ('G1','G2','G3','G4')]
                cells += [f"{row['relative_gains_pct'][g]['mse']:+.2f}%" for g in ('G1_vs_G4','G2_vs_G4','G2_vs_G1')]
                lines.append('| '+' | '.join(cells)+' |')
            lines += ['', '|模型|G1 OT MSE/MAE|G2 OT MSE/MAE|G3 OT MSE/MAE|G4 OT MSE/MAE|','|---|---|---|---|---|']
            for row in rows:
                if row['strategy']==strategy and row['label']==label:
                    lines.append('| '+' | '.join([row['method']]+[f"{row[g]['ot_mse']:.6f}/{row[g]['ot_mae']:.6f}" for g in ('G1','G2','G3','G4')])+' |')
            lines.append('')
    lines += ['完整逐变量及 matched-zero 对照见 comparison.json；20组初始化/共同epoch采样顺序见 paired_checks.json。',
              '验证集曾用于prompt开发；测试标签F1在本轮前已知，本轮固定配方不据此调整。Oracle是参考，不保证严格上界。']
    (a.run/'summary.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__': main()
