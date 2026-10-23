# -*- coding: utf-8 -*-
"""Build an offline, single-file research report from verified experiment metrics."""
import json,html,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
load=lambda p:json.loads((ROOT/p).read_text())
e1=load('runs/20261019_dao_etth1_allsplits_v1/metrics.json')
e2=load('runs/20261019_etth1_llm_downstream_v1/comparison.json')
e3=load('runs/20261019_etth1_oracle_noise_v2/comparison.json')
assert len(e2['rows'])==20 and len(e3['rows'])==100
names={'dlinear':'DLinear','patchtst':'PatchTST','itransformer':'iTransformer','frets':'FreTS','fredf':'iTransformer + FreDF'}
schemes=[('A','l1'),('A','l2'),('B','l1'),('B','l2')]
f=lambda x:f'{x:.6f}'
def table(headers,rows,caption=''):
    return '<div class="table-wrap"><table>'+('<caption>'+caption+'</caption>' if caption else '')+'<thead><tr>'+''.join('<th scope="col">'+str(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(('<th scope="row">' if i==0 else '<td>')+str(x)+('</th>' if i==0 else '</td>') for i,x in enumerate(row))+'</tr>' for row in rows)+'</tbody></table></div>'
agg=[]
for s,l in schemes:
    rs=[r for r in e2['rows'] if r['strategy']==s and r['label']==l]
    agg.append({'scheme':s+'-'+l.upper(),**{g:sum(r[g]['mse'] for r in rs)/5 for g in ['G1','G2','G3','G4','zero']}})
base=agg[0]['G4']
for a,(s,l) in zip(agg,schemes):
    v=next(v for v in e3['means'] if v['strategy']==s and v['label']==l and v['accuracy']==100)
    assert abs(v['mse']-a['G3'])<1e-12
labelrows=[]
for split,title,n in [('train','训练集',8065),('val','验证集',2817),('test','测试集',2817)]:
    m=e1['metrics'][split]
    labelrows.append([title,f'{n:,}',f"{m['L1']['macro_f1']:.4f}",f"{m['L1']['accuracy']:.2%}",f"{m['L2']['macro_f1']:.4f}",f"{m['L2']['accuracy']:.2%}"])
labeltable=table(['切分','窗口数','L1 Macro-F1','L1 Accuracy','L2 Macro-F1','L2 Accuracy'],labelrows,'表 1｜方向标签分类性能。按每个窗口 × 变量统计，固定三类 Macro-F1。')
grouptable=table(['方案','G1 直接替换','G2 重训','G4 原始基线','G3 Oracle'],[[r['scheme']]+[f(r[g]) for g in ['G1','G2','G4','G3']] for r in agg],'表 2｜五模型等权平均的全变量测试 MSE（↓）。每个方案独立比较。')
noiserows=[]
for s,l in schemes:
    noiserows.append([s+'-'+l.upper()]+[f(next(r['mse'] for r in e3['means'] if (r['strategy'],r['label'],r['accuracy'])==(s,l,k))) for k in [60,70,80,90,100]])
noisetable=table(['方案','60%','70%','80%','90%','100% Oracle'],noiserows,'表 3｜控制标签正确率后的五模型平均全变量 MSE（↓）。100% 复用表 2 的 G3。')
colors=['#1c527e','#b9702d','#2c7767','#81577d']
svg=['<svg class="chart" viewBox="0 0 840 330" role="img" aria-labelledby="noise-title noise-desc"><title id="noise-title">标签正确率与下游平均MSE</title><desc id="noise-desc">四个方案均随正确率提高而降低MSE。90%时均低于原始基线，80%时均高于基线。数值见表3。</desc>']
x=lambda a:80+(a-60)/40*680
y=lambda v:260-(v-.33)/.09*210
for v in [.34,.36,.38,.40,.42]:svg.append(f'<path d="M80 {y(v)}H760" stroke="#e3e7eb"/><text x="67" y="{y(v)+4}" text-anchor="end">{v:.2f}</text>')
for k in [60,70,80,90,100]:svg.append(f'<text x="{x(k)}" y="285" text-anchor="middle">{k}%</text>')
svg.append(f'<path d="M80 {y(base)}H760" stroke="#5f6670" stroke-dasharray="6 5"/><text x="765" y="{y(base)-7}" text-anchor="end">基线 {base:.6f}</text>')
for i,(s,l) in enumerate(schemes):
    points=[(k,next(r['mse'] for r in e3['means'] if (r['strategy'],r['label'],r['accuracy'])==(s,l,k))) for k in [60,70,80,90,100]]
    svg.append(f'<polyline points="'+ ' '.join(f'{x(k)},{y(v)}' for k,v in points)+f'" fill="none" stroke="{colors[i]}" stroke-width="2.5"/>')
    for k,v in points:svg.append(f'<circle cx="{x(k)}" cy="{y(v)}" r="4" fill="{colors[i]}"><title>{s}-{l.upper()} · {k}% · {v:.6f}</title></circle>')
    svg.append(f'<rect x="{90+i*170}" y="12" width="18" height="3" fill="{colors[i]}"/><text x="{115+i*170}" y="18">{s}-{l.upper()}</text>')
svg.append('<text x="420" y="317" text-anchor="middle">标签正确率（非错误率）</text><text x="80" y="41">平均 MSE ↓</text></svg>')
chart=''.join(svg)
details=[]
for s,l in schemes:
    rs=[r for r in e2['rows'] if (r['strategy'],r['label'])==(s,l)]
    body=table(['模型','G1','G2','G3 Oracle','G4 基线','匹配 zero'],[[names[r['method']]]+[f(r[g]['mse'])+' / '+f(r[g]['mae']) for g in ['G1','G2','G3','G4','zero']] for r in rs],'全变量 MSE / MAE')
    body+=table(['模型','G1 OT','G2 OT','G3 OT','G4 OT'],[[names[r['method']]]+[f(r[g]['ot_mse'])+' / '+f(r[g]['ot_mae']) for g in ['G1','G2','G3','G4']] for r in rs],'OT 单变量 MSE / MAE')
    details.append(f'<details><summary>{s}-{l.upper()} · 四组预测及 zero 对照</summary>{body}</details>')
noiseappend=[]
for s,l in schemes:
    rows=[];ot=[]
    for m in names:
        rs=[next(r for r in e3['rows'] if (r['method'],r['strategy'],r['label'],r['accuracy'])==(m,s,l,k)) for k in [60,70,80,90,100]]
        rows.append([names[m]]+[f(r['mse'])+' / '+f(r['mae']) for r in rs])
        ot.append([names[m]]+[f(r['ot_mse'])+' / '+f(r['ot_mae']) for r in rs])
    noiseappend.append(f'<details><summary>{s}-{l.upper()} · 五档正确率敏感性</summary>'+table(['模型','60%','70%','80%','90%','100%'],rows,'全变量 MSE / MAE')+table(['模型','60%','70%','80%','90%','100%'],ot,'OT 单变量 MSE / MAE')+'</details>')
variables=e2['variables'];varrows=[]
for v in variables:
    a=e1['metrics']['test']['L1']['per_variable'][v];b=e1['metrics']['test']['L2']['per_variable'][v]
    varrows.append([v,f"{a['macro_f1']:.4f}",f"{a['accuracy']:.2%}",f"{b['macro_f1']:.4f}",f"{b['accuracy']:.2%}"])
varappend=table(['变量','L1 F1','L1 Accuracy','L2 F1','L2 Accuracy'],varrows,'原测试集逐变量方向分类性能')
classrows=[]
for l in ['L1','L2']:
    m=e1['metrics']['test'][l]
    for k in ['-1','0','1']:
        a=m['per_class'][k];classrows.append([l,k,a['support'],f"{a['f1']:.4f}"])
classtable=table(['标签','类别','测试样本数','类别 F1'],classrows,'类别诊断：总体准确率不能替代三类 Macro-F1')
manifest=[('E1','20261019_dao_etth1_allsplits_v1','431 anchors；13,699 窗口；95,893 行'),('E2','20261019_etth1_llm_downstream_v1','20 推理 + 20 训练；复用 35 份历史结果'),('E3','20261019_etth1_oracle_noise_v2','80 推理；复用 20 Oracle；训练 0')]
CSS='''*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:30px}body{margin:0;background:#eef0f3;color:#20262e;font:16px/1.8 Georgia,"Songti SC","Noto Serif CJK SC",serif}a{color:#1c527e;text-decoration:none}a:hover{text-decoration:underline}.sidebar{position:fixed;top:0;left:0;width:220px;height:100vh;padding:38px 24px;background:#182f43;color:#d9e4ec;font:13px/1.6 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}.brand{font-size:12px;letter-spacing:2px;color:#9db7c9}.sidebar strong{display:block;color:white;font-size:20px;margin:14px 0 26px}.sidebar a{display:block;color:#d9e4ec;padding:9px 0;border-bottom:1px solid #ffffff12}.sidebar .small{margin-top:28px;color:#a8bccb;font-size:11px}.paper{margin:30px 30px 70px 250px;max-width:1120px;background:white;padding:64px 70px;box-shadow:0 6px 28px #2030480b}.eyebrow,.meta,caption,figcaption,summary,th,.label,.pill,button{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}.eyebrow{font-size:11px;letter-spacing:2px;text-transform:uppercase;color:#6c7c89}.rule{height:4px;background:#183f5d;width:55px;margin:20px 0}h1{font-size:35px;line-height:1.4;letter-spacing:-.5px;margin:14px 0}h1 span{display:block;font:19px/1.6 Georgia,serif;color:#6a7883;margin-top:12px}.meta{font-size:12px;color:#78818b;margin:25px 0}.abstract{padding:25px 28px;background:#f5f7f9;border-left:3px solid #244f70;margin:30px 0}.abstract h2{margin:0 0 10px;font-size:17px}.abstract p{font-size:15px;margin:0}.keywords{font-size:12px;color:#637280;margin-top:15px}section{margin-top:46px}h2{font-size:25px;line-height:1.4;margin:0 0 20px;border-bottom:1px solid #dde2e7;padding-bottom:12px}h2 .num{font-family:Georgia,serif;color:#7890a3;margin-right:12px}h3{font-size:18px;margin:26px 0 10px}p{margin:12px 0}strong{font-weight:700}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:24px 0}.card{border:1px solid #dce2e7;padding:20px}.card .label{font-size:11px;color:#6e8596;letter-spacing:1px}.card h3{margin:10px 0;font-size:17px}.card p{font-size:13px;line-height:1.7;margin:0}.note{background:#fcf7ef;border-left:3px solid #b9702d;padding:14px 20px;margin:20px 0;font-size:14px}.finding{border-left:3px solid #2c7767;padding:12px 20px;background:#f3f8f6;margin:20px 0}.table-wrap{overflow:auto;margin:22px 0}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums;line-height:1.65}caption{text-align:left;color:#586976;font-size:12px;margin-bottom:10px}thead{border-top:2px solid #324c61;border-bottom:1px solid #667e90;background:#f5f7f9}th,td{padding:11px 10px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}tbody tr{border-bottom:1px solid #e4e8ec}tbody tr:last-child{border-bottom:2px solid #324c61}tbody th{font-weight:500}tbody tr:hover{background:#f5f9fc}figure{margin:25px 0}.chart{width:100%;height:auto;display:block;background:#fafbfc;border:1px solid #e3e7eb}.chart text{font:12px -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;fill:#526574}figcaption{font-size:12px;line-height:1.7;color:#63717f;margin-top:10px}.equation{font:17px/1.85 Georgia,"Songti SC",serif;overflow:auto;background:#fafbfc;border:1px solid #e5e9ed;padding:17px 22px;margin:17px 0;text-align:center}.equation small{font-size:12px;color:#75818b}.protocol{display:grid;grid-template-columns:1fr 1fr;gap:16px}.protocol>div{padding:18px;border:1px solid #dce2e7}.protocol h3{margin-top:0}details{border:1px solid #dbe2e8;padding:0 18px;margin:14px 0}summary{cursor:pointer;padding:14px 0;font-size:14px;color:#244f70;font-weight:600}details .table-wrap{margin-top:0}code{font:12px/1.7 Menlo,monospace;overflow-wrap:anywhere}li{margin:7px 0}.footer{border-top:1px solid #dbe2e8;padding-top:20px;margin-top:45px;color:#78818b;font-size:12px}button{background:#fff;border:1px solid #c8d4de;color:#244f70;padding:8px 14px;cursor:pointer;margin:8px 5px 0 0}button:hover{background:#edf3f7}.pill{font-size:11px;color:#2c7767;border:1px solid #bad5cc;padding:3px 9px;display:inline-block}.refs li{font-size:13px}.muted{color:#6d7780;font-size:13px}@media(min-width:1500px){.paper{margin-left:calc(220px + (100vw - 1340px)/2)}}@media(max-width:1050px){.sidebar{width:180px;padding:25px 18px}.paper{margin-left:200px;padding:45px 35px}.grid{grid-template-columns:1fr}}@media(max-width:750px){.sidebar{position:static;width:auto;height:auto;padding:18px 22px}.sidebar strong{font-size:18px;margin:6px 0}.sidebar nav{display:flex;flex-wrap:wrap;gap:4px 18px}.sidebar a{padding:4px 0;border:0}.sidebar .small{display:none}.paper{margin:0;padding:30px 20px;box-shadow:none}h1{font-size:27px}h2{font-size:22px}.protocol{grid-template-columns:1fr}.abstract{padding:20px}.grid{gap:10px}}@media print{@page{size:A4;margin:17mm}.sidebar,.actions{display:none}body{background:white;font-size:10.5pt}.paper{margin:0;padding:0;max-width:none;box-shadow:none}h1{font-size:25pt}h2{font-size:16pt;break-after:avoid}h3{break-after:avoid}section{margin-top:24px}table{font-size:8pt}th,td{padding:5px 4px}thead{display:table-header-group}tr{break-inside:avoid}.table-wrap{overflow:visible}figure,.equation,.card{break-inside:avoid}details{border:0;padding:0}summary{font-size:11pt}.grid{grid-template-columns:repeat(3,1fr)}a{color:inherit}p{orphans:3;widows:3}}'''
body=f'''<aside class="sidebar"><div class="brand">AGENTIC TIME SERIES</div><strong>方向标签辅助预测</strong><nav><a href="#abstract">摘要</a><a href="#setup">01 · 问题与设置</a><a href="#experiment1">02 · 方向标签预测</a><a href="#experiment2">03 · 下游预测效用</a><a href="#experiment3">04 · 标签错误敏感性</a><a href="#discussion">05 · 讨论与局限</a><a href="#appendix">附录 · 完整结果</a><a href="#reproduce">复现与来源</a></nav><div class="small">研究报告 · 2026.10.23<br>实验完成 · 2026.10.19<br>ETTh1 / 512 → 64<br>离线单文件 · 无外部资源</div></aside>
<main class="paper"><header><div class="eyebrow">Empirical Research Report · Direction-Conditioned Forecasting</div><div class="rule"></div><h1>从方向判断到数值预测：<br>LLM 标签的效用与错误敏感性<span>From Direction Labels to Forecasts:<br>An Empirical Study of Utility and Error Sensitivity</span></h1><div class="meta">ETTh1 · 五种预测模型 · 两种方向定义 × 两种接入方式<br>报告日期：2026 年 10 月 23 日　｜　实验完成：2026 年 10 月 19 日</div><span class="pill">全部实验完成 · 指标独立复核</span><div class="actions"><button onclick="printReport()">打印 / 导出 PDF</button><button onclick="document.querySelectorAll('details').forEach(d=>d.open=true)">展开全部附录</button></div></header>
<div class="abstract" id="abstract"><h2>摘要</h2><p>我们研究一个具体问题：仅由历史时间序列预测的离散方向信号，能否提高多变量时间序列的数值预测精度？围绕 ETTh1，我们开展三个互补实验。首先，使用启用推理的 DeepSeek-v4-pro 预测每个变量的三分类方向标签，完成原训练、验证和测试集的标签覆盖。其次，将预测标签接入五种时序模型，比较直接替换 Oracle 条件、使用预测条件重新训练、真实条件 Oracle 和无条件基线。最后，在固定 Oracle 模型上人为控制标签正确率，评估模型对随机错误的敏感性。结果表明：测试集 L1/L2 Macro-F1 分别为 <strong>0.3517/0.4758</strong>；直接替换在全部 20 个配置上均劣于原始基线，重训虽在 18 个配置上缓解退化，但未产生稳定的额外标签收益。随机误标实验中，四种方案的平均 MSE 均随正确率提高而下降，90% 档位优于基线、80% 档位劣于基线。本报告将这些结论限定为单数据集、单训练种子和单噪声种子的实证观察，不将随机误标结果解释为 LLM 的部署准确率门槛。</p><div class="keywords">关键词：时间序列预测 · 大语言模型 · 条件预测 · Oracle 分析 · 标签噪声 · 负结果</div></div>
<section id="setup"><h2><span class="num">1</span>研究问题与统一实验设置</h2><p>本报告的三项工作是同一研究框架下的<strong>三个实验</strong>，而非三个独立预测模型：实验一测量方向信号的可预测性，实验二测量其下游效用，实验三测量固定预测模型对标签错误的敏感性。先前已完成的 Oracle 与原始基线作为共同参照，本轮不重复训练这些参照。</p><div class="grid"><div class="card"><div class="label">RQ1 · PREDICTABILITY</div><h3>方向标签能否预测？</h3><p>历史 512 步 → LLM → 每变量 −1 / 0 / +1<br>评价：Macro-F1、Accuracy</p></div><div class="card"><div class="label">RQ2 · DOWNSTREAM UTILITY</div><h3>预测标签是否有用？</h3><p>历史 + 预测条件 → 五种时序模型<br>评价：数值 MSE、MAE</p></div><div class="card"><div class="label">RQ3 · ERROR SENSITIVITY</div><h3>错误如何影响收益？</h3><p>真实标签 + 控制噪声 → 冻结模型<br>评价：正确率—MSE 曲线</p></div></div>
<h3>1.1 数据、任务与评价单位</h3><p>所有实验使用 ETTh1 的原始时间切分，7 变量输入、7 变量输出；输入历史长度 T = 512，预测长度 H = 64。原训练、验证和测试集分别包含 8,065、2,817 和 2,817 个有效窗口。原始目标起点分别为 512–8576、8640–11456 和 11520–14336（从 0 计数，含端点）。数值模型沿用仅由原训练集拟合的 StandardScaler。</p><p>五个配置为 DLinear、PatchTST、iTransformer、FreTS 和 iTransformer+FreDF；其中 FreDF 是沿用的 iTransformer 频域损失配置，而非另一个基础架构。主表先对各模型的所有测试窗口、预测步及 7 个变量计算 MSE，再对五种模型等权平均。OT 另行报告；全变量提升不等于 OT 一定提升。</p>
<div class="equation">MSE<sub>m</sub> = (NHC)<sup>−1</sup> ∑<sub>n=1…N</sub> ∑<sub>h=1…H</sub> ∑<sub>c=1…C</sub> (ŷ<sub>n,h,c</sub> − y<sub>n,h,c</sub>)<sup>2</sup><br>MSE<sub>avg</sub> = ⅕ ∑<sub>m=1…5</sub> MSE<sub>m</sub><br><small>所有数值指标均在原标准化空间计算；MSE / MAE 越小越好。主表不是相对改善率的平均。</small></div>
<h3>1.2 两种模型无关的方向标签</h3><p>令 t 为预测区间第一步，μ<sup>F</sup><sub>t,c</sub> 为未来 64 步均值，μ<sup>R</sup><sub>t,c</sub> 为最近历史 64 步均值，μ<sup>X</sup><sub>t,c</sub> 和 σ<sup>X</sup><sub>t,c</sub> 为历史 512 步均值和总体标准差（ddof = 0）。两种标签均不依赖任何基础模型的预测。</p><div class="equation">r<sup>L1</sup><sub>t,c</sub> = (μ<sup>F</sup><sub>t,c</sub> − μ<sup>R</sup><sub>t,c</sub>) / max(σ<sup>X</sup><sub>t,c</sub>, ε)<br>r<sup>L2</sup><sub>t,c</sub> = (μ<sup>F</sup><sub>t,c</sub> − μ<sup>X</sup><sub>t,c</sub>) / max(σ<sup>X</sup><sub>t,c</sub>, ε)<br>z(r) = +1，r &gt; τ；　z(r) = −1，r &lt; −τ；　其余 z(r) = 0<br><small>本轮固定 τ = 0.5、ε = 0.001。0 表示均值变化处于阈值内，不表示未来曲线没有波动。</small></div>
<div class="note">真实标签需要未来数据，因此只能用于离线评分或明确标注的 Oracle 实验。LLM 推理输入不含未来值；方向标签的 Accuracy 也不是未来每一步数值变化方向的准确率。</div></section>
<section id="experiment1"><h2><span class="num">2</span>实验一：历史条件下的方向标签预测</h2><h3>2.1 设计与执行</h3><p>通过服务器 DAO API 调用 <code>aliyun/DeepSeek-v4-pro</code>，启用推理，使用冻结的 pro-concise 提示词。输入包含历史时间序列及历史派生信息，输出每个变量的 L1、L2 三分类判断；不臆造变量语义或因果关系，不用未来统计量辅助请求。前期完成了接口连通性、输出长度、格式和提示词的开发检查，最终版本在测试集上不再调整。</p><p>为降低调用量，采用因果组首复用：每个切分内，每 32 个连续窗口共用第一个窗口的预测标签；后续窗口只能复用先前已获得的判断，不跨切分。训练、验证、测试分别有 253、89、89 个代表窗口，共 431 个有效代表窗口，覆盖 13,699 个窗口及 95,893 条窗口×变量记录。评分覆盖全部窗口，而非只评估代表窗口。</p><p>训练期 API 输入的归一化只拟合截至当前代表窗口的已观测训练前缀；验证和测试使用完整原训练集统计。这个 API 输入变换与下游数值模型固定的原训练集 scaler 分开管理。真实未来仅用于离线标签评分。</p>{labeltable}<h3>2.2 结果与解释</h3><p>L2 的测试 Macro-F1 为 0.4758，高于 L1 的 0.3517，但两者均未达到最初希望的 0.5。L2 从验证集 0.6049 下降到测试集 0.4758，因此不能以验证结果替代泛化成绩。所有标签均有效，但输出完整性不等同于预测有效性。</p><p>测试集恒预测 0 的参照准确率为 L1 74.78%、L2 72.68%，高于 LLM 的 68.32% 和 62.01%；其 Macro-F1 仅为 0.2852 和 0.2806，低于 LLM。因而 LLM 获得了一定三类区分能力，但整体准确率与 Macro-F1 呈现不同结论，应同时报告。</p><div class="finding"><strong>发现 1：</strong>历史条件下的三分类预测流程已跑通并完整覆盖三段数据，但当前版本的测试标签质量仍有限。分类成绩本身不能证明下游数值预测收益。</div></section>
<section id="experiment2"><h2><span class="num">3</span>实验二：LLM 标签的下游预测效用</h2><h3>3.1 两种接入机制与 2 × 2 设计</h3><div class="protocol"><div><h3>A · 联合条件预测头</h3><p>在原模型输出相关特征处加入条件头。骨干与条件头共同训练，使用每个变量自己的标签。G2-A 从与旧 Oracle 对应配置相同的初始化开始，不从已训练的 Oracle 权重微调。</p></div><div><h3>B · 冻结骨干校正模块</h3><p>加载原无标签基线并冻结参数及 buffers，只训练共享的逐变量校正模块。模块使用原预测、最近 64 步历史及该变量的方向标签，输出长度 64 的校正量。</p></div></div><div class="equation">A：h<sub>c</sub> → [h<sub>c</sub>, Emb(z<sub>c</sub>)] → 条件残差 → 原模型输出变换<br>B：ŷ<sub>c</sub> = f<sub>θ₀</sub>(X)<sub>c</sub> + g<sub>φ</sub>([f<sub>θ₀</sub>(X)<sub>c</sub>, X<sub>recent,c</sub>, Emb(z<sub>c</sub>)])<br><small>A 为概念性接入表示：具体残差位置随骨干而异，保留其原有输出变换；B 的 θ₀ 完全冻结。</small></div><p>条件模块采用三类 embedding（维度 8）、宽度 32 的隐藏层、GELU 和 64 维输出，末层零初始化。A/B 分别使用 L1 或 L2，共四方案；不会将 L1 和 L2 同时输入一个模型，也不将其他变量的未来标签输入当前变量的条件头。</p>
{table(['组别','训练 / 验证条件','测试条件','本轮工作'],[['G1 直接替换','旧 Oracle 训练与选点','LLM 标签','20 次测试推理'],['G2 重训','LLM 标签；数值验证 MSE 选点','LLM 标签','20 个训练配置'],['G3 Oracle','真实未来离散标签','真实未来离散标签','引用旧 20 结果'],['G4 原始基线','无方向条件','无方向条件','引用旧 5 结果']],'四组比较。另引用 10 个 A/B-zero 无信息条件对照。')}
<p>G1 与 G3 使用完全相同的 checkpoint，仅改变测试条件。G2 的回归目标仍是未来真实数值序列，LLM 标签是条件输入而非训练目标；验证时也使用预测标签，以全变量数值 MSE 选择 checkpoint。因此 G2 相对 G1 同时改变了训练条件分布和验证选点条件。</p>{grouptable}<h3>3.2 结果与对照解释</h3><p>直接替换条件后，G1 的 20/20 个配置 MSE 均差于 G4。G2 在 18/20 个配置上优于 G1，但仅 6/20 略优于 G4，且相对匹配的 zero 对照只有 7/20 改善。四个方案的五模型平均 MSE 均未超过原始基线；B-L1 重训最接近基线。</p><p>必须区分架构/训练路径本身的影响与标签信息的影响。DLinear-A 的 G2-L1/L2 MSE 为 0.426923 / 0.426023，而 A-zero 已为 0.426912，原始基线为 0.353263。大部分差距在零标签条件下同样存在，不能将其全部归因于 LLM 标签错误。另一个例子是 iTransformer B-L1：G2 的全变量 MSE 略有改善，但 OT MSE 比原始基线恶化 18.42%。</p><div class="finding"><strong>发现 2：</strong>用预测标签重训通常比直接替换更稳妥，但本轮没有建立稳定的额外标签收益。G2 接近基线，不等价于 LLM 条件带来了有效增益。</div></section>
<section id="experiment3"><h2><span class="num">4</span>实验三：受控标签错误敏感性</h2><h3>4.1 固定权重、仅干预测试标签</h3><p>为研究标签错误与数值损失的关系，冻结原 Oracle checkpoint，仅在测试时将一定比例的真实标签替换为错误类别。正确率设为 60%、70%、80%、90% 和 100%；四个非完美档位新增 5 × 4 × 4 = 80 次推理，100% 复用原来的 20 份 Oracle 结果。不训练噪声适配模型，不调用 LLM。</p><div class="equation">z̃<sub>t,c</sub> = z<sub>t,c</sub>，若 (t,c) ∉ S<sub>q</sub><br>z̃<sub>t,c</sub> ∼ Uniform(&#123;−1,0,+1&#125; ∖ &#123;z<sub>t,c</sub>&#125;)，若 (t,c) ∈ S<sub>q</sub><br><small>q 为错误率，正确率为 1−q。被选中的标签一定错误，不会随机抽回原标签。</small></div><p>采用固定噪声种子 20261019，逐变量无放回抽取窗口。10%、20%、30%、40% 的错误位置为嵌套集合，同一位置使用固定错误类别；所有模型、A/B 和 L1/L2 共用位置与类别偏移，以减少比较中的随机差异。每个变量有 2,817 个测试标签，实际错误数分别为 282、563、845、1,127，正确率因整数取整略偏离标称值。</p><figure>{chart}<figcaption>图 1｜四方案的正确率—平均 MSE 曲线。虚线为五模型原始基线均值。无误差带：本轮只有一个噪声种子，不提供跨种子不确定性估计。</figcaption></figure>{noisetable}<h3>4.2 结果与适用范围</h3><p>四条平均曲线均随正确率增加而下降。在已测试的离散档位中，90% 正确率的四种方案均优于基线 0.368701，80% 均差于基线。该观察不能给出精确临界正确率，也不能说明所有单模型或变量具有相同阈值。</p><div class="note"><strong>关键区别：</strong>随机误标实验针对每个窗口分别加入错误，不采用 LLM 的每 32 窗口标签共享。随机对称错误与 LLM 的结构性错误在类别混淆、变量分布、时间相关性和窗口难度上可能不同。因此，不能把这条曲线直接解释为“LLM 达到 90% 准确率就一定有效”。</div><div class="finding"><strong>发现 3：</strong>固定 Oracle 模型对标签错误敏感。该实验给出受控随机错误下的性能响应，而不是对真实 LLM 错误机制的因果归因。</div></section>
<section id="discussion"><h2><span class="num">5</span>综合讨论与局限性</h2><p>三个实验共同区分了<strong>信号是否可预测、信号是否有用、系统能否容忍信号错误</strong>。Oracle 结果支持方向信息具有改善平均预测的潜力；当前 LLM 标签尚未将这种潜力转化为稳定收益。重训缓解直接替换的退化，与训练/测试条件分布适配的解释相容，但该解释并未通过独立消融得到确认。</p><ol><li><strong>有限实验范围。</strong>只有 ETTh1、一个训练种子协议和一个噪声种子；窗口高度重叠，不能将窗口数量当作独立实验重复数。未报告显著性检验或置信区间。</li><li><strong>测试使用边界。</strong>验证集参与了提示词开发；下游实验前已经观察到测试标签 F1，但本轮冻结模型配方、提示词、阈值和数据切分，没有据此调参。随机误标实验属于已观测测试集上的诊断。</li><li><strong>Oracle 不是部署结果。</strong>G3 和实验三均使用真实未来构造标签；Oracle 也不保证每个有限训练配置、每个变量都构成严格数学上界。</li><li><strong>准确率不足以描述错误。</strong>类别比例、±1互相错判与错判为0、误差持续时间、窗口难度和变量权重都可能影响数值损失。本轮尚未分离这些因素。</li><li><strong>复现协议保留。</strong>历史骨干/CPU初始化为2021，条件模块为2022，实际CUDA随机状态为2022；已披露历史包装器的RNG行为，没有在本轮静默修复以改变配对条件。</li></ol><p>后续值得验证的问题包括：按类别与时间结构分解 LLM 错误、匹配 stride32 的噪声对照、历史分类器参照及多种子重复。这些是研究建议，<strong>并非已经执行的实验</strong>。</p></section>
<section id="appendix"><h2><span class="num">A</span>附录：完整指标与实现细节</h2><p class="muted">点击展开；打印按钮会自动展开全部附录。所有表格数据均内嵌，可断网阅读。</p><details><summary>A.1 · 方向标签的逐变量与类别诊断</summary>{varappend}{classtable}</details>{''.join(details)}{''.join(noiseappend)}
<details><summary>A.4 · 模型接入位置、损失与训练配置</summary><p>A 的接入位置分别为：DLinear 的季节/趋势历史特征拼接；PatchTST 的展平 patch 特征预测头；FreTS 的末层预测特征；iTransformer/FreDF 的变量 token 投影特征。iTransformer 的时间戳 token 不添加变量标签残差。B 则在标准化数值输出上进行逐变量校正。骨干原有时间戳输入和输出变换保留。</p>{table(['方法','Batch','学习率','最大 epoch','Patience','损失'],[['DLinear',32,'0.005',10,3,'MSE'],['PatchTST',128,'0.0001',100,100,'MSE'],['iTransformer',32,'0.0001',10,3,'MSE'],['FreTS',16,'0.0001',10,3,'MSE'],['iTransformer+FreDF',32,'0.0001',10,3,'0.2 MSE + 0.8 复数时间轴 rFFT 绝对误差']])}<p>沿用原 Adam、学习率调度、drop_last、worker 和独立 split generator。G2-A/B 配置及共同 epoch 采样顺序与旧 Oracle 对应实验一致；实际早停和选出的 epoch 可因验证表现而不同。三类 embedding 维度8，隐藏宽度32，输出64；原末层零初始化。</p></details></section>
<section id="reproduce"><h2><span class="num">B</span>复现、核验与证据来源</h2>{table(['实验','运行标识','完成规模'],[[a,'<code>'+b+'</code>',d] for a,b,d in manifest])}<p>实验一核验了431个有效代表窗口、窗口×变量唯一性、预测标签NPZ和评分记录一致性、因果复用及数据哈希。实验二重算75份保存预测，验证20组初始化、506对共同epoch采样顺序、10个B骨干全部参数与buffers一致。实验三独立重算100份结果，验证嵌套噪声、实际错误数、输入标签、权重和目标窗口一致；100%均值与旧Oracle完全相同。</p><p>实验二下游代码版本 <code>20fdc14</code>；实验三最终版本 <code>5fcbe20</code>。实验三v1曾在准备阶段因标签函数导入错误停止，未执行GPU推理；修复后经授权重启，保留失败记录并继承原截止时间。最终80次评估全部完成。</p><p>数据SHA256：<code>3dbe45a7a25cbf0eb6d82cb84ef4487ea2b73a9341e5ff0d082e340b9d943f1d</code>。完整权重与预测数组留在服务器，轻量指标已同步。本文仅使用既有实验记录，不新增训练或API请求。</p><ol class="refs"><li>实验一：<code>reports/direction_labels_allsplits_20261019.md</code> 与对应 <code>metrics.json</code>。</li><li>实验二：<code>reports/etth1_llm_downstream_20261019.md</code> 与对应 <code>comparison.json</code>。</li><li>实验三：<code>reports/etth1_oracle_noise_20261019.md</code> 与对应 <code>comparison.json</code>。</li></ol><p class="muted">采用论文式组织形式呈现内部实证结果；不是已发表或已同行评审论文。不添加未核验的文献结论、实验或作者信息。</p></section><footer class="footer">研究记录 · ETTh1 Direction-Conditioned Forecasting<br>单文件 HTML｜样式、图形、表格均内嵌｜无外部脚本、字体或网络请求</footer></main>'''
document='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="ETTh1方向标签辅助预测：LLM分类、下游效用及标签噪声敏感性三个实验的完整研究报告。"><title>方向标签辅助预测：三个实验的实证研究报告</title><style>'+CSS+'</style></head><body>'+body+'<script>function printReport(){document.querySelectorAll("details").forEach(d=>d.open=true);window.print();}</script></body></html>'
# Read one frozen request/response only. This never calls the model API.
remote_code = '''import json,sys
from pathlib import Path
p=Path("/nfs/dataset-ofs-661-142656/qiminhao/Foundation_model/agentic_TS/TSRAG/runs/20261019_dao_etth1_allsplits_v1")
sys.path.insert(0,str(p/"code"))
from explore_server import pack
w=json.loads((p/"inputs/ETTh1-test-11520.json").read_text())
r=json.loads((p/"responses/ETTh1-test-11520.json").read_text())
m=json.loads((p/"manifest.json").read_text())
print(json.dumps(dict(payload=pack(w),input=w,prompt=m["prompt"],content=r["raw_response"]["choices"][0]["message"]["content"],prediction=r["prediction"])))
'''
import shlex
example=json.loads(subprocess.check_output(['ssh','-o','ClearAllForwardings=yes','-o','BatchMode=yes','-o','ConnectTimeout=10','GPU_Foundation','python3 -c '+shlex.quote(remote_code)],text=True))
manifest_local=load('runs/20261019_dao_etth1_allsplits_v1/manifest.json')
assert example['prompt']==manifest_local['prompt']
input_digest=hashlib.sha256(json.dumps(example['input'],sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
anchor=next(a for a in manifest_local['anchors'] if a['id']=='ETTh1-test-11520')
assert input_digest==anchor['input_hash']
# Keep the recorded hash visible; compare predictions against independently scored rows.
scored=[]
with (ROOT/'runs/20261019_dao_etth1_allsplits_v1/test_window_variable_results.jsonl').open() as stream:
    for line in stream:
        row=json.loads(line)
        if row['target_start']!=11520:break
        scored.append(row)
pred={r['variable']:r for r in example['prediction']['labels']}
assert len(scored)==7
assert all(pred[r['variable']][l]==r[l+'_pred'] for r in scored for l in ['L1','L2'])
def codeblock(value):
    return '<pre class="prompt-code">'+html.escape(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2))+'</pre>'
sampletable=table(['变量','L1 预测','L1 真实','L2 预测','L2 真实'],[[r['variable'],r['L1_pred'],r['L1_true'],r['L2_pred'],r['L2_true']] for r in scored],'真实测试窗口 ETTh1-test-11520；真实标签仅来自离线评分文件。')
bounds=example['payload']['variables']['OT']
boundstable=table(['OT 历史派生字段','数值'],[[k,f(bounds[k])] for k in ['mean','recent64_mean','population_std','L1_low','L1_high','L2_low','L2_high']])
counts={l:sum(r[l+'_correct'] for r in scored) for l in ['L1','L2']}
request={'messages':[{'role':'system','content':example['prompt']},{'role':'user','content':json.dumps(example['payload'],separators=(',',':'))}]}
prompt_section='''<section id="prompt-example"><h2><span class="num">2A</span>实际提示词与端到端实例</h2>
<p>以下 System Prompt 来自最终 all-splits 运行的冻结 manifest，并与本地记录逐字核对；不是为报告改写的示意提示词。实例固定选取测试集第一个代表窗口，而非按预测效果筛选。读取旧记录不会新增 API 调用。</p>
<h3>提示词设计：让模型判断同一个未来均值</h3>
<p>模型接收七个变量各 512 个历史值（保留四位小数），以及历史均值、最近 16/64/128 步均值、历史总体标准差、32 个连续 16 步块均值和四个分类边界。完整历史始终保留，统计摘要只是补充。边界完全由历史计算，不包含真实未来均值。模型需要用同一个未来均值判断 L1 与 L2，避免两种标签在数值上相互矛盾；0 表示落入中间区间，不表示“不确定”。</p>
<h3>System Prompt · 实际英文原文</h3>'''+codeblock(example['prompt'])+'''
<h3>User Prompt · 历史证据包</h3><p>User 消息是紧凑 JSON，而不是额外一段自然语言。顶层包含窗口 ID、512/64 长度、起止时间、小时频率、单位说明和变量字典。每个变量字典保留完整 history 数组和历史统计。下面可展开全部七个变量的实际输入，未省略历史值；为阅读重新缩进，内容与实际请求一致。</p>
<details><summary>展开完整 User JSON（7 × 512 个历史数值及统计量）</summary>'''+codeblock(example['payload'])+'''</details>
<h3>实例：测试集第一个代表窗口</h3><p><code>ETTh1-test-11520</code>：历史索引 11008–11519，预测索引 11520–11583，均从 0 计数、含端点。历史时间为 2017-10-02 16:00 至 2017-10-23 23:00；未来 64 小时从 2017-10-24 00:00 开始。源数据未声明时区，不擅自赋予业务语义或时区。</p>
<p>以 OT 为例，实际传入的历史均值和分类边界如下，全部处于原训练集标准化后的数值空间：</p>'''+boundstable+'''
<p>模型对 OT 返回 L1=0、L2=+1，含义是：它判断未来均值相对最近 64 步仍处于中间档，但相对历史 512 步明显偏高。这两个标签可以同时成立，不能将它们理解为互相矛盾的“趋势不变”和“趋势上升”。请求不要求输出精确未来均值，因此不能从类别反推出模型实际预测的均值。</p>
<h3>模型最终输出 · 原始 content</h3>'''+codeblock(example['content'])+'''
<p>此处展示 API 最终答案字段，不展示内部推理文本，也不编造逐步推理过程。实际运行启用了 reasoning；接口认证信息不包含在报告中。</p>'''+sampletable+f'''
<p>这个窗口的 L1 正确 {counts['L1']}/7、L2 正确 {counts['L2']}/7。它只是一个完整流程实例，不代表总体准确率；全测试集指标仍以表 1 为准。该代表窗口的预测随后供目标起点 11520–11551 共 32 个窗口使用，但每个窗口评分都与自己的未来真实标签比较，而不是与代表窗口的真实标签比较。</p>
<details><summary>请求结构与可追溯信息</summary><p>模型：<code>aliyun/DeepSeek-v4-pro</code>；reasoning 开启；响应上限 16,384 tokens。下方只保存 messages，不含地址、认证头或密钥；User content 保持紧凑 JSON 字符串。历史输入记录 SHA256：<code>{anchor['input_hash']}</code>。</p>'''+codeblock(request)+'''</details>
<div class="note">数据边界：输入只有历史及其派生统计；真实未来标签仅在上面的离线对照表出现。JSON 合法、类别属于 −1/0/+1、L1/L2 区间一致只是有效性检查，不代表预测正确。</div></section>'''
document=document.replace('<section id="experiment2">',prompt_section+'<section id="experiment2">')
document=document.replace('<a href="#experiment2">','<a href="#prompt-example">提示词与真实实例</a><a href="#experiment2">',1)
document=document.replace('</style>','.prompt-code{white-space:pre-wrap;overflow-wrap:anywhere;max-height:540px;overflow:auto;padding:18px;background:#f5f7fa;border:1px solid #dce2e7;font:12px/1.7 Menlo,monospace;text-align:left}@media print{.prompt-code{max-height:none;overflow:visible;font-size:7pt}}</style>',1)
output=ROOT/'reports/direction_conditioning_three_experiments.html'
# Public Git artifact: no historical arrays or sample statistics. Private copy
# can be generated explicitly, but must stay under ignored runs/.
import sys
if '--private-example' in sys.argv:
    output=ROOT/'runs/20261019_dao_etth1_allsplits_v1/report_with_private_example.html'
else:
    import re
    document=re.sub(r'<details><summary>展开完整 User JSON.*?</details>', '<p class="note">Git 分享版不携带历史数值。完整输入由有权限的使用者在本地读取数据后构造，字段结构见下方；此前完整实例保留在本地、不提交。</p>', document, flags=re.S)
    document=document.replace(boundstable,'<p><code>mean, recent64_mean, population_std, L1_low, L1_high, L2_low, L2_high</code> 均由历史生成；分享版不附具体历史统计值。</p>')
    document=document.replace(codeblock(request),codeblock({'messages':[{'role':'system','content':example['prompt']},{'role':'user','content':'JSON: window_id, context=512, horizon=64, time_start, time_end, interval, units, time_zone, variables.{ID}.{history[512],mean,recent64_mean,population_std,L1_low,L1_high,L2_low,L2_high,recent16_mean,recent128_mean,block16_means[32]}'}]}))
    document=document.replace('下面可展开全部七个变量的实际输入，未省略历史值；为阅读重新缩进，内容与实际请求一致。','实际请求包含完整历史；本 Git 分享版仅展示输入字段，不包含原始或归一化后的时间序列。')
    document=document.replace('User content 保持紧凑 JSON 字符串。','User content 在此仅展示结构，非可直接发送的完整请求。')
output.write_text(document)
assert 'src="http' not in document and '<link ' not in document
assert document.count('<table>')==document.count('</table>')
print(output)
print('Bytes:',output.stat().st_size,'tables:',document.count('<table>'),'SHA256:',hashlib.sha256(output.read_bytes()).hexdigest())
