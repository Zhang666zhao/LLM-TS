"""Render saved metrics and all judgement rows; never calls an API."""
import argparse,csv,json,statistics
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run=args.run
    m=json.loads((run/'metrics.json').read_text());manifest=json.loads((run/'manifest.json').read_text())
    b=json.loads((run/'budget.json').read_text());s=json.loads((run/'status.json').read_text())
    rows=list(csv.DictReader((run/'window_variable_results.csv').open()))
    assert len(rows)==19719
    assert len({(r['window_id'],r['variable']) for r in rows})==19719
    assert len({r['window_id'] for r in rows})==2817
    groups={}
    for r in rows:
        t=int(r['target_start']);a=int(r['anchor_target_start']);assert a<=t<a+32
        assert t-a==int(r['label_age_steps'])
        for lab in ('L1','L2'):
            key=(a,r['variable'],lab)
            if key in groups:assert groups[key]==r[lab+'_pred']
            groups[key]=r[lab+'_pred']
    lines=['# ETTh1完整验证集：每32窗口复用方向标签','',
           '日期：2026-10-19。服务器GPU_Foundation运行，模型aliyun/DeepSeek-v4-pro，推理开启，沿用成功的pro-concise提示词。','',
           f"覆盖：{m['valid_anchors']}/89个代表窗口，扩展到2817个预测窗口；保留19719条“窗口×变量”记录，每条均含L1/L2真值、预测、对错、来源窗口、复用步数和状态。",'',
           '## 全部窗口的主要指标','',
           '|定义|Macro-F1|准确率|缺失标签|恒0 Macro-F1|','|---|---:|---:|---:|---:|']
    for lab,r in m['metrics'].items():lines.append(f"|{lab}|{r['macro_f1']:.4f}|{r['accuracy']:.2%}|{r['invalid']}|{r['always_zero']['macro_f1']:.4f}|")
    lines+=['','每种标签19719个样本。固定类别[-1,0,+1]，Macro-F1按三类平均，缺失类别F1按0。若有无效响应，主指标将缺失预测计错，metrics.json另列valid_only。','',
            '## 逐变量结果','',
            '|变量|L1 F1|L1准确率|L2 F1|L2准确率|','|---|---:|---:|---:|---:|']
    for n in manifest['variables']:
        a=m['metrics']['L1']['per_variable'][n];c=m['metrics']['L2']['per_variable'][n]
        lines.append(f"|{n}|{a['macro_f1']:.4f}|{a['accuracy']:.2%}|{c['macro_f1']:.4f}|{c['accuracy']:.2%}|")
    lines+=['','## 代表窗口自身与块内稳定性','',
            '|定义|89代表窗口 F1|代表窗口准确率|块内真实标签保持率|','|---|---:|---:|---:|']
    for lab,r in m['metrics'].items():
        a=r['anchor_only'];lines.append(f"|{lab}|{a['macro_f1']:.4f}|{a['accuracy']:.2%}|{r['true_label_carry_agreement']:.2%}|")
    lines+=['','代表窗口指标按89×7=623个判断计算。块内真实标签保持率比较“当前窗口真实标签”和“组首窗口真实标签”，是离线的稳定性诊断；真实组首标签从未进入API。它不是本模型的准确率或理论上界。','',
            '## 分类别结果','',
            '|定义|类别|真实数量|类别F1|','|---|---:|---:|---:|']
    for lab,r in m['metrics'].items():
        for cls,c in r['per_class'].items():lines.append(f"|{lab}|{cls}|{c['support']}|{c['f1']:.4f}|")
    lines+=['','## 方案与记录','',
            '完整验证集目标起点8640..11456，窗口步长1；每组起点8640+32k，k=0..88。前88组各32窗口，最后1组仅1窗口。每组第一个窗口以其512步历史生成7变量L1/L2，标签仅向后复用；不能使用组中点或末点预测去填较早窗口。',
            '每个窗口用自己的未来64步真值、最近64/历史512基准和历史总体标准差生成真实标签；阈值±0.5、epsilon=.001、训练集拟合StandardScaler均与之前实验一致。L1/L2分别描述未来均值相对近期/整个历史的位置。',
            '仅代表窗口检查L1/L2联合一致性。复用到后续窗口后，参考均值变化可能使这组旧标签不再对应同一个未来均值，这是复用方案本身的近似误差，记录不会被改写或隐藏。','',
            f"运行状态：{s['status']}；从原轮次准备开始至评分约{m['elapsed_seconds']/60:.2f}分钟。",
            f"4个历史代表窗口经实际模型、完整prompt、输入与响应核验后复用。本轮总新增{manifest.get('prior_attempted',0)+s['attempted']}次请求（包含保留的失败及修复记录），成功窗口不重复调用。初始最大30在途；遇到429后先按3.5秒派发，再进一步限制到10秒派发，原截止时间保持不变。异常响应记录保留，使用同模型同prompt补跑。",
            f"收到的usage合计：{m.get('combined_new_request_usage',m['new_request_usage'])}。429未返回usage，费用及该请求计费未知。",'',
            f"初版stride32_v1在缓存模型默认值核验时停止，没有发送请求；后续运行保留各轮API记录，当前完成版本为{run.name}。模型、prompt、数据与标签定义保持一致，版本变化为缓存核验与调度恢复。",'',
            '19,719条完整判断保存在window_variable_results.csv和.jsonl；汇总、逐变量、类别支持数、混淆矩阵及有效子集指标在metrics.json。API响应和调用来源保留在服务器responses目录。','',
            '验证样本在历史和未来区间均大量重叠，不能将所有记录当作独立统计样本。结果只代表ETTh1完整验证集上的stride32近似方案，不是测试集结论，也没有接入A/B训练。',
            '先前30窗口结果与这里的样本分布、标签复用方式不同，不能把分数差异直接归因于某一种改进。','',
            '服务器目录：/nfs/dataset-ofs-661-142656/qiminhao/Foundation_model/agentic_TS/TSRAG/runs/'+run.name+'。',
            '原始数据SHA256：3dbe45a7a25cbf0eb6d82cb84ef4487ea2b73a9341e5ff0d082e340b9d943f1d。']
    with args.output.open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(verified_rows=len(rows),unique_anchors=len({r['anchor_window_id'] for r in rows}),report=str(args.output))))


if __name__=='__main__':main()
