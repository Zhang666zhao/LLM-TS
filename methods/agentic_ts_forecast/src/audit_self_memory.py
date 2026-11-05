"""Ground textual self-error lessons in observable fields and recorded training costs."""
from common import *
raw=json.loads((RUN/'cache/self_error_rules_parsed.json').read_text());request=json.loads((RUN/'cache/self_error_summary_request.json').read_text());cases=json.loads(request[-1]['content'])['cases'];bank=np.load(RUN/'cache/self_error_bank.npz');cost=bank['cost'];choices=np.array([0,.25,.5,.75]);kept=[];rejected=[];records=[]
# Explicit bindings of the observed draft, preserved verbatim alongside the audit.
bindings=[('recent_drift','negative','<',-.3,[2,3]),('recent_drift','positive','>',.3,[0]),('unit_correction_mean','negative','<',0,[2,3]),('unit_correction_mean','positive','>',0,[0]),('history_std','high','>',.8,[0])]
for i,rule in enumerate(raw['rules']):
 if i>=len(bindings):rejected.append({'draft_rule':rule,'reason':'Condition uses realized error or is outside observable feature whitelist; not available for current future'});continue
 field,word,op,threshold,options=bindings[i]
 if field not in rule['condition'] or word not in rule['condition']:raise RuntimeError('Draft changed: explicit binding requires re-audit')
 assert set(rule.get('evidence_origins',[]))<=set(bank['origins'].tolist())
 a=np.asarray([c[field] for c in cases]);mask=a<threshold if op=='<' else a>threshold;assert mask.any();base=float(cost[:,1][mask].mean());scores=[float(cost[:,j][mask].mean()) for j in options];j=options[int(np.argmin(scores))];alt=min(scores);item={'feature':field,'comparison':op,'threshold':threshold,'recommended_strength':float(choices[j]),'training_variable_cases':int(mask.sum()),'training_origins':int(mask.any(1).sum()),'default_training_mse':base,'recommended_training_mse':alt,'training_absolute_mse_gain':base-alt,'evidence_status':'training-supported hypothesis, not independently validated' if alt<base else 'contradicted by aggregate training counterfactual costs','uncertainty':rule['uncertainty'],'draft_rule':rule};records.append(item)
 if alt<base:kept.append(item)
 else:rejected.append(item)
result={'rules':kept,'limitations':['Only25 past training queries, correlated variables and cases; full-trained DLinear is in-sample','Thresholds are heuristic hypotheses; aggregate training support is not proof of transfer','Only history-observable fields permitted; current realized error is forbidden'],'rejected_rules_count':len(rejected),'source_rules_sha256':hashlib.sha256((RUN/'cache/self_error_rules_parsed.json').read_bytes()).hexdigest()};save_json(RUN/'cache/self_error_rules_audited.json',result);save_json(RUN/'cache/self_error_rules_audit_record.json',{'kept':kept,'rejected':rejected,'all_observable_bindings':records,'test_access':False,'code_revision':git_head()});print('RULES_AUDITED',len(kept),'kept',len(rejected),'rejected',flush=True)
