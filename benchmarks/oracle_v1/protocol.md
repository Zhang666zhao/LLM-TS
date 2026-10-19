# oracle_v1：历史口径，不可覆盖

五个数据集ETTh1/ETTh2/ETTm1/ETTm2/weather：175配置（109复用+66补跑）；Exchange：独立轮次35配置；Electricity未完成，排除完整比较。
五模型DLinear/PatchTST/iTransformer/FreTS/iTransformer+FreDF，512/64，全变量预测和监督。
L1=(未来64均值-最近64均值)/max(历史512标准差,0.001)；L2参考历史512均值。ddof=0，阈值严格超过±0.5才为±1。
A联合条件头；B冻结基线输出校正；各自zero对照。全变量验证MSE选检查点，FreDF保留历史混合损失。
CPU2021/CUDA2022，非宣称精确历史CUDA2021复现。单seed、真实未来oracle标签；不是部署收益。
历史175任务与50配对检查已于2026-10-15验收。本分支本次只有本地报告及其内嵌指标，不等于新完成远端验收或完整可复现归档。
