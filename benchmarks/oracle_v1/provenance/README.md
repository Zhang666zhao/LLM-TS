# 归档来源与缺口

2026-10-19从最新origin/main 112c8a9创建研究分支；主仓库原main工作目录保持不变。
本次SSH GPU_Foundation返回Permission denied (keyboard-interactive)，无法重新核查服务器。

本地来源：/Users/didi/Desktop/Write_Papers/agenticTS。
source_copies中的代码为本地副本，保持原样；它们含服务器布局依赖，未声称在新仓库可直接运行，也未重新验证与历史提交逐字一致。

历史服务器项目：/nfs/dataset-ofs-661-142656/qiminhao/Foundation_model/agentic_TS/TSRAG
- runs/20261014_exchange_multivariate_oracle_v1：35配置；模型6e48e84，授权续跑调度f665bd3。
- runs/20261014_six_multivariate_oracle_v1：112/210完成后10h超时，保留全部中断记录；模型ad69667。
- runs/20261015_five_multivariate_oracle_continue_v1：排除Electricity；109复用+66补齐，175完成；调度50a261f。
- Weather PatchTST B-zero因无完整优化器/RNG断点，在新目录从原seed重跑；旧记录未覆盖。

待补：远端代码哈希验证、每任务实际config/provenance/data_version、status/budget/paired_checks、Exchange35条完整指标、Electricity部分任务清单、依赖源码版本与许可证映射。
未创建oracle-v1-results-20261015标签，避免将不完整归档标记为冻结完成。
本地报告中的175条结果已另提取为results/five_datasets.json；该提取不是重新计算预测数组。
