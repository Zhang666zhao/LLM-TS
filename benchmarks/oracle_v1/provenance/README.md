# 归档来源与缺口

## 2026-11-02 补齐轻量运行证据

已重新连通服务器并导出 [remote_archive_20261102](../remote_archive_20261102/README.md)：
五数据集175任务、Exchange35任务、原始四类骨干28评测，共238任务的可用
指标、配置、状态和训练记录，以及suite预算/配对检查。2,193文件逐个SHA校验；
175条原汇总和28条原始基线指标均与服务器一致。未重新计算预测数组。
下面2026-10-19描述保留为历史记录，不能再理解为当前完全没有远端结果归档。
仍未补齐骨干依赖源码/许可证映射、代码副本与历史提交逐字校验、Electricity
部分Oracle逐任务全清单；不宣称仅clone即可完整复现，也未创建冻结标签。

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
