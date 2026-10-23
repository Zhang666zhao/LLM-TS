# 未来方向标签辅助预测

## 已完成实验与数据获取（2026-10-23）

三切分 LLM 标签、四组下游比较和噪声敏感性实验均已完成。查看
[三项实验报告](../../reports/direction_conditioning_three_experiments.html) 和
[数据获取与复现限制](../../data/RESEARCH_DATA.md)。代码保留历史服务器路径；
完整复现还需要未纳入 Git 的原始 trainer 环境、标签缓存与 checkpoint。
下文保留早期接口原型用法；其历史状态描述不代表最新实验状态。

## 服务器并发调用（2026-10-19）

`concurrent_server.py`复用成功的Pro简洁推理prompt，推理保持开启。并发默认50、上限50，必须明确总请求数和派发时限。每个历史窗口一次请求，返回所有变量L1/L2；不是每变量发一次，避免重复输入开销。固定历史标签定义±0.5，其他阈值需适配新的compact版本。

输入目录只能放已审计的history-only窗口JSON。没有`--send`只检查并生成manifest，不联网。例（在服务器代码目录，路径替换为已批准的实验输入/输出）：

```bash
python concurrent_server.py --inputs /path/to/approved/history_windows \
  --config configs/dao_pro_concurrent.json --output /path/to/new_run \
  --concurrency 50 --max-requests 50 --wall-seconds 1800
```

添加`--send`才真正执行；凭证通过环境传入，不能写配置。上述50请求仅演示参数，不代表已授权新窗口集。截止时间控制新任务派发，网络超时按剩余时间缩短；已被网关接收的请求不能保证远端即时取消，可能在期限后结束/计费。首个错误（含429限流）停止新派发，最多50个在途请求继续记录，不自动重试或重复收费。不能承诺50倍加速，网关并发/TPM限额仍需真实运行确认。

19项离线测试通过，其中60个模拟任务达到峰值50并发且每个只执行一次。尚未做50路真实API压测。评价继续用独立真值文件，不由并发推理进程读取。

先读[完整实施计划](IMPLEMENTATION_PLAN.md)。标准库单次API原型不是批量训练入口；DAO服务器连通性及最终Pro批次已经验证，结果见上述报告。

从仓库根目录运行离线测试：

```bash
python3 -m unittest discover -s research/direction_conditioning/tests -v
```

历史JSON包含window_id、带时区的cutoff、递增timestamps、variables和history[T,C]。最后时间等于cutoff，默认至少64行，真实实验512行。必须由导出器核实尺度和时间索引。

先建立本目录private和outputs目录，将历史文件放private/window.json。默认不联网：

```bash
python3 research/direction_conditioning/predict.py \
  --config research/direction_conditioning/configs/dao.json \
  --input research/direction_conditioning/private/window.json \
  --output research/direction_conditioning/outputs/dry_run.json
```

单次调用获准后，环境中设置TOKEN_SERVICE_API_KEY，添加--send并选新的输出路径。--model aliyun/DeepSeek-v4-pro可切模型，默认tencent/DeepSeek-V4-Flash。同协议服务修改配置中的base_url/api_key_env/model即可。密钥不写配置。

include_stats/include_semantics控制可选增量，完整历史始终保留。输出含原始历史和响应，不提交Git。失败不自动重试；异常留下空文件时先诊断，不当作成功缓存。cost=null表示未知费用。

labels、modules、adapters、configs、scripts、tests为后续实现边界；原样代码副本保存在benchmarks/oracle_v1/source_copies，不重构历史实现。
