# ETTh1：LLM方向标签四组比较执行计划（2026-10-19）

## 固定范围与预算

用户授权前两组运行，使用 GPU_Foundation 两张 A6000。仅 ETTh1，五方法 DLinear、PatchTST、iTransformer、FreTS、iTransformer+FreDF。每方法分别 A-L1、A-L2、B-L1、B-L2。固定20推理+20训练，不搜索、不自动重试；新建独立10小时安全上限，包含准备/检查/最终校验，不沿用或重置旧预算。单GPU单任务，需至少13GiB空闲，不干预其他进程。失败、OOM、非有限、数据/配对不一致全队停止；超时保留已完成和中断证据。

## 四组的可比定义

|组|训练条件|验证条件与checkpoint选择|测试条件|本轮工作|
|---|---|---|---|---|
|G1|历史oracle训练不变|沿用旧oracle验证所选checkpoint|LLM标签|20次测试推理|
|G2|LLM标签|LLM标签下全变量数值MSE最小|LLM标签|20次训练及测试|
|G3|真实未来离散标签|原oracle验证|真实未来离散标签|引用旧20结果|
|G4|无标签|原基线验证|无标签|引用旧5结果|

额外引用旧10个A/B-zero，无新增训练。65份核心独立结果（G4在各行复用）+10份辅助zero。
G1→G2差异包括条件分布匹配的训练和验证选点，不能说只改变了训练输入而无选点变化。
G1/G3必须使用同一checkpoint SHA；B各组引用同一基线。

## 输入、目标与缓存

全部7变量输入/输出，历史512预测64；数值目标仍为真实未来序列，不将LLM离散标签当回归目标。
按变量只接入自己的一个L1或L2，不同时拼两种标签，不输入其他变量未来信息。
train8065、val2817、test2817窗口，原始时间切分/stride1/目标/原训练集StandardScaler不变。
LLM缓存来自 runs/20261019_dao_etth1_allsplits_v1/labels，三份NPZ均已独立核验。
严格用NPZ的predicted l1/l2，禁止从CSV的truth列构造条件；每32窗口因果组首复用，模型历史仍每窗更新。
API训练证据的因果前缀scaler与数值模型原训练scaler是两套用途，不能互相替换。
不调用API，不调整prompt、tau0.5、epsilon0.001、标签定义或数据切分。

## A/B和配置

A：新初始化骨干+条件头联合训练，匹配旧oracle初始参数，不加载已训练oracle权重。
B：加载旧base_zero骨干并冻结参数和buffers，训练新初始化的共享逐变量校正模块。
通过校验SHA后调用原ad69667训练实现，优化器/损失/学习率/worker/批量/采样/早停完全沿用保存配置。

|方法|batch|lr|max epochs|patience|scheduler|workers|loss|
|---|---:|---:|---:|---:|---|---:|---|
|DLinear|32|.005|10|3|type1|4|MSE|
|PatchTST|128|.0001|100|100|type3|0|MSE|
|iTransformer|32|.0001|10|3|type1|0|MSE|
|FreTS|16|.0001|10|3|type1|0|MSE|
|iTransformer+FreDF|32|.0001|10|3|type1|0|.2 MSE+.8 complex time-rFFT absolute loss|

保留CPU/骨干2021、条件模块2022、实际CUDA2022协议，不静默修复历史RNG缺陷。
独立每split generator2021，train shuffle/drop_last；val/test全量不打乱。
自动记录完整配置并与旧对应oracle任务逐字段比较，20组初始化hash与共同epoch采样顺序一致。

## 检查与执行顺序

1. 核验原始数据SHA、431anchors缓存manifest、三split starts/target/shape/整数域/age；校验35历史checkpoint与配置。
2. generic preflight；每方法GPU合约验证A梯度、零初始化等价、标签通道隔离、B冻结，无优化器训练。
3. 两GPU队列执行20 G1与20 G2，不超出固定清单。每任务独立进程避免上游Python模块冲突。
4. 重算所有75份指标，65核心+10zero；完全相同test目标/窗口；G1权重与G3一致；G2预测标签等于缓存；20配对检查。
5. 按四方案分别汇报全变量MSE/MAE、7变量指标及OT，改善=(参考-候选)/参考。
   对比G1/G4、G2/G4、G2/G1、G1/G3、G2/G3及matched-zero，不把四方案合并掩盖差异。
6. 输出comparison.json、paired_checks.json、verification.json、summary.md和本地报告；失败不重跑。

限制：单seed、重叠窗口和stride32相关；验证集用于prompt开发；测试标签F1已被观测，本轮冻结配方不据此调参。
G3是未来信息oracle参考，非部署收益，亦不保证为每个有限训练模型的严格上界。

## 状态

准备中；实际run目录、deadline和提交号以launch记录为准。尚不能声称运行完成。
