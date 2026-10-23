# ETTh1：LLM 方向标签接入五种预测模型的四组对照计划

日期：2026-10-19。最新状态：G1/G2于13:21:28 CST全部完成，40/40新任务，耗时1143.380秒，无重试/调参/新增API。75份结果独立重算通过，20组初始化、506对共同epoch采样顺序一致，10个B骨干权重/buffer完全冻结。379份轻量文件同步并校验SHA；监控已暂停，无追加训练。[完整四组结果](../../reports/etth1_llm_downstream_20261019.md)。G1的20个方案均差于基线；G2的18个方案优于G1，但相对matched-zero未体现稳定收益。运行目录 runs/20261019_etth1_llm_downstream_v1；代码20fdc14。以下早期未授权/缺失标签及运行中说明仅保留为历史记录。

后续完成（2026-10-19）：服务器 `runs/20261019_dao_etth1_allsplits_v1` 已完成 431 个有效 anchors，覆盖原训练/验证/测试全部 13,699 个窗口。标签位于该目录 `labels/{train,val,test}.npz`，本地已同步；[完整结果](../../reports/direction_labels_allsplits_20261019.md)。第 2 组的标签覆盖阻塞已解除；后续应保留原始切分，不采用第 10 节局部训练备选。训练期 API 输入 StandardScaler 只拟合截至 anchor 的历史训练前缀，验证/测试使用完整原训练集统计，已检查标签边界不变与 epsilon 不激活。本文下半部分的“缺少训练/测试标签”和“先做验证集”说明保留为最初计划记录，以本段完成状态及下一段执行口径为准。

完成标签后的执行口径：G2 使用完整原训练集的预测标签训练，使用原验证集预测标签下的数值 MSE 选择 checkpoint；G1/G2/G3/G4 最终统一在原测试窗口 11520..14336 比较。G3/G4 及旧 zero 的既有测试结果可原样复用并核验索引；G1 只需旧 oracle checkpoint + 测试 LLM 标签推理。核心新增训练仍为 20 个 G2 配置。没有因上游测试 F1 低于 0.5 改动模型、prompt 或阈值；新的训练资源和预算须在执行轮次中记录，本轮标签完成不自动启动下游训练。

## 1. 研究问题与当前可执行范围

比较四种条件：① oracle 训练的模型在推理时改用 LLM 标签；②用 LLM 标签重新训练并在推理时继续使用 LLM 标签；③已有 oracle 标签模型；④已有无标签原始模型。

仅使用 ETTh1，五种方法为 DLinear、PatchTST、iTransformer、FreTS、iTransformer+FreDF。每种条件模型保留 **L1/L2 × A/B** 四方案：A-L1、A-L2、B-L1、B-L2。这里的两个维度是标签定义和接入方式，不是两个数据集。

最初制定计划时，LLM 标签只覆盖原验证集，原训练/测试标签缺失；当前已经全部补齐。旧报告的主要 MSE/MAE 是测试集结果，因此后续应使用新缓存的原测试标签与旧结果对齐。验证与测试恰好各有 2,817 个窗口，样本数量相同不代表时间范围相同。

因此，保持“只用现有标签、零 API 调用”的要求时：

- 第 1、3、4 组可以立即准备同验证窗口的比较，全部使用已有 checkpoint，无须训练。
- 第 2 组的严格同协议重训方案在本文完整定义，但原训练集缺少 LLM 标签，当前不能执行。不能把 oracle 真标签冒充预测标签，也不能将验证标签按序号搬到训练集。
- 第 10 节给出仅用当前缓存进行训练的备选方案。它改变了训练区间，只能作为单独的探索性实验，不能默认为用户要求的严格重训，也不能用它声称第 2 与第 1 组只差训练标签。

## 2. 已核实的资产与时间范围

SSH：`GPU_Foundation`。服务器项目根目录：

`/nfs/dataset-ofs-661-142656/qiminhao/Foundation_model/agentic_TS/TSRAG`

以下服务器相对路径均以该目录为根：

|资产|位置与状态|
|---|---|
|冻结数据|`artifacts/baseline_raw_v1/ETTh1.csv`|
|数据 SHA256|`3dbe45a7a25cbf0eb6d82cb84ef4487ea2b73a9341e5ff0d082e340b9d943f1d`|
|完整 LLM 标签|`runs/20261019_dao_etth1_stride32_v5/window_variable_results.csv`，89/89 anchors 有效，19,719 条唯一窗口×变量记录|
|已同步标签 CSV 的 SHA256|`40db7952c8fb44bd91752279d95dc448d2046651d523c572dda8e9716de0d1c0`；执行时与服务器文件再次核对|
|LLM 标签来源|同目录 `manifest.json`、`data_version.json`、`responses/` 和各轮来源记录|
|35 个旧实验的统一入口|`runs/20261015_five_multivariate_oracle_continue_v1/ETTh1/jobs/`|
|被引用的旧结果|`runs/20261014_six_multivariate_oracle_v1/ETTh1/jobs/`，通过路径解析和 checkpoint 哈希去重，不能当成两次独立实验|
|原实现|`research/multivariate_oracle_six/{dataset,conditioning,train,alignment}.py`，历史模型版本 ad69667|

2026-10-19 只读检查：35 个 job 均 complete，均存在 `best.pth`、`best_validation.json` 和测试预测文件。历史训练程序只保存测试集逐窗口预测；验证集保存每轮及最佳轮的聚合/逐变量 MSE、MAE，未保存逐窗口预测。

下表 t 为 **0-based 原始行号，指预测区间的第一行**。历史输入为 `[t-512,t)`，预测目标为 `[t,t+64)`。

|原切分|预测起点 t（含两端）|窗口数|本次 LLM 标签|
|---|---:|---:|---|
|训练|512–8576|8,065|已补齐，allsplits_v1/labels/train.npz|
|验证|8640–11456|2,817|齐全|
|测试|11520–14336|2,817|已补齐，allsplits_v1/labels/test.npz|

固定变量顺序：`HUFL, HULL, MUFL, MULL, LUFL, LULL, OT`。全部模型为 7 变量输入、7 变量输出；全变量监督，OT 另列指标。DLinear/PatchTST 保持原通道独立结构，不退回先前 outcome-only 的第一阶段。

StandardScaler 只用原训练区间 `[0,8640)` 拟合，沿用历史 float32 数据和统计约定。原数据长度超出已冻结测试终点的部分不自动纳入。

## 3. 标签与监督目标

对窗口 t、变量 c，记真实未来数值序列为 Y，方向真标签为 z，LLM 预测标签为 z_hat。

\[
s_{t,c}=\max\{\operatorname{Std}_{ddof=0}(X_{t-512:t,c}),10^{-3}\},
\quad z^{L1}_{t,c}=Q\!\left(\frac{\overline Y_{t,c}-\overline X_{t-64:t,c}}{s_{t,c}}\right),
\quad z^{L2}_{t,c}=Q\!\left(\frac{\overline Y_{t,c}-\overline X_{t-512:t,c}}{s_{t,c}}\right).
\]

其中 Q(r) 在 r>0.5 时为 +1、r<−0.5 时为 −1，其余为 0。当前阈值沿用历史语义；后续可变，本轮不搜索阈值。

当前 z_hat 使用组首标签前向复用：

\[
a(t)=8640+32\left\lfloor\frac{t-8640}{32}\right\rfloor,\qquad
\widehat z^{Lk}_{t,c}=\widehat z^{Lk}_{a(t),c},\quad k\in\{1,2\}.
\]

只复用标签；数值模型对每个 t 仍读取它自己最新的 512 步历史，并预测它自己的未来 64 步。仍评估全部 2,817 个窗口，而非仅 89 个 anchors。

**第 2 组的“使用 LLM 标签训练”指用 z_hat 作为条件输入。数值监督目标始终是真实未来序列 Y。** 不将三分类标签当成数值回归目标，不增加标签分类损失，也不训练 DAO 大模型。

每个 A-L1 模型只收到该窗口各变量的 L1；A-L2 只收到 L2，B 同理。每个变量的输出条件模块只读取本变量标签，不把所有标签拼入 backbone，不同时输入 L1 和 L2。

API 已完成，执行下游工作时无需 API 密钥。训练/推理输入适配器仅允许读取 CSV 中的 `L1_pred/L2_pred` 及索引字段；`L1_true/L2_true/correct` 只用于独立分析，不参与 LLM 条件路径。

## 4. 四组实验矩阵

|组|已有/新模型|训练条件|评估条件|配置数|新增训练|
|---|---|---|---|---:|---:|
|G1：直接替换|原 oracle checkpoint|历史已用真实 z 训练|缓存的 z_hat|5×4=20|0|
|G2：预测标签重训|新 checkpoint|z_hat|z_hat|5×4=20|20，原训练集标签齐全才可执行|
|G3：oracle 参照|与 G1 完全相同的 checkpoint|历史已用真实 z 训练|当前窗口真实 z|5×4=20|0|
|G4：原始模型|已有 `method_base_zero/best.pth`|无方向条件|无方向条件|5|0|

G1 与 G3 必须复用同一 checkpoint 文件哈希，仅改变推理标签。G4 使用这轮 oracle suite 对应的五个 base_zero，且 B 路径必须使用其训练时所引用的同一 baseline checkpoint。更早 9 月历史结果可作背景，不混入主表，以免混淆已披露的 RNG 差异。

已有 A-zero/B-zero 共 10 个 checkpoint 可作为辅助参照，均不新增训练。这里的 zero 指**已经用恒 0 条件训练的模型**，不是给 oracle 模型临时喂 0 来冒充 matched-zero。

全部四组齐备时共有 65 行核心结果，另有 10 行旧 zero 参照。当前能准备的是 G1/G3/G4 的 45 行核心结果和 10 行 zero 参照。缺失的 G2 记为 `missing_train_llm_labels`，不能填 0 或拿其他组结果替代。

## 5. A/B 路径与五种模型适配

共享条件模块为 Embedding(3,8) + Linear(width+8,32) + GELU + Linear(32,64)，末层零初始化。标签整数 −1/0/+1 转成 embedding 索引 0/1/2。保持历史接入点和归一化/反归一化流程。

|模型|A：联合训练条件头|B：冻结骨干后训练校正块|
|---|---|---|
|DLinear|季节/趋势的 512+512 特征接条件残差头；原线性层与条件头共同训练|冻结原 DLinear；其 64 步预测与本变量最近 64 步历史拼接，再接条件校正块|
|PatchTST|原 patch 编码及 flatten 特征接条件残差头；编码器/原预测头/新条件头共同训练|冻结整个原 PatchTST，采用相同输出校正块|
|iTransformer|原变量 token 的 projector 输入接条件残差头；骨干共同训练；时间戳 token 不加标签残差|冻结整个原 iTransformer；保留历史时间戳输入，采用相同输出校正块|
|FreTS|原最后线性预测层之前的特征接条件残差头；频域骨干共同训练|冻结整个原 FreTS，采用相同输出校正块|
|iTransformer+FreDF|结构接入同 iTransformer，保留原 FreDF 训练损失|冻结该方法自身的 baseline，校正块保留同一 FreDF 损失|

对 B，输入特征宽度为 128，即 `[基础预测64, 最近历史64]`；校正块输出 64 步残差，并与基础预测相加。冻结包括参数和运行状态，backbone 保持 eval；训练后核验全部 backbone 张量哈希不变。

严格版 G2-A 从原配对随机初始化重新训练全部可训练参数，不从 G3 的已训练 oracle checkpoint 接着微调。严格版 G2-B 加载原 G4 baseline，重新初始化校正块并只训练它。这样分别对应历史 A/B 的真实训练流程。

## 6. 严格版 G2 的训练规范

数据范围、训练样本数、batch、优化器、学习率、scheduler、最大 epoch、patience、损失、初始化和 sampler 与同模型的历史配置一致。唯一预定变化是 train/val 的条件输入从 z 换成 z_hat。未来数值 Y 的监督保持不变。

2026-10-19 从服务器 `config.json` 核实的 ETTh1 参数：

|方法|batch|Adam 初始 lr|最大 epoch|patience|scheduler|workers|损失|
|---|---:|---:|---:|---:|---|---:|---|
|DLinear|32|0.005|10|3|type1|4|MSE|
|PatchTST|128|0.0001|100|100|type3|0|MSE|
|iTransformer|32|0.0001|10|3|type1|0|MSE|
|FreTS|16|0.0001|10|3|type1|0|MSE|
|iTransformer+FreDF|32|0.0001|10|3|type1|0|0.2 MSE + 0.8 复数时间轴 rFFT 差的平均绝对值|

A/B 都继承对应方法的历史配置。完整模型结构、dropout、weight decay 等直接加载原 config，不由这张摘要表重新猜默认值。CPU/骨干初始化 seed=2021；当前历史实现的实际 CUDA RNG=2022，条件模块初始化为 2022；本轮保留并披露，不静默修正 RNG。

训练 loader 沿用历史随机采样、drop_last、pin_memory 和独立 generator；评估不丢窗口。验证 checkpoint 选择依据为**输入预测标签后的全变量数值预测 MSE**，不能用 oracle 条件下的验证分数挑选 G2。FreDF 也按该 MSE 选 checkpoint，训练时才使用频域损失。

配对检查：G2 与对应旧实验的 backbone/condition 初始参数一致，B 的基础 checkpoint 一致，原训练窗口排序和共同 epoch 的采样顺序一致。因不同标签导致提前停止时间不同可以接受，并记录 epoch/优化步数；不能通过读测试结果改 epoch。

现有标签覆盖了严格版 G2 所需的原验证集，但缺原训练集 8,065 个窗口。若日后另行补齐，按同样 stride32 需要 253 个训练 anchors。若要独立原测试集报告，还缺 89 个测试 anchors。**这些仅是缺口核算，本计划不发送这些请求。** API 模型、prompt 和阈值也不能凭本计划自动更换。

## 7. 当前零 API 方案的执行顺序

1. 冻结 `label_manifest`：缓存路径/SHA256、模型/prompt 版本、变量顺序、时间范围、stride32 规则、每窗口标签完整性。冻结 35 个旧 checkpoint 的真实来源和哈希。
2. 建立独立下游输入适配器：以 `(dataset, target_start, variable)` 联结，输出 `[N,7]` 的 int64 L1/L2 标签矩阵；历史 `window_start.npy` 是输入起点，必须加 512 才能与 `target_start` 对齐。
3. 只读合同检查：所有窗口的输入/数值目标与历史 loader 一致；标签合法且无缺失；组首不晚于目标窗口，age 为 0–31；所有变量顺序严格相同。绝不按标签是否预测正确筛选样本。
4. G1：20 个旧 oracle checkpoint 在全部验证窗口使用缓存 z_hat 推理，记录全量预测和逐窗口误差。
5. G3/G4/zero：原 `best_validation.json` 已有同验证区间的聚合与逐变量 MSE/MAE，可直接读取作为参照；为了获取逐窗口配对误差，安排 35 次旧 checkpoint 验证集推理，并核对聚合值与旧记录在数值容差内一致。这里是推理，无训练。
6. 输出四方案分别比较的主表和差异分析。当前执行规模为 55 次 checkpoint 推理，0 次训练，0 次 API 调用。G2 保持待数据状态。
7. 只有严格版 G2 数据条件齐全，或者明确选择第 10 节的新探索切分，才创建对应训练任务。旧训练和 API 预算均不自动滚动沿用。

这轮主表属于 **原验证集开发分析**。历史 checkpoint 本来就由这份验证集选出，上游 prompt 开发/F1 分析也使用过其中样本；不能把结果称为独立测试性能。旧测试集指标保留在附表，不能用于计算当前验证集上的相对改善。

## 8. 指标与报告口径

下游主要指标为标准化数值预测的全变量 MSE，另报 MAE、7 个变量各自 MSE/MAE 和 OT 指标。对每个窗口的 64×7 个数值按历史方法统计；不先挑好变量，不改变权重。

每个模型均分别展示 A-L1、A-L2、B-L1、B-L2：

|方法/方案|G4 原模型|G3 oracle|G1 直接替换|G2 预测标签训练|matched-zero|
|---|---|---|---|---|---|
|例如 DLinear / A-L1|同窗口 MSE/MAE|同窗口 MSE/MAE|同窗口 MSE/MAE|未具备训练标签时为 NA|旧 A-zero 同窗口 MSE/MAE|

所有改善率均报告正负号，以 MSE 为例：

\[
\mathrm{Gain}(G_i;G_j)=100\%\times\frac{\mathrm{MSE}(G_j)-\mathrm{MSE}(G_i)}{\mathrm{MSE}(G_j)}.
\]

- G1 vs G4：直接使用当前 LLM 标签和旧校准模型，是否改善基础预测。
- G1 vs G3：同 checkpoint 下，替换标签带来的性能变化。差异同时包含 LLM 预测错误与 stride32 复用误差，不仅仅是模型推理能力。
- G2 vs G1：严格同协议时衡量“训练时适应预测标签”是否有帮助；若采用第 10 节备选，必须同时披露训练区间/样本量变化，不能单因果归因。
- G1/G2 vs 各自旧 matched-zero：检查条件方法相对零标签模块的表现。若使用新探索切分，旧 zero 未在该切分重新训练，因此只是旧参照，不能冒称与新 G2 完全配对的控制实验。
- G3 是经验 oracle 参照，不保证构成每个窗口或每个指标的数学上界。

上游 L1 Macro-F1=0.403547、accuracy=0.632639；L2 Macro-F1=0.604884、accuracy=0.635022。两者均保留；较高 F1 不自动意味着较低预测 MSE，不据此删掉 L1。

诊断按真实标签类别、是否判断正确、是否正负反向错误、标签 age 桶 0–7/8–15/16–23/24–31 分组；这些字段只在预测完成后分析。记录校正前后幅度和每变量误差，避免只看均值掩盖 OT 退化。

窗口高度重叠，不能把 19,719 条标签或每个预测步当成独立样本来声称统计显著。首轮报告配对误差差值、分组分布和单 seed 局限，不用任意独立样本检验制造显著性。

## 9. 实现边界与产物

建议新建独立目录 `research/direction_conditioning/downstream/`，保存标签适配、旧 checkpoint 装载、批量评估、G2 训练入口和汇总工具。引用历史骨干/条件模块，保留冻结原实现；不能让旧 Windows 默认读 oracle cache 而误用于 G1/G2。

本地维护代码的工作树：`/Users/didi/Desktop/github/LLM-TS-oracle`，分支 `research/direction-conditioned-forecasting`。实际模型推理/训练在 GPU_Foundation。运行目录分别用 `runs/YYYYMMDD_etth1_llm_labels_eval_v1` 和 `..._retrain_v1`，不写入历史运行目录。

每个运行保存：任务 manifest、完整配置、代码快照/哈希、数据与标签缓存哈希、checkpoint 来源与哈希、环境版本、窗口/变量映射、预测与真实数值、L1/L2 输入、每窗口每变量 MSE/MAE、聚合指标、运行日志和状态。Git 仅保存代码、配置、轻量指标与报告，不保存原数据、标签明细缓存、checkpoint、完整预测数组或密钥。

执行前运行 training-autoresearch preflight 和一次模型/标签合同验证。资源方案建议沿用两张 A6000、单卡单队列，但当前不启动任务，也不自动复用已结束的 10 小时预算。正式执行的资源和总时限应写入该轮单独 budget.json。

无效标签、窗口错配、checkpoint 不兼容、非有限值、OOM 或超时都停止队列并保留已完成结果；不自动改 batch、重训基线、换 prompt 或扩大数据范围。

## 10. 不新增 API 时，第 2 组的探索性备选（未选定）

若希望当前缓存即可完成一轮“用 LLM 标签训练”的试验，可将已标注的原验证区间创建一个**新 benchmark 版本**，按时间划分训练/选择/探索评估。不得覆盖原 split，更不能仍称它为原训练集重训。

一个预先可冻结的具体划分如下。边界按 32 窗口组对齐，在两处去掉各 64 个预测起点，防止前一部分的未来 64 步监督目标跨入后一部分的目标区间：

|用途|预测起点（含两端）|窗口数|
|---|---:|---:|
|局部训练|8640–10175|1,536|
|边界隔离，不训练/评分|10176–10239|64|
|checkpoint 选择|10240–10623|384|
|边界隔离，不训练/评分|10624–10687|64|
|探索评估|10688–11456|769|

标签总计仍来自现有缓存；新增 API=0。历史输入可以包含当时已观测的早期数值，与真实滚动预测一致；目标区间隔离不等于历史输入完全不重叠，统计依赖仍需披露。

该备选中，20 个 G2 模型仍遵循第 5 节 A 从随机初始化联合训练、B 冻结原 baseline 后训练新校正块的方式，超参数保持第 6 节。只在局部训练区优化，只用选择区的 LLM 条件预测 MSE 选 checkpoint。G1/G3/G4/旧 zero 全部在同一 769 个探索评估窗口重新推理，不重训这些参照。

但此设计有两项不可消除的局限：一是旧模型使用原训练区的 8,065 个窗口，新 G2 只使用较晚的 1,536 个窗口；二是旧 checkpoint 的选择和上游开发已经接触过整个原验证区间。因此它能描述小规模适配现象，不能将 G2−G1 解释为纯标签训练策略效果，也不是独立测试结论。

**建议先执行第 7 节的同窗口替换评估，再决定是否接受这个局部训练探索。** 若要求严格回答用户原始的第 2 组问题，就保留原 split，等待原训练集预测标签；在“禁止新增 API”的当前范围内，不伪造完整四组已可运行的结论。
# Execution authorization update — 2026-10-19

User has now authorized ETTh1 G1/G2 execution on two server A6000 GPUs.
The authoritative fixed-task execution plan is [downstream/PLAN.md](downstream/PLAN.md).
All earlier "not yet authorized" or incomplete-label notes below are historical.
Original splits are retained; no local re-split alternative is selected.
