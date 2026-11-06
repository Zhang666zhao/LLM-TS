# 第一次运行 Agentic TS Forecast

如果你刚接手这个项目，可以先从 ETTh1 的一个例子看起：同一个负荷变量，两个工具都能预测，怎样决定该取谁的结果？[HTML 论文](../../reports/agentic_ts_forecast/ETTh1_Top10_NeurIPS_Style.html)的图2展示了实际的选择过程。

如果你更想动手，先运行下面的十种方法自检就好，这一步只需几秒，也用不到密钥。之后有两条路：想核对论文里的数字，继续第 2、3 节；想把自己的512条观测变成一份预测，看第 4 节。

## 1. 克隆和安装

以下命令用于 macOS / Linux。建议使用 64 位 Python 3.9–3.12，并预留至少 2 GB 临时磁盘空间（另计 Python 环境）。

```bash
git clone --branch agentic_ts_forecast --single-branch https://github.com/Zhang666zhao/LLM-TS.git
cd LLM-TS
python3 -m venv .venv
source .venv/bin/activate
cd methods/agentic_ts_forecast
python -m pip install -r requirements-local.txt
python smoke.py
```

看到 M1–M10 各自输出 `(1, 64, 7)` 就表示十种组合入口可用。本步骤使用确定性合成输入，不读取原始数据，也不报告实验精度。**以下命令均在 `LLM-TS/methods/agentic_ts_forecast` 目录执行。** Windows 用户可用 `.venv\Scripts\activate` 激活环境。

## 2. 找到与你的工作环境匹配的实验材料

克隆后，你已经有了代码、规则和结果表。要复现同一组数字，还需要与原实验一致的数据和权重。推荐直接下载下面的 Release 冻结包。已有完整副本或共享存储的同学，也可以使用后面的 A、B、C。恢复程序会逐个校验文件。[数据使用说明](../../data/RESEARCH_DATA.md)

**直接下载冻结包（推荐）。** [2026-11-06 运行资源 Release](https://github.com/Zhang666zhao/LLM-TS/releases/tag/agentic-forecast-assets-20261106) 包含两个数据集各自所需的资源，共 66 个文件。ETTh1 包含固定基准的上游预测和核对数组；ETTm1 包含历史输入、权重和拟合后的工具状态。两个包都包含与实验哈希一致的 ETT CSV；数据来源为 [ETDataset](https://github.com/zhouhaoyi/ETDataset)，使用时请遵循原项目的数据许可与引用要求。这是从研究目录整理出的运行依赖包，未包含完整搜索日志和 LLM 原始交互。

```bash
release=https://github.com/Zhang666zhao/LLM-TS/releases/download/agentic-forecast-assets-20261106
curl -fL "$release/agentic-forecast-ETTh1-20261106.tar.gz" -o /tmp/agentic-forecast-ETTh1-20261106.tar.gz
curl -fL "$release/agentic-forecast-ETTm1-20261106.tar.gz" -o /tmp/agentic-forecast-ETTm1-20261106.tar.gz
python restore_assets.py --archive /tmp/agentic-forecast-ETTh1-20261106.tar.gz --dataset ETTh1
python restore_assets.py --archive /tmp/agentic-forecast-ETTm1-20261106.tar.gz --dataset ETTm1
```

恢复程序根据仓库清单逐文件校验 SHA-256，并生成本地窗口索引和示例输入。只运行一个数据集时，下载并恢复对应包即可。Release 另附 `SHA256SUMS`，可用于核对整个压缩包。资源保存在 Git 忽略目录中。

**方式 A：已有完整本地目录。** 用你自己的实际路径替换下例。命令会从既有完整包复制两套实验的必需文件并校验 SHA-256。

```bash
python restore_assets.py --from-package /path/to/full/agentic_ts_forecast
```

**方式 B：已挂载课题组共享存储。** 根目录下应有 `LLM_predict/` 与 `TSRAG/`。目录由维护者提供。

```bash
python restore_assets.py --source-root /path/to/shared/agentic_TS
```

**方式 C：通过已获授权的 SSH 别名访问相同存储。** 需要可用的 SSH 登录与远端 `tar`，不需要远端 Python 环境。程序只读取清单中的文件。

```bash
python restore_assets.py --ssh GPU_Foundation --source-root /path/to/shared/agentic_TS
```

只跑一个数据集时加 `--dataset ETTh1` 或 `--dataset ETTm1`。恢复会跳过已验证文件，遇到哈希不符时报告实际路径并停止。ETTm1 单窗口示例的历史输入在恢复后由本地数据生成。[逐文件来源、大小和校验值](restore_manifest.json)

**245份历史权重的位置。** [forecast-weights-20261102](https://github.com/Zhang666zhao/LLM-TS/releases/tag/forecast-weights-20261102) 是研究分支的历史模型权重 Release，不含数据。这两个实验还需要版本匹配的冻结输入或 ETTm1 工具状态。完整复现应以本分支的恢复清单为准，不能只下载历史权重，也不能将公共 ETT CSV 静默当成相同字节的冻结数据。本分支的精确复现资源现已通过上面的独立 Release 提供，无需共享存储权限。

## 3. 复现指标

| 任务 | 命令 | 预期 |
|---|---|---|
| ETTh1 全量对照 | `python run_local.py` | M1–M10 与 DLinear；验证和测试各 2,817 窗，逐元素和指标检查通过 |
| ETTm1 全量对照 | `python transfer/reproduce.py` | 从原始历史计算，验证、测试各 11,457 窗，每个划分显示 `29/29 passed` |
| ETTh1 单方法 | `python run_local.py --method M2 --split test` | 只测 M2 与对照 DLinear |
| ETTm1 只算测试 | `python transfer/reproduce.py --split test` | 对照全部已冻结的测试 MSE、MAE |

结果写入新的本地目录。ETTh1 使用 `local_results/<运行时间>/`，ETTm1 使用 `outputs/ettm1/<运行时间>/`，两者都输出 `metrics.csv` 和 `report.json`。典型参照值如下：

| 数据集 / 方法 | 测试 MSE | 测试 MAE |
|---|---:|---:|
| ETTh1 / M1 | 0.317932 | 0.356481 |
| ETTh1 / DLinear | 0.353263 | 0.384085 |
| ETTm1 / M1（与 LLM5 同预测） | 0.271753 | 0.328249 |
| ETTm1 / 验证选定 Rule1 | 0.271542 | 0.327961 |
| ETTm1 / DLinear | 0.292529 | 0.339546 |

ETTh1 固定基准包含18个已冻结的上游数值源，由本地重新计算岭回归、DLinear 和31个组合节点。ETTm1 是可部署的轻量工具池，从每个起点的512步历史计算预测，无预计算预测依赖。[ETTh1 完整结果](evaluation/top10_results.csv) · [ETTm1 完整结果](transfer/artifacts/ETTm1/results.csv)

## 4. 用 ETTm1 模型预测自己的新窗口

提供按时间升序的512行 CSV，列顺序为 `date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT`，相邻观测间隔15分钟。输入覆盖128小时，输出随后16小时（64步）的七变量原量纲预测。用恢复步骤生成的示例运行：

```bash
python transfer/predict.py --plan transfer/artifacts/ETTm1/frozen_plan.json \
  --history transfer/artifacts/ETTm1/example_history.csv --output forecast.csv
```

默认执行验证集选定的 Rule1。要运行 M1 等其他冻结配置，加 `--method M1`。输出文件名必须是新文件。推理只需配置、工具状态和历史输入；对新序列预测适用 ETTm1 模型，ETTh1 的入口用于固定基准回放。

## 5. 可选：启用方舟，重新生成 LLM 提案

仅在主动启动新的研究轮次时需要模型权限与计费额度。已有实验的离线核对、单窗口部署均可跳过本节。

```bash
cp -n .env.example .env
# 本机编辑 .env 并填入自己的 ARK_API_KEY，再导入环境
set -a; . ./.env; set +a
python transfer/ark_client.py --check
python transfer/ark_client.py --ping
```

`--check` 只显示是否已配置密钥，缺失时返回非零。**`--ping` 会发出一次真实、可能计费的请求。** 默认模型是 `deepseek-v4-pro`，默认基址是 `https://ark.cn-beijing.volces.com/api/coding/v3`。个人 `.env` 被 Git 忽略。不要在代码、命令记录或 issue 中粘贴密钥。[连接模块](transfer/ark_client.py)

重新研究时使用**新的运行目录**，例如先将原始 `fit.json` 和恢复的 `tools.npz` 复制进去，复用已有训练状态，再执行：

```bash
new_run=$(mktemp -d)
cp transfer/artifacts/ETTm1/{fit.json,tools.npz} "$new_run/"
python transfer/run_pipeline.py validate --out "$new_run"
python transfer/run_pipeline.py freeze --out "$new_run"  # 两轮，每轮三个提案
python transfer/controls.py --out "$new_run"
python transfer/run_pipeline.py test --out "$new_run"
```

以上命令代表新的实验，新的 LLM 响应可能产生不同规则。重现已发表的指标请使用第 3 节。若需重新拟合底层工具，使用 [fit.py](transfer/fit.py)，另行配置获授权的 GPU 训练环境和预算。

## 6. 查代码与排错

- 方法编号和候选配置在 [methods.json](methods.json)。分块选择、三根选择、局部融合分别见 [block_column_selection.py](src/block_column_selection.py)、[column_selection.py](src/column_selection.py)、[horizon_policy.py](src/horizon_policy.py)。
- ETTm1 的部署入口在 [forecast.py](transfer/forecast.py)，冻结方案在 [frozen_plan.json](transfer/artifacts/ETTm1/frozen_plan.json)。`B16` 对每个变量在四个16步块上分别选择来源。`C` 是较早池中逐变量选择的整段预测，`R` 是日历基底加训练误差修正的完整预测。
- 出现缺文件或校验错误时，先确认第 2 节选择了正确数据集与目录，再检查报错对应的清单项。避免通过更改哈希来绕过校验。
- 分支继承自 [research/direction-conditioned-forecasting](https://github.com/Zhang666zhao/LLM-TS/tree/research/direction-conditioned-forecasting)。该分支承载历史方向标签实验；本分支的 ETTm1 部署仅使用可观测历史。


如需接入仓库统一 launcher，在仓库根目录将 `configs/paths.example.yaml` 复制为 `configs/paths.local.yaml`，设置当前 Python 和可写的 `run_root`。恢复材料后，执行 `python scripts/run_method.py --method agentic_ts_forecast --dataset ETTm1 --run-id ettm1_replay_001`。它输出已选方案的标准 `metrics.json`；十种方法的完整对照请使用第 3 节的专用入口。该 adapter 使用原冻结 seed 2021。
