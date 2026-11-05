# ETTm1 冻结部署包

用 Python、NumPy 和 PyTorch CPU 运行。在已验证的 Mac 环境使用 Python 3.9.25、NumPy 2.0.2、PyTorch 2.8.0。解压后提供一个包含连续512条观测的 CSV，表头为 `date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT`，按15分钟升序排列。此模型面向 ETTm1 训练期之后、同一变量顺序和采样频率的数据。

```bash
python predict.py --plan frozen_plan.json --history history.csv --output next64.csv
```

默认执行验证选定方案。加 `--method M1` … `--method M10` 可选固定配置，加 `--method LLM5` 可选验证最优的 LLM 提案。输出是随后64个时间点、七个变量的原始量纲预测。

Python 接口：

```python
from forecast import load_forecaster
model = load_forecaster("frozen_plan.json")
forecast = model.predict(history, timestamps, model.plan["selected"])
# history: [512, 7]，timestamps: [512]，forecast: [64, 7]
```

部署依赖只有本包的程序、冻结配置和 tools.npz（含标准化相关工具、训练历史案例、回归权重、残差树参数与 DLinear 权重）。冻结清单会核对配置、数值状态与运行时代码的哈希。按训练标准差计算误差时，可调用 `predict_batch(..., standardized=True)`。独立打包的文件中没有验证缓存、未来实测目标或远程 API 凭证。
