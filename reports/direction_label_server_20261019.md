# DAO服务器连通性与ETTh1筛查

## 执行位置更正

之前2048-token筛查是在本地Mac发请求，服务器仅提供数据。本次全部DAO请求由GPU_Foundation执行，后续禁止本地发API请求。

## 连通性

- DNS解析正常，地址10.88.128.112。
- 实验Conda Python默认SSL报unable to get local issuer certificate。
- 使用服务器已有/etc/ssl/certs/ca-certificates.crt后正常，无关闭TLS验证、无安装证书或修改环境。
- 不含数据的Flash最小请求返回HTTP200、finish_reason=stop、content=OK，耗时1.46秒，88输入+14输出=102token。

## ETTh1实际请求

服务器目录：/nfs/dataset-ofs-661-142656/qiminhao/Foundation_model/agentic_TS/TSRAG/runs/20261019_dao_etth1_server8192。

保留原两窗口/两模型设计，输入和标签定义不变，仅输出上限2048→8192、timeout180秒，最大4次/900秒，失败停止。代码副本及运行来源在远端manifest中记录。

第一条Flash请求ETTh1-val-8640：输入43533token（缓存43008）、输出8192token（全部reasoning）、合计51725token；82.11秒；finish_reason=length、最终content为空。整轮93.54秒后停止，其他3请求未发送。模型Pro未测试真实数据。

没有有效标签，因此F1不可评估。远端metrics.json按缺失预测计错产生的零值仅是失败覆盖率记账，不是模型分类能力的有效测量，不能与0.5目标比较。价格未知，未报告费用。

下一步需确认DAO支持的thinking关闭/预算参数，先进行不含实验数据的协议预检，再新开版本重试真实窗口。不得猜测参数已被生效或无限增大token上限。旧记录保留，不覆盖。无GPU训练，无密钥文件。
