# 环境配置

先按 README 安装并运行离线测试。以下服务路径作为手动配置入口提供，本次整理没有启动它们。

## 本地基础设施

```bash
python -m pip install -e ".[infra]"
```

复制 `.env.example` 为私有 `.env`，填入 `POSTGRES_USER`、`POSTGRES_PASSWORD`、`POSTGRES_DB`。Docker Compose 会读取本目录 `.env`，缺少这些值时会拒绝启动。`PG_URI` 需使用与数据库一致的连接信息。

Python 代码不会自动读取 `.env`；运行 Python CLI 前，须将 `PG_URI`、`REDIS_URL`、`QDRANT_URL` 等值设置到当前 shell。请自行使用合适的凭据管理方式，不要将连接串或 `.env` 提交到 Git。

```bash
docker compose -f infra/docker-compose.yml up -d
```

Compose 服务端口只绑定本机回环地址。Redis/Postgres/Qdrant 的版本来自原工程配置，此整理过程没有拉取镜像、核实镜像可用性或做服务兼容性验证。Windows 可使用 `scripts/up_infra.ps1` 和 `scripts/check_infra.ps1`。停止使用 `docker compose -f infra/docker-compose.yml down`，该命令默认保留命名卷。

## 合成负载

不需要复制原工作区数据。生成器使用固定种子构造虚拟文档与访问记录：

```bash
python -m memlab.workloads.gen_corpus --docs 4 --tasks 4 --doc-bytes-min 64 --doc-bytes-max 128 --out data
python -m memlab.workloads.gen_trace --agents 2 --duration-s 1 --out traces/demo.jsonl
```

这些命令写入已忽略的目录；文件不应作为真实研究样本或性能结果发布。stub runner 是基础设施实验入口，不能因为名字含 stub 就视作无服务演示。

## 外部 DeerFlow 与模型

自行准备 [DeerFlow](https://github.com/bytedance/deer-flow) checkout，并选择版本 `7de94394d4295182701ffb47e938e7c39b963091`。按该版本的上游文档安装 backend 环境并配置服务。本子工程不分发该源码、虚拟环境或配置文件。

设置 `MEMLAB_DEERFLOW_ROOT` 为外部 checkout 路径；旧适配器使用 `DEERFLOW_BACKEND_PATH`，如使用该适配器还应设置到 checkout 的 `backend` 目录。设置自己的 `OPENAI_BASE_URL`、`OPENAI_MODEL`、`OPENAI_API_KEY`。E2E 运行要求显式配置这些值，公开版没有预设供应商、免费额度或默认模型。此处变量命名只是现有兼容协议约定，不要求使用特定模型供应商。

嵌入式 worker 仍按原工程的 Windows backend 虚拟环境布局寻找 `.venv/Scripts/python.exe`。跨平台嵌入式路径、DeerFlow API 兼容性和在线推理均未在此次整理中验证。

## 可选基准准备脚本

`benchmarks/public_main/longmemeval/prepare.py` 保留纯源码，profile 判断可离线测试。显式运行准备命令前需安装 `.[benchmarks]`，核对数据集来源、访问权限和使用条件。该命令会联网下载数据，默认输出在忽略目录中；不属于离线验证流程。
