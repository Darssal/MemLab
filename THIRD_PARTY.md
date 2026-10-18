# 第三方来源

- **DeerFlow**：上游 [bytedance/deer-flow](https://github.com/bytedance/deer-flow)，适配参考 commit [`7de94394d4295182701ffb47e938e7c39b963091`](https://github.com/bytedance/deer-flow/tree/7de94394d4295182701ffb47e938e7c39b963091)。该固定版本采用 [MIT 许可证](https://github.com/bytedance/deer-flow/blob/7de94394d4295182701ffb47e938e7c39b963091/LICENSE)，署名 ByteDance Ltd. 和 DeerFlow Authors。本仓库不复制其源码；自行获取时应保留并遵循上游许可证。
- **LongMemEval**：保留准备适配代码，不分发数据。脚本中的默认数据来源为 Hugging Face `yuyijiong/LongMemEval_24k`；运行前自行核对该来源、许可证及其与原始基准的关系。
- **Python 依赖**：Pydantic、PyYAML、HTTPX、Redis、asyncpg 等通过包管理器安装，不在本仓库 vendoring。各自的许可证与使用条件以其发行版本为准。

原代码工作区未提供 MemLab 自身的许可证。此整理过程不代替权利人选择许可证，也不把第三方工作表述为个人独立实现。
