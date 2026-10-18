# 架构与代码导航

`memlab.memory_manager.manager.MultiLevelMemoryManager` 统一处理写入、检索和上下文组装。各层通过独立适配器连接，配置见 `memlab/config.py`。

| 部件 | 职责 | 主要实现 |
| --- | --- | --- |
| L0 | 保留最近消息、限制检索注入预算、组装上下文 | `memory_manager/l0.py` |
| L1 | Redis 缓存与批量写入队列 | `memory_manager/l1.py`、`backends/redis_store.py` |
| L2 | Postgres 元数据、Qdrant 向量检索 | `memory_manager/l2.py`、`memory_manager/vector.py` |
| L3 | 文件对象存储及大内容指针 | `memory_manager/l3.py` |
| 运行控制 | 分支标识、失效记录、验收字段和回滚阶段 | `pipeline/control_plane.py` |
| 汇总 | 从运行 manifest 和事件记录生成表格 | `metrics/summary_builder.py` |

`runner_stub.py` 生成确定性的访问负载；`runner_e2e.py` 执行外部模型/DeerFlow 路径。它们需要真实服务配置。`envcheck.py` 检查服务与模型环境，失败时记录原因。公开版没有把环境失败自动解释为实验通过。

测试中的小数、延迟值、任务编号和断言值是构造输入及预期值。源码中的预算、阈值、批量大小和默认种子属于算法/运行参数，不能作为实测性能或论文结果引用。轻量 `hash_embedding` 用于确定性实验，其实现不等同于高质量语义嵌入模型。

历史整体 orchestrator 依赖未发布的实验矩阵和报告脚本，已从此公开子工程排除。这里保留的是组件、独立 runner 和验证逻辑。

共享记忆管理器只安装一次操作 logger；会话调用通过 `ContextVar` 记录当前 agent，避免并发 worker 覆盖 logger。运行初始化和独立后台批次没有单一会话归属，其 agent/session 字段可为空，不能强行分配给最近执行的会话。
