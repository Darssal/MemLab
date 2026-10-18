# 工程实现与设计取舍

## MemLab：控制上下文注入，并记录运行是否有效

**问题。** 更多记忆不一定应该全部放进提示：候选可能重复、相关性不足，或与现有工具证据重叠。多个异步会话共享记忆管理器时，日志归属也不能依赖反复覆盖的全局回调。实验成功执行后，还需要判断其配置和记录是否满足比较条件。

**机制。**

- **上下文组装：** [`l0.py`](../memlab/memory_manager/l0.py) 先保留近期消息，再按检索分数排序候选；过滤低相关项，按键与版本去重，并用规范化文本前 512 个字符的哈希去除重复内容。选中记忆受注入预算约束，最终组装还会从前部裁剪消息。 [`manager.py`](../memlab/memory_manager/manager.py) 在已有直接工具证据时可绕过检索，并返回检索跳过原因、缓存命中和降级轨迹。
- **分层放置：** 同一管理器按重要度与对象大小选择 L0 近期上下文、L1 Redis、L2 Postgres / Qdrant 或 L3 对象存储。大对象可保存为正文加指针；异步写入队列提供批处理与背压策略。服务失败会进入降级记录，后台写入失败会向调用方暴露。
- **并发归属：** [`session_logging.py`](../memlab/session_logging.py) 安装共享操作 logger，从 `ContextVar` 读取当前 agent。会话 worker 设置上下文并在 `finally` 中恢复；A、B 会话交错执行时，各自的操作仍读取各自上下文。初始化和独立后台批次没有唯一会话归属时，字段可以为空。
- **运行有效性：** [`control_plane.py`](../memlab/pipeline/control_plane.py) 区分 accepted、invalid 和 diagnostic 等状态。新格式运行只有在已接受、汇总链有效、非诊断且未被替代时才进入正式报告；旧格式保留环境、复现和真实模型标志的兼容检查。失效运行记录异常、确定最早回滚阶段并生成重跑计划。[`summary_builder.py`](../memlab/metrics/summary_builder.py) 据此排除不合格运行。

**小例子。** 近期消息已经包含所需工具证据时，策略可以直接组装上下文，并记录 `direct_tool_evidence_present`；无需为了“使用记忆”再次检索。另一条运行即使生成了答案，若 manifest 被标记为 diagnostic，也不会成为正式对比表中的有效样本。

**设计取舍。** token 数按 `ceil(len(text) / 4)` 估算，属于启发式预算；它不是模型 tokenizer 的精确计数，单条过长的最终用户消息也不会被该裁剪循环删除。文本去重只比较规范化后的前缀，并非语义去重。默认 `hash_embedding` 用于确定性实验，不等同于经过训练的语义嵌入模型。`ContextVar` 解决异步调用上下文的日志归属，没有使跨服务写入成为事务，也不会为所有后台工作自动补上会话身份。有效性门禁约束报告输入，不能替代实验设计与人工判断。

**测试入口：** [检索绕过策略](../tests/test_memory_manager_policy.py) · [L3 指针化与降级轨迹](../tests/test_system_rebaseline.py) · [交错会话日志归属](../tests/test_stub_event_attribution.py) · [失效运行与回滚](../tests/test_control_plane.py) · [汇总排除与作用域](../tests/test_branch_audit.py) · [指标分母](../tests/test_metrics_denominator.py)

