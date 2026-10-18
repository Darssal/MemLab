# 整理来源

本仓库从已经公开且完成去敏审查的 MemLab 当前源码整理为独立项目。首次导出仅包含原工程目录的 Git tracked 文件，并建立新的 Git 历史；没有复制原研究工作区、历史提交或生成目录。

- 来源：[Research-Code-Collection / projects/memlab](https://github.com/Darssal/Research-Code-Collection/tree/8f143f67598f88fbb8a16d599a7f3d17c623d281/projects/memlab)
- 固定来源提交：`8f143f67598f88fbb8a16d599a7f3d17c623d281`
- 独立整理日期：2026-10-19
- 工程笔记仅提取同一提交中 MemLab 相关段落，并将链接改为当前仓库路径。

README、文档入口、发布边界策略与 CI 随独立仓库根目录调整。Python 模块、配置、服务脚本和 53 项离线测试沿用该提交。后续维护以本仓库为项目入口。

## 上游边界

DeerFlow 是外部依赖，固定参考版本为 `7de94394d4295182701ffb47e938e7c39b963091`；未在本仓库复制其源码或 Git 历史。ByteDance Ltd. 与 DeerFlow Authors 的 MIT 署名和许可链接保留于 [THIRD_PARTY.md](../THIRD_PARTY.md)。本仓库不将其上游许可套用于其他实验代码。

论文、数据集、真实对话、运行日志、服务存储和模型凭据没有导出。整理来源记录不代表完整论文实验已经复现。
