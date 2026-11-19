# RedisDoctor

RedisDoctor 是面向 Redis 运维与技术支持场景的证据约束问答工具。它先依据 Redis 版本、模块和来源元数据筛选证据，再生成带引用的结构化回答；证据不足或引用不合法时会拒答。

## 处理流程

1. 读取 Markdown 文档或 JSON 语料，并保留版本、模块、来源 URL 等元数据。
2. 路由问题并检索候选证据。
3. 执行版本兼容过滤、引用校验和证据不足拒答。
4. 输出结构化回答：意图、模块、检索模式、引用来源与拒答原因。

## 可复现运行示例

```powershell
python -m redisdoctor.cli "为什么 AOF 重写可能带来延迟？" `
  --document examples/redis/persistence.md --version 7.2 --module persistence

powershell -ExecutionPolicy Bypass -File scripts/run_redisdoctor_example.ps1
```

## 评测摘要

固定的版本感知 RAG 基准包含 17 个带来源标注的问题；词法 RAG 的 Recall@5、MRR 和回答准确率均为 92.86%，引用约束满足率为 100%。完整的评测范围、QLoRA 人工盲评结果和解释边界见 [评测摘要](../docs/evaluation.md)。

## QLoRA 训练

训练部分使用 `Qwen/Qwen3-4B`、4-bit QLoRA 与 LLaMA-Factory。公开仓库保留最终训练配置、训练/评测样例和评测汇总；完整标注数据与远端运行材料不公开。详见 [训练说明](../training/redisdoctor/README.md)。

## 边界

- `--generator openai` 支持 OpenAI 兼容接口；密钥只能通过环境变量提供。
- 当前正式评测使用自定义的版本感知检索、证据过滤与引用验证链路。
- LightRAG 仅为可选集成方向，尚未作为本项目已完成的正式实验能力。
