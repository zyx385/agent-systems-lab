# RedisDoctor QLoRA 训练资产

本目录保留 RedisDoctor 最终 QLoRA 实验的可审查资产：训练配置、数据模式、依赖清单、消融运行器，以及脱敏样例。

## 配置

- 基座模型：`Qwen/Qwen3-4B`
- 微调方式：4-bit QLoRA
- 训练框架：LLaMA-Factory
- 最终配置：[qlora_qwen3_4b_reviewed_v3.yaml](autodl/qlora_qwen3_4b_reviewed_v3.yaml)

最终实验使用 352 条训练样本、81 条验证样本和 38 条源隔离评测问题。公开版仅提供与真实 schema 一致的训练和评测样例，完整人工审核数据集不公开。

## 数据格式

[训练样例](../../examples/redisdoctor/final_training_sample.json) 展示 ShareGPT 消息格式；[评测样例](../../examples/redisdoctor/final_evaluation_sample.json) 展示问题、证据和期望引用字段。数据构建的核心校验见 `dataset_schema.py`。

## 运行边界

`autodl/run_ablation.py` 需要本地模型、Adapter、完整私有数据集和 GPU 环境。公开配置用于说明实验条件，不承诺在没有完整数据和模型权重的情况下直接复现正式结果。最终结论及限制见 [评测摘要](../../docs/evaluation.md)。
