# RepoMedic 验证材料

本目录提供可离线回放的成功运行记录、工具轨迹图和备用运行回放视频，用于验证 RepoMedic 的证据闭环。

```powershell
python scripts/replay_repomedic_example.py docs/verification/repomedic-example-run.json
```

配置外部模型环境变量后，也可以运行实时示例：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_repomedic_example.ps1
```

该记录展示从 Issue 与失败测试出发，经候选文件定位、工具调用和 Verifier 确认形成诊断结论的过程。它是已保存的成功运行回放，不是一次新的模型调用。
