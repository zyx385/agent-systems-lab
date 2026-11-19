# RepoMedic 演示材料

本目录提供可离线回放的 RepoMedic 成功案例、工具轨迹图和备用演示视频。

```powershell
python scripts/replay_repomedic_demo.py docs/demo/repomedic-demo-success.json
```

配置外部模型环境变量后，也可以运行实时演示：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_repomedic_demo.ps1
```

演示展示从 Issue 与失败测试出发，经候选文件定位、工具调用和 Verifier 确认形成诊断结论的过程。它是已保存的成功运行回放，不是一次新的模型调用。
