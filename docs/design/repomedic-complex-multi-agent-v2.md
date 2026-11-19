# RepoMedic Complex Multi-Agent v2 设计

## 目标

把 RepoMedic 从“按近义检索词并行搜索”升级为“按证据缺口调度专项角色”。

v2 不试图让多 Agent 覆盖所有问题。简单、单文件、证据一致的故障仍走单 Agent；只有存在跨模块调用、配置依赖、共享状态或安全边界时，Coordinator 才升级为多 Agent 调查。

本设计新增 12 个**合成但可真实复现**的复杂故障夹具。它们用于评估 Agent 系统；不能表述为来自生产事故或真实开源仓库。

## 当前 Flow 的问题

当前 `Triage → Investigator(search_term) × N → Verifier` 把 `fetch_name`、`async def fetch_name`、`await fetch_name` 一类近义词拆为不同调查分支。各分支重复读取同一源码和测试，汇总后才去重；重复发生在模型调用之前，去重无法回收 Token。

v2 的调查单元改为角色任务，例如“建立路由到存储层的调用链”“验证组织隔离条件”“复现最小失败测试”。同一份共享证据只读取一次，后续角色只能引用或补充它。

## 夹具约束

- 每例至少包含 3 个源码文件和 1 个测试文件；每例有唯一的 `expected_file`，作为主要根因文件。
- 每例的失败必须由 `verification_command` 在干净临时目录中稳定复现；禁止依赖网络、时钟、随机数、数据库或外部服务。
- 每例必须有至少两条不同类型的必要证据，例如调用路径加状态键、配置源加解析逻辑、授权条件加失败断言。
- 不允许仅替换函数名、变量名或错误文案来凑数量；一个根因模式只出现一次。
- 现有 12 个简单夹具保留为控制组；Complex-12 是独立复杂组，结果必须分开报告。

## Complex-12 案例设计

| ID | 故障与涉及模块 | 主要根因文件 | 失败测试的关键条件 | 需要的专项角色 |
| --- | --- | --- | --- | --- |
| `tenant-cache-key` | Route 传入 `tenant_id`，权限服务从缓存读取角色，缓存键忽略租户 | `app/permissions/cache.py` | 同一用户在两个租户角色不同，第二次读取返回第一个租户权限 | 调用链、状态/缓存、安全、测试 |
| `resource-scope-bypass` | 路由校验登录，Service 请求资源，Repository 查询仅按资源 ID | `app/repositories/documents.py` | 用户属于组织 A，却能读取组织 B 的同 ID 文档 | 调用链、安全、测试 |
| `async-session-lifecycle` | API 创建后台任务，Service 捕获数据库会话，任务执行时会话已关闭 | `app/jobs/dispatch.py` | 请求返回后执行队列任务，访问会话抛出已关闭错误 | 调用链、异步生命周期、测试 |
| `webhook-idempotency-race` | Webhook Handler 和订单服务同时处理重复事件，去重标记在副作用之后写入 | `app/services/webhooks.py` | 同一 event ID 连续处理两次，产生两笔订单 | 状态/并发、调用链、测试 |
| `retry-transaction-leak` | Service 重试事务，Repository 捕获异常但没有回滚，第二次尝试继承失败状态 | `app/services/payments.py` | 第一次写入失败、第二次成功；实际仍抛事务不可用错误 | 调用链、状态、测试 |
| `feature-flag-coercion` | Config Loader 读取环境变量字符串，Flag 服务以真值判断 `"false"` | `app/config/flags.py` | `FEATURE_EXPORT=false` 仍暴露导出接口 | 配置、调用链、安全、测试 |
| `config-precedence-drift` | 默认配置、文件配置和环境变量合并顺序错误，生产覆盖值被默认值覆盖 | `app/config/settings.py` | 环境变量指定超时，最终客户端仍使用默认超时 | 配置、调用链、测试 |
| `timezone-cache-mismatch` | API 解析含时区时间，Service 归一化，Cache 使用本地时间字符串做键 | `app/services/calendar.py` | 等价 UTC 时刻产生两个缓存条目并返回过期数据 | 调用链、状态、测试 |
| `serialization-cycle` | Controller 返回对象图，Serializer 递归遍历 parent/child 双向引用 | `app/serializers/project.py` | 项目引用任务、任务反向引用项目，响应递归超限 | 调用链、对象图、测试 |
| `adapter-signature-drift` | Service 调用存储 Adapter，Adapter 已改为关键字参数但包装层保留旧位置参数 | `app/adapters/storage.py` | 上传带 `content_type` 时，适配器收到错位参数 | 调用链、依赖接口、测试 |
| `stale-search-index` | 更新文档后写入主存储，但索引更新事件使用旧对象快照 | `app/indexing/events.py` | 修改标题后全文搜索仍返回旧标题 | 调用链、事件状态、测试 |
| `exception-translation-mask` | Middleware 将领域权限异常转换为通用 500，而 Controller 未保留错误类型 | `app/middleware/errors.py` | 无权限请求应返回 403，实际返回 500 | 调用链、安全、测试 |

## 角色协议

### 1. Coordinator

Coordinator 只负责路由，不得直接确认根因。它先读取 Issue、Traceback 和最多两项初始搜索证据，然后输出：

```json
{
  "complexity": "simple | complex",
  "reason": "为什么需要或不需要协作",
  "roles": ["call_chain", "security", "state", "config", "async", "test_forensics"],
  "candidate_files": ["仓库相对路径"],
  "evidence_questions": ["每个角色必须回答的具体问题"]
}
```

`simple` 只能调用单 Agent 工具循环。`complex` 至少选择两个职责不同的角色，且不得因同义检索词创建多个角色。

### 2. 专项调查角色

所有专项角色接收同一个只读 `SharedCaseState`：Issue、Traceback、当前证据、已读文件、候选文件和未解决问题。它们不得重新读取已经在共享状态中存在的完整文件，除非请求不同的行范围并说明原因。

| 角色 | 唯一职责 | 可输出的结论 |
| --- | --- | --- |
| `call_chain` | 从入口、导入和函数调用建立跨文件数据流 | 调用边、参数在何处丢失/变形、候选文件 |
| `security` | 检查身份、租户、组织、资源范围和异常到 HTTP 状态映射 | 缺失的授权条件或错误映射 |
| `state` | 检查缓存键、幂等键、事务、事件快照和可变共享状态 | 状态边界或顺序错误 |
| `config` | 检查默认值、文件、环境变量、类型转换和优先级 | 配置来源、最终值和变形点 |
| `async` | 检查 await、后台任务、会话/资源生命周期和重试 | 生命周期边界或未等待操作 |
| `test_forensics` | 读取相关测试并运行白名单测试，缩小最小失败条件 | 失败断言、返回码、与候选文件的绑定 |

每个角色必须输出 Pydantic 校验的 `RoleFinding`：

```json
{
  "role": "state",
  "question": "缓存键是否包含 tenant_id？",
  "evidence_ids": ["evidence-3", "evidence-5"],
  "candidate_files": ["app/permissions/cache.py"],
  "claim": "缓存键只使用 user_id，跨租户复用角色结果。",
  "confidence": 0.82,
  "open_questions": ["失败测试是否能证明跨租户读取？"]
}
```

没有引用证据、越界路径、未定义角色或与职责无关的结论必须被拒绝，记录为无效角色输出，不得中断其他角色。

### 3. Evidence Coordinator

Evidence Coordinator 不新增模型角色；由 Python 合并 `RoleFinding`：按文件、行号、片段去重，保留贡献角色，拒绝没有证据的候选。它只将“尚未回答的问题”交给下一角色，避免重复搜索。

### 4. Verifier

Verifier 独立于调查角色。它只能在候选文件与失败测试之间存在导入、调用、失败输出或明确证据关系时运行白名单命令。验证输出须包含命令、返回码、失败片段和确认/拒绝理由。

任何角色的自然语言结论都不能替代 Verifier 的实际测试。

## 预算与失败隔离

- 复杂案例的整例上限固定为 `35,000` Token；这是所有角色共享上限，不是每个角色各有 35,000 Token。
- Coordinator 最多选择 3 个专项角色；每个专项角色最多 3 轮工具调用；Verifier 最多 3 轮。
- 单个角色的 `invalid_json`、超时或无效工具调用只产出结构化失败 Finding，其他角色继续；只有 Coordinator 或最终汇总无法生成 Pydantic 输出时案例才为 `error`。
- `run_command` 工具参数必须是 `list[str]`。字符串命令先作为无效调用返回给该角色，由其在剩余轮次中改正；不得在安全层静默执行。

## 公平评测设计

### 分组

- `Simple-12`：现有 12 个单文件夹具。
- `Complex-12`：本设计的 12 个跨模块夹具。

### 对照

在每个分组上均比较单 Agent 与 v2 多 Agent。两组固定使用同一模型、温度、工具集、Prompt 安全约束、35,000 Token 总上限、工具超时和测试命令白名单。

### 指标

- Top-1 主要根因文件定位率；
- 已验证根因率；
- 角色问题覆盖率：每例声明的必要证据问题中，被有效 Finding 回答的比例；
- 平均 Token、延迟、工具调用数、无效工具率、拒答率和错误率；
- 重复读取率：同一完整文件被不同角色重复读取的次数／总读取次数。

结果必须分别报告 Simple-12 与 Complex-12。只有 Complex-12 的已验证根因率提升，同时预算和错误率处于预先设定范围，才能声称“多 Agent 在复杂故障上带来收益”。

## 成功标准

在改 Flow 前，先完成以下可检查产物：

1. `complex_cases.json`：12 个案例，每例包含源码、唯一主要根因文件、验证命令、必要证据问题和建议角色。
2. 单元测试：12/12 夹具稳定失败；案例 ID 唯一；主要根因文件存在；每例至少 3 个源码文件；验证命令在白名单内。
3. 角色协议模型的单元测试：无证据 Finding、重复读取、字符串命令、单角色失败与复杂度错误路由均被拒绝或隔离。
4. 同一模型的真实运行：Simple-12 与 Complex-12 的单 Agent/v2 多 Agent 各 12 条，完整保留审计日志。

在这些条件满足以前，不能把 v2 描述为“多 Agent 更强”；可以准确描述为“已完成针对复杂度自适应协作的设计与可复现评测方案”。
