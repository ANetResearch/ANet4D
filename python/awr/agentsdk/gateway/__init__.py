"""指挥网关（引擎计划第 6 步，ADR-227）：外部智能体经本地 HTTP 接口指挥一个 `BenchEnv` 回合，世界按墙钟独立推进。

| 模块 | 内容 |
|---|---|
| `bridge` | `DecisionBridge`：一次只有一个决策在途的观测 / 动作交换，`validate_decision` 校验，`SessionCancelled` |
| `realtime` | `RealtimeClock`（1 s 仿真 = 1 s 墙钟，滞后超限判为基础设施失败）、`DecisionWorker`（策略调用在工作线程，仿真只在主线程改） |
| `server` | `Gateway` 基类与 HTTP 处理器：`/v1/status`、`/v1/tasks`、`/v1/sessions`、`…/observation`、`…/actions`、`…/project`、`…/cancel`，Bearer 令牌 |
| `projection` | `camera_metadata`（交付帧的公开标定与导航）与 `project_pixel`（像素 → 公开地形的地面点，不查演员） |

任务层（CHIMERA）在 `Gateway` 子类里实现会话的创建、运行与评分，在 `DecisionBridge` 的 `pre_validate` 里实现调用账本；
网关协议、决策校验与投影工具属于 Anet4D，外部智能体的客户端（`examples/http_agent.py` 式的轮询）不随任务层变化。

2026-11-05 自 CHIMERA 基准移入（`chimera.sandbox.bridge`、`chimera.sandbox.server`、`chimera.sandbox.projection`、
`chimera.sandbox.telemetry`、`chimera.commander.realtime`）；协议与文案不变。
"""

from __future__ import annotations

from awr.agentsdk.gateway.bridge import DecisionBridge, SessionCancelled
from awr.agentsdk.gateway.realtime import DecisionWorker, RealtimeClock, RealtimeOverrun
from awr.agentsdk.gateway.server import Gateway, handler, load_token, serve

__all__ = ["DecisionBridge", "DecisionWorker", "Gateway", "RealtimeClock", "RealtimeOverrun", "SessionCancelled", "handler", "load_token", "serve"]
