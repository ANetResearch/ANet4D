"""M26 智能体观测 SDK 的 Env 侧（AWR-05 §5.5；`bench/obs.v1` 的唯一出口与 `bench/act.v1` 的唯一入口）。

- `awr.agentsdk.env`：lockstep Env（BenchEnv）、观测投影（L1 检测级）、导航解、动作守卫、链路层与运行写入器。
- `awr.agentsdk.task`：由事件版本目录生成任务契约 `bench/task.v1` 与 Agent 简报。
- `awr.agentsdk.skills`：指挥机队的封闭技能词表（`bench/decision.v1`）与把技能展开成逐 tick 动作的执行器、机队阶段机（ADR-227）。
- `awr.agentsdk.gateway`：外部智能体的本地 HTTP 指挥网关：观测 / 动作交换、墙钟配速、像素到公开地形的投影（ADR-227）。

Agent 侧（研究者实现的 DroneAgent、基线策略）在独立 SDK 包 `anet4d`（tools/sdk/anet4d；旧名 `drone4d` 为已弃用的兼容别名），不导入本包。
"""

from __future__ import annotations
