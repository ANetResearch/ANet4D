# 技能词表与指挥网关移入记录（第 6 步，ADR-227）

2026-11-05。分支 `engine/skills-gateway`，基于 main `47b6566`（5e 合并后）。设计与接口说明见
[docs/sdk/skills-gateway.zh-CN.md](../sdk/skills-gateway.zh-CN.md)；本文记录移入的来源、改动与验证。

## 1. 来源与对应

| CHIMERA 模块（2026-11-05 `main` `0af51b4`） | Anet4D | 改动 |
|---|---|---|
| `chimera/commander/skills.py` | `awr/agentsdk/skills/runner.py` | 算法与数值不变；`drone4d.agent` 的动作构造改为 Env 侧副本 `skills/actions.py`；`point_in_polygon` 随带 |
| `chimera/sandbox/bridge.py` 的指令校验 | `awr/agentsdk/skills/__init__.py` | 词表表化为 `SKILLS`；文案不变；`speed` 加入数值校验；报告校验改为任务层注入 |
| `chimera/commander/harness.py` 的逐机阶段与逐 tick 动作 | `awr/agentsdk/skills/fleet.py` | 从 `Commander.run` 抽出为 `SkillFleet`（`observe` / `command` / `actions`）；规则与文案不变 |
| `chimera/commander/terrain.py` | `awr/agentsdk/skills/terrain.py` | 栅格读取与双线性插值 → `GridTerrain`；派生 2 m 层的选择留在 CHIMERA |
| `chimera/sandbox/bridge.py` 的交换 | `awr/agentsdk/gateway/bridge.py` | 账本改为 `pre_validate` 钩子（持锁、在词表校验之前） |
| `chimera/commander/realtime.py` | `awr/agentsdk/gateway/realtime.py` | 不变（线程名 `anet4d-policy`） |
| `chimera/sandbox/server.py` 的处理器与 `main` | `awr/agentsdk/gateway/server.py` | 路由与错误码不变；任务层改为 `Gateway` 子类的 `status` / `public_tasks` / `start` / `project` |
| `chimera/sandbox/projection.py`、`telemetry.py` | `awr/agentsdk/gateway/projection.py` | 不变 |
| — | `packages/contracts/bench/decision.v1.schema.json` | 新契约：决策对象与按技能的 `SkillCommand`；Python / TS 类型再生成 |

## 2. 测试

| 位置 | 用例 | 内容 |
|---|---|---|
| `tests/d3/skills/test_skills_vocab.py` | 27 | 词表封闭；10 条有效指令同时通过校验函数与契约；18 条无效指令两边都拒绝且文案一致；决策形状；有限数；任务层报告校验注入 |
| `tests/d3/skills/test_skills_runner.py` | 8 | 自 CHIMERA `test_skill_terrain.py` 移入并改为断言 `act.v1` dict；另加拒绝文案、速度夹限、割草机航线、与 `anet4d.agent` 的构造对拍 |
| `tests/d3/skills/test_skills_fleet.py` | 3 | 起飞 → 待执行技能 → rtl → landed_final 全程动作；电量耗尽；接口拒绝与 zoom 钩子 |
| `tests/d3/skills/test_gateway.py` | 16 | 自 CHIMERA `test_sandbox_bridge.py`、`test_sandbox_http.py`、`test_realtime.py` 移入；另加账本钩子次序、令牌文件权限、投影几何 |
| `tests/d3/env/test_skill_fleet_env.py` | 1 | 真实 `BenchEnv`（合成世界、L1、fast、锁步）：goto 起飞、到点、rtl 落地结束；无被守卫拒绝的动作（需 open3d 与 Node 构建合成世界，CI 的 no-data 分片跳过，开发机 71 s 通过） |

## 3. CHIMERA 等价见证

CHIMERA 分支 `anet4d-interface`（工作树 `chimera-p6`）与切换前的 `main` `0af51b4`，在同一 Anet4D 工作树上跑同一脚本计划：合成世界
`synth-glb` 与合成事件（Anet4D 测试夹具，寻路后端 grid）、两架 x500、`physics = kinematic`、L1 观测、锁步、决策周期 5 s、种子 7、
130 s；计划覆盖 goto、follow（standoff、speed）、look_at、orbit、sweep（lane）、速度夹限、rtl，两条被拒绝的指令（`orbit needs x, y`、
`unknown skill 'land'`）、一次无镜头的 zoom（`Optical lens unavailable in this run`）、地面上被忽略的 hold、未知机号、一条 found 报告与 done。

| 结果 | 文件 |
|---|---|
| 逐字节相同（22 个） | `chimera/telemetry.jsonl`、`reports.jsonl`、`dialogue.jsonl`、`done.json`、`system_prompt.txt`、`clock.jsonl`、`timebase.json`、`fleet-config.json`、`sensor-schedule.json`、`usage.jsonl`、`harness-notes.json`（三条拒绝文案）、`truth/run.mcap`、运行种子等 |
| 去掉墙钟字段后相同（2 个） | `chimera/summary.json`（`wall_s`、`model_wall_s`）、`chimera/clock-summary.json` |
| 只差墙钟（3 个） | `manifest.json`（`agent_wall_s`、记录器的 `wall_s` / `write_s` / `block_*` / `max_depth`）；`raw.mcap`（579,537 字节中 22 字节不同，全部在 `run.wall` 元数据记录的起止 unix 时间戳内）；`frames.idx`（只含 `raw.mcap` 的 sha256） |

两侧 `flight_s` 均为 uav-01 124.9 s、uav-02 114.9 s，1250 tick，1 条报告，125 s 由计划宣告 done。脚本在会话暂存区
`p6/witness.py`，产物在开发机 `.tools/p6-witness/`（不入库）。

CHIMERA 分支的全量 `tests/`（同一 Anet4D 工作树）：1945 通过、88 跳过、27 失败；这 27 项在切换前的 `main` 上同样失败（按钉住的上游
`705a1be` 加补丁写成，如 `Director(place_arrival_radius_m=…)`、`ActorFrame(speed=…)` 在当前 Anet4D `main` 上不存在），分支没有新增失败。

## 4. 没做的与后续

- CHIMERA 在 dmax 上钉住上游 `705a1be` 加 4 个本地补丁，其中 3 个在当前 `main` 上不再适用；CHIMERA 真正在新接口上运行须先重做或合入
  这些补丁并升级钉版（F-153）。CHIMERA 分支因此不并入其 `main`，等钉版升级一起。
- 词表只有多旋翼技能；地面车、无人艇、固定翼的词表另立（F-154）。
- CHIMERA 的系统提示文案仍手写在 `harness.py`，没有从 `SKILLS[*].help` 生成。
