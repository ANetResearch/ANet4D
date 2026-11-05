# 技能词表与指挥网关（接口 ⑤ 的动作侧，第 6 步）

## 1. 边界

研究者接口 ⑤（观测与动作）此前只有 Anet4D 一侧：`bench/obs.v1` 的观测白名单与 `bench/act.v1` 的逐 tick 动作。指挥整个机队
用的高层词表（goto、sweep、orbit……）、把词表展开成逐 tick 动作的执行器，以及外部智能体接入用的 HTTP 网关，都在 CHIMERA 基准里
（`chimera.commander.skills`、`chimera.commander.harness`、`chimera.sandbox.*`）。按架构分界"要看场景才能定的归场景进程，其余归
Anet4D"，这三样不看场景、不看任务，属于设备引擎对研究者的接口，2026-11-05 移入 Anet4D（ADR-227）：

| 归 Anet4D（本步移入） | 留在任务层（CHIMERA） |
|---|---|
| 词表与决策校验 `awr.agentsdk.skills`（`SKILLS`、`validate_command`、`validate_decision`），契约 `bench/decision.v1` | 报告的类型与字段（found、track、count、plate、event……）及其校验 |
| 单机执行器 `skills.runner.SkillRunner`、机队阶段机 `skills.fleet.SkillFleet`、公开地形 `skills.terrain.GridTerrain` | 任务、机队与场景的装配，系统提示文案，运行目录的制品布局 |
| 网关 `awr.agentsdk.gateway`：`DecisionBridge`、`RealtimeClock` / `DecisionWorker`、HTTP 处理器与 `Gateway` 基类、`camera_metadata` / `project_pixel` | 会话的准入（是否入榜、自定义 query）、模型调用账本与计费、评分与证据 |

CHIMERA 的对应模块改为从这两个包导入（见 §6）；HTTP 协议、拒绝文案与客户端（`examples/http_agent.py` 式的轮询）不变。

## 2. 词表

| 技能 | 参数 | 含义 |
|---|---|---|
| goto | x, y, [agl, speed] | 飞到一点（地形以上 agl 米，缺省 60），到达后相机对着它 |
| sweep | polygon 或 x, y, [size]; [agl, lane, speed] | 割草机式覆盖，相机垂直向下；lane 缺省为该高度地面足迹高度的 80 %，至少 10 m |
| orbit | x, y, [radius, period, agl, speed] | 绕点盘旋（半径缺省 50 m，不小于 20 m；周期缺省 90 s），相机对着圆心 |
| follow | x, y, [agl, standoff, speed] | 飞到一个（移动的）地面点上方并保持相机对着它；每步重发新位置即跟随；standoff 在来向停短 |
| look_at | x, y | 只改相机瞄点，不改飞行 |
| hold | — | 悬停 |
| rtl | — | 返回起飞点降落，结束该机 |
| zoom | factor（1 至 12） | 光学变焦；只在运行注入了光学镜头时可用 |

一条决策是 `{commands: [{uav, skill, params}], reports: [object], done: bool, summary?, thought?}`（`bench/decision.v1`）。`validate_decision`
按词表校验每条指令，报告只校验为对象数组并交给任务层注入的校验，全部数字须有限。拒绝文案（HTTP 400）：`invalid command`、
`unknown UAV`、`unknown skill`、`missing skill parameter`、`skill coordinates/limits must be numeric`、`invalid orbit`、`zoom must be 1..12`、
`invalid sweep polygon`、`sweep needs polygon or x,y`、`commands/reports must be arrays`、`done must be boolean`、`invalid report`、
`numbers must be finite`。与 CHIMERA 网关相比唯一的收紧：`speed` 与其他数值参数一起在网关校验（此前由执行器在下一 tick 拒绝）。

词表目前是多旋翼的（起飞、悬停、返航落地）。统一 L2 引擎已有地面车、无人艇与固定翼域，它们需要各自的词表（如 drive_to、
station_keep、loiter），登记为 F-154。

## 3. 执行

`SkillRunner`（每机一个）把当前技能展开成 `(nav, gimbal)`：

- 每条航段贴地形：目标点取沿航段最多 25 m 之前、该处地形之上 `agl` 的点；对着远端航点的高度直飞会在上坡处跌出高度包络。
- 高度由指挥者决定，执行器只保留地形以上 2 m 的物理下限；任务的高度包络由评分判定，不在执行器强制。
- rtl 以 `max(包络下限 + 2, 30)` m 的高度返航，距起飞点 2 m 内下 `land`，只下一次。
- 速度 1 至巡航 8 m/s；地形 `terrain(x, y) -> z` 由运行注入（CHIMERA 注入公开 2 m 派生层），缺省用观测里本机脚下的 DTM。

`SkillFleet` 是机队的阶段机：

```
landed ──第一条运动技能──▶ takeoff ──到达 agl − 3 m──▶ airborne ──rtl 完成且落地──▶ landed_final
                                                      │
                                                      └──battery_soc ≤ 0──▶ exhausted（原地 land 一次，不再接受指令）
```

地面上的 hold、rtl 被忽略；起飞中的新指令替换待执行技能；`land` 不是接口里的技能（只用于电量耗尽）；`zoom` 经运行注入的
`zoom(uav, factor, t_s)` 执行，未注入时拒绝 `Optical lens unavailable in this run`。`actions(k, obs, agents, t_s)` 给出本 tick 全部动作
（`bench/act.v1`），守卫照常校验。被拒绝的指令以文案返回给指挥者，不中断回合。

## 4. 网关协议

`awr.agentsdk.gateway.server.handler(gateway, token)` 是 `http.server` 的处理器；任务层给出 `Gateway` 子类并调用 `serve()`：

| 请求 | 用途 |
|---|---|
| `GET /v1/status` | 当前会话与任务层状态摘要（`Gateway.status()`） |
| `GET /v1/tasks` | 公开任务列表，不含真值（`Gateway.public_tasks()`） |
| `POST /v1/sessions` | 创建会话（`Gateway.start(request)`） |
| `GET /v1/sessions/<id>/observation?after=N` | 最多等 15 s 取尚未消费的回合：`system`、`history`、`observation`、`images`、`sensor_metadata`、`sequence`、`world_time_s` |
| `POST /v1/sessions/<id>/actions` | `{"sequence", "action"}`；动作即 §2 的决策 |
| `POST /v1/sessions/<id>/project` | 由当前交付帧估计像素的地面坐标（`Gateway.project`） |
| `POST /v1/sessions/<id>/cancel` | 取消会话 |

错误：未授权 401；未知任务 / 会话 404；输入不合规 400；过期或重复的序号、已有活动会话、源码已变 409。`DecisionBridge` 一次只有
一个决策在途：仿真线程 `complete()` 登记观测后阻塞，HTTP 侧 `submit()` 依次经 `pre_validate`（任务层钩子，如账本：被拒绝的动作也是
计费调用产生的，须先入账）、`validate`（词表 + 报告校验）、`check_fresh`（序号）。世界按墙钟推进（`RealtimeClock`，滞后超过 2 s 判为
基础设施失败），策略调用在 `DecisionWorker` 的工作线程里，仿真状态只在仿真线程改。

## 5. 公开地形与投影

`GridTerrain` 读 `f32-grid@1` 栅格（世界包的 `geometry/terrain/dtm_10m.json` 或同格式的派生层），双线性插值，越界与未知高程抛
`ValueError`（文案面向智能体）。`camera_metadata` 把 L2 相机帧的公开标定（按交付尺寸缩放的内参、畸变、名义安装、云台读数、采集时
与交付时的导航解）交给智能体；`project_pixel` 以这些读数与公开地形把像素投到地面，结果是估计值，适合人的脚点与车辆接地点，不查
演员、不查动态深度。

## 6. CHIMERA 的切换

CHIMERA 分支 `anet4d-interface`：`chimera/commander/skills.py`、`realtime.py`、`chimera/sandbox/projection.py`、`telemetry.py` 改为从
Anet4D 导入；`commander/terrain.py` 的 `TerrainMap` 继承 `GridTerrain`（保留派生 2 m 层优先）；`sandbox/bridge.py` 的 `DecisionBridge`
继承网关的桥，账本放在 `pre_validate`，报告校验经 `reports=` 注入；`sandbox/server.py` 的 `Gateway` 继承网关基类，只剩会话准入、运行
与评分；`commander/harness.py` 的逐机阶段与逐 tick 动作改用 `SkillFleet`；`drone4d.agent` 改为 `anet4d.agent`。CHIMERA 自己的测试
在 Anet4D 开发检出上通过，等价见证见 §7。

CHIMERA 在 dmax 上运行的是钉住的上游 `705a1be` 加 4 个本地补丁；其中 3 个（`realistic-civil-world`、`gimbal-noise-units`、
`vegetation-layer`）在当前 `main` 上不再适用。CHIMERA 真正切到新接口运行，须先把这些补丁重做或合入上游并升级钉版，登记为 F-153。

## 7. 测试与验证

- `tests/d3/skills/`：词表与契约一致性（有效指令两边都通过，无效指令两边都拒绝）、执行器（贴地形、rtl 下限、standoff、orbit 相位，
  与研究者 SDK 的动作构造对拍）、机队阶段机（起飞、待执行技能、rtl 结束、电量耗尽、拒绝文案）、网关（桥的恰好一次与取消、HTTP
  鉴权 / 404 / 400 / 409、账本钩子次序、墙钟与工作线程、投影）。
- `tests/d3/env/test_skill_fleet_env.py`：真实 `BenchEnv`（合成世界、L1、fast、锁步）上 goto 起飞、贴地形到点、rtl 落地结束，
  无被守卫拒绝的动作。
- CHIMERA 等价见证：同一脚本计划、同一种子，切换前后 `telemetry.jsonl`、`reports.jsonl`、`dialogue.jsonl`、`done.json` 逐字节相同
  （见 `docs/impl/skills-gateway-v1.zh-CN.md`）。
