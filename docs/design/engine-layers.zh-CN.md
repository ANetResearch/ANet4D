# ANet4D 引擎分层方案：高斯场景引擎、目标行为引擎与具身设备引擎

| 项 | 内容 |
|---|---|
| 文档 | 设计提案（design proposal），不是 ADR，不定义契约字段；需要负责人裁定的事项集中在 §9，待以 AWR-05 或 AWR-06 的 ADR 登记后生效 |
| 版本 | v0.1 草案（2026-11-02） |
| 作者 | houqiii（CHIMERA 基准与演示场景的维护者） |
| 输入 | AWR-05（ADR-132、137、138、139、140、156、160、186 至 192、196 至 199）；AWR-06 v0.4（§1.8、§3.2、§3.3）；16 World 数据规范 §2.2、§6.4；[场景适配目标引擎](scene-target-engine.zh-CN.md)；CHIMERA 演示场景的代码（仓库 `houqiii/chimera`，提交 `b5bede3`，2026-11-02）；[r05 开源生态调研](../research/d4/r05-三维高斯与具身仿真开源生态.md) |
| 适用 | D4 实现阶段（D3 封板之后）；本文的工作包以"D3 封板后第 N 周"之外的独立序列编号，纳入 AWR-06 §10 的排期由负责人决定 |
| 不做 | 不修改已冻结的 D3 契约；不改变 `drone4d-event@1` 的排名规则与外观来源规则（ADR-187、ADR-188）；不新增 D4 内核（M28）的第三方依赖 |

---

## 0. 结论（先读）

1. **CHIMERA 演示场景在 ANet4D 之外新增了三类通用引擎能力**：约 3.5 万行 Python 的目标行为引擎（居民、车辆、活动、停车与路缘、SUMO 与 JuPedSim 后端）、一套实时世界的外部指挥接口（封闭技能词表、决策桥、实时网关）与一套动态实体的外观层（骨骼角色、接触阴影、车辆贴片）。其中目标行为引擎相对 ANet4D 现有的 M18 识别物运行时与 m23 场景目标基础（AWR-06 §1.8）是数量级的扩展，应当首先回收进 ANet4D。
2. **现有场景表示（GLB 三角网格）只能是过渡形态。** 三份演示世界都是带烘焙光照的摄影测量网格（r2），装载与渲染成本随三角面数增长，不是 3DGS，也不能从航拍直接得到。ANet4D 已经为 3DGS 登记了格式（`spz@2`、`sog@1`、`sog-lod@1`）、工具链（splat-transform、SuperSplat、Spark，ADR-192）与决策门（G-3DGS，ADR-186 至 188），但全部放在 P3 且只作为**外观层**。本文建议把高斯表示提升为**场景表示**：新世界以 ply 或 splat 入库，几何层（支撑面、占据、距离场、表面网格、导航网格）从高斯派生，GLB 世界继续由同一派生接口服务。
3. **分层的核心是一条契约，而不是三个包。** 场景引擎与目标行为引擎之间只允许经过"场景可供性契约"（Scene Affordance Contract，暂名 `ct:scene/affordance@1`）通信：道路网、可走面、活动区域、设施锚点与容量、停靠位、支撑高度查询。行为引擎不读取高斯、网格或点云；场景表示从网格换成高斯时，行为引擎不改一行代码。CHIMERA 的实测支撑、路缘与停车普查正是这条契约的第一批实现。
4. **目标行为引擎的事实来源仍是 Canonical Event Tape。** CHIMERA 的生活运行时以固定 0.1 s 步长整组推进、"运行时拥有每一个位姿"，输出与 D3 磁带兼容；移植后保持 ADR-135、ADR-199 的 Exogenous 与 Interactive 两种模式不变，m23 的 `py:sim/activities/` 作为轻量生成器与之并列，共用 M31 的 `Generator` 协议。
5. **具身设备引擎以现有的三档物理（ADR-137）、PoseHistory（ADR-160）、M25 传感器服务与 L0 至 L3 观测档为基础推广，不重写。** 推广的方向是：`DroneAdapter` 推广为按机体类别参数化的 `BodyAdapter`（多旋翼、固定翼、地面车、水面艇），云台作为带自身保真度档的子设备，指令按四个层级（执行器、速度与偏航率、技能、自然语言）分层，CHIMERA 临时的运动学无人机与封闭技能词表只保留词表部分。
6. **一套这样的基础设施还缺的能力**（§7）：高斯资产的构建与出处流水线（部分已有）、面向大模型指挥者的实时网关与技能接口（CHIMERA 已有雏形）、场景准入诊断（m23 的 scene-audit 与 CHIMERA 的 envpack 合并）、地面与水面机体、硬件桥（MAVLink、ArduPilot、云台协议）、动态实体的高斯外观（研究项）、批量向量化推进（研究项）。
7. **开源选型原则**：成熟、许可清楚、可锁定版本的组件直接采用（gsplat、splat-transform、Spark、Recast/Detour、SUMO、JuPedSim、PX4 SITL 与 gz-sim）；论文级方法（从高斯提取表面、从高斯仿真 LiDAR、高斯动态场景）作为可替换的后端做评估，不进入正式成绩，直到通过本文 §4.2 的几何门。
8. **实施顺序**：第一批移植目标行为引擎并让它在现有 GLB 世界上运行（§8 WP-E2-01 至 03），这一批不依赖高斯；第二批建立高斯世界包与派生几何层（WP-E1-01 至 03）；第三批高斯相机与网页共渲染（WP-E1-04、05）；第四批设备引擎推广与新机体（WP-E3-01 至 04）。粗估合计约 52 人周（±40%），细目见 §8。
9. **需要负责人裁定的事项**有 12 条（§9），其中 D-01（高斯几何层可作为世界真值的条件）与 D-03（行为引擎的包位置与 m23 的关系）影响后续全部工作包，建议先裁。
10. 本文的全部代码陈述以 ANet4D `c79a3e0`（V0.3.4，2026-11-02）与 CHIMERA `b5bede3` 为准；开源项目的核实状态以 r05 标注为准，r05 标为 C（待核实）的条目在本文只作为候选，不作为依据。

---

## 1. 文档定位

### 1.1 写什么

- ANet4D 世界与物理层（AWR-06 §3.2 的 L0 与 `hosts/hifi`）的内部分层：场景引擎、目标行为引擎、具身设备引擎，以及三者之间的接口。
- CHIMERA 演示场景中通用引擎性质的代码回收进 ANet4D 的映射、改造要求与顺序。
- 从 GLB 网格场景迁移到 3DGS 场景的技术路线与门限。
- 作为具身协同基础设施还应提供的能力清单，以及每项能力的开源依托。

### 1.2 不写什么

- 不定义任何契约的字段与 schema（§3.4 只给出契约的职责与消费者，字段在立项后的模块 PRD 中定义）。
- 不改变 D4 实验语义层（M28 至 M34）的设计；本文全部内容位于 L0、`hosts/hifi` 与平台基础设施，与 AWR-06 §3.5"不进入实验语义"的边界一致。
- 不评估论文方法的具体精度（r05 只登记核实状态与可取之处；精度以本项目复现为准）。

### 1.3 与现有文档的关系

| 现有文档 | 本文的处理 |
|---|---|
| AWR-05 ADR-186 至 ADR-190（高斯外观层维持 P3，排名只接受 `appearance = mesh`） | 不修改。本文把高斯作为**几何来源**的用途与作为**外观来源**的用途分开：外观来源的分档与排名约束照旧，几何用途另设几何门（§4.2.4，D-01） |
| AWR-05 ADR-192（splat-transform、SuperSplat、Spark 随外观层移至 P3） | 沿用其版本锁定、格式注册与出处清单；本文把高斯入库提前到 D4 的 WP-E1-01，是否与 WP-D3-39 合并由负责人决定（D-09） |
| AWR-05 ADR-196、ADR-197、ADR-199（Recast/Detour、Python 生成器、局部偏离模型） | 不修改。高斯派生的表面网格作为 Recast 烘焙的输入之一；navsvc 不变 |
| AWR-06 §1.8（m23 与 D4 模块的映射） | 扩展：CHIMERA 生活引擎作为 M31 生成器的第二个实现，与 `py:sim/activities/` 并列 |
| 16 World 数据规范 §6.4（碰撞代理 L1 至 L3 占位） | 本文 §4.2 给出从高斯派生这三级代理的方法，作为 §6.4 定稿的输入 |
| [场景适配目标引擎](scene-target-engine.zh-CN.md) | 其"四批交付"中的第 2 批（场景适配与精细资产）与第 3 批（seed 活动引擎）由本文 §4.3 与 §5 承接并扩大范围 |

---

## 2. 现状盘点

### 2.1 ANet4D 现有能力按层

| 层 | 现有能力 | 代码或文档 | 评价 |
|---|---|---|---|
| 场景表示 | World Package v1：Potree 点云、DTM/DSM/HAG 栅格、语义区域、锚点；D3 的 GLB 世界（`appearance.glb`、raycast 网格、米制与配准）；`MeshWorldQuery`（每瓦片 Open3D `RaycastingScene`） | 16 §3 至 §7；M22；`py:world/{glb,mesh,package,terrain,semantic}` | 完整，但以网格为中心；`walkable_height`、`nav_path` 未实现（scene-target-engine §目标与现状） |
| 碰撞代理 | L0 DSM/DTM；L1 体素占据、L2 SDF 为 V0.3 占位；L3 网格 BVH 运行时生成 | 16 §6.4 | L1、L2 未交付 |
| 导航 | Recast/Detour 烘焙（recast-navigation 0.43.1，Node 子进程）、navsvc 查询（吸附、寻路、视线、高度、随机点）；`detour-navmesh@1`、`geojson/awr-navgraph@1`；栅格 A* 过渡实现 | ADR-196；`tools/navsvc/`；`py:sim/events/nav.py` | 正式口径已定；车辆路网只有 OSM 先验，无车道方向、路权、停靠位 |
| 事件与目标 | Canonical Event Tape（生成器 Python，TS 解码对拍）；M18 识别物运行时（10 Hz stage、运动、声源、判据、状态机）；m23 场景目标基础（场景准入审计、生活活动编译器、群体推进、DeepSeek 设计助手） | ADR-197；`py:sim/events/`、`py:sim/targets/`、`py:sim/activities/` | 磁带与运行时成熟；生活活动是种子级（固定模板、无交通、无停车与路缘） |
| 物理 | 三档：Fast（FleetSim）、Realistic（gz-sim 锁步，正式成绩唯一来源）、SITL（PX4 + gz，`ranked = false`）；统一 `DroneAdapter`，指令语义为世界系速度 + 偏航角速度；PoseHistory 亚 tick 读取 | ADR-137、138、160；`py:phys/{adapter,fast,kinematic,gz,pose_history}` | 成熟，但只有多旋翼（固定翼有 `awr_f1` 机型包） |
| 传感器 | M25 纯函数服务：EGL 光栅相机（`rgb1080-rs`、`rgb720-gs`、深度、实例）、网格 LiDAR（逐点时刻求交）、热成像、IMU、导航解与云台编码器噪声（随机流 22） | ADR-139、140、198；`py:sensorsvc/` | 成熟；相机只支持网格场景 |
| 观测档 | L0 Oracle、L1 检测（`mesh_los`，5 Hz）、L2 渲染（含 L2-depth 子档）、L3 SITL/HIL | M26 | 成熟 |
| 机型与传感器目录 | `vehicles/{awr_f1, awr_q1, awr_q9, awr_v1, p600, x500, d3}`、7 个传感器模型、`catalog.json` | 20 | 只有飞行器 |
| 高斯 | 格式 `spz@2`（预留）、`sog@1`、`sog-lod@1`、`ssproj@1`、`svo-voxel@1.1`；r11 Skyfall-GS 评估（结论 C0）；r12 SuperSplat 与 splat-transform 实测；P3 工作包 WP-D3-22、39 至 42 | ADR-186 至 192；r11、r12 | 全部在 P3，只作外观层 |
| D4 内核 | 轻量 `py:kernel/`（numpy 闭包）、PlanarWorld、`hosts/hifi` 的 MeshWorld（P1 可选） | AWR-06 §3.2、§3.3 | 概念设计；`hosts/hifi` 是本文三个引擎的挂载点 |

### 2.2 CHIMERA 演示场景新增的通用能力

CHIMERA（基准）在 ANet4D 之上构建了一个 30 ha 的小镇演示场景：150 名居民、31 辆车、3 架无人机、7 类任务，公开站点 research.agentnetwork.org.cn/chimera/。其代码按通用程度分三类（规模与依赖按附录 A 的扫描结果）。

| 包 | 规模 | 内容 | 通用价值 | 与 `awr` 的耦合 |
|---|---|---|---|---|
| `chimera.civil` | 103 文件，20.9 k 行 | 场景语义（车道网、路廊、路缘、停车场与规则、公共区域、户外场所、实测支撑）、人口与身份、活动需求与日程（`activity/` 19 文件 4.9 k 行：导演、配方、驾驶适配、停车准入、出行证据）、一次性构建流水线（SUMO net → 语义 → 人口 → 日程 → 仿真 → QA → 组装，阶段缓存） | 目标行为引擎的主体 | `awr.sim.events`（磁带）11 处、`awr.world.mesh` 3 处 |
| `chimera.life` | 65 文件，10.1 k 行 | `life_v1` 运行时：0.1 s 整组推进，决策、上下车、驾驶（出行准入与路廊预留、实时停车前瞻、Reeds-Shepp 机动）、步行（静态路由、让行、拥堵改道、礼让）、人群、记录、失败证据、生活审计 | 目标行为引擎的推进内核 | `awr.sensorsvc.camera` 1 处 |
| `chimera.ambient` | 13 文件，4.3 k 行 | 路线 A 后端：SUMO 车辆仿真于实测车辆支撑、JuPedSim 行人仿真于实测可走地面、路口耦合、转磁带；本身无运动模型 | 行为引擎的外部求解器后端 | `awr.sim.events` 1 处 |
| `chimera.envpack` | 9 文件，1.7 k 行 | 静态环境诊断（几何、语义、连通性、来源版本、历史测量），面向特性准入契约 | 场景准入诊断 | `awr.sim.events` 1 处 |
| `chimera.render` | 35 文件，3.6 k 行 | 骨骼角色（ozz 动画、地形 IK、脚步落点）、接触阴影、车辆贴片与透明度、L2 相机的动态实体合成 | 动态实体外观层 | `awr.sensorsvc.camera` 6 处、`awr.sim.events` 4 处 |
| `chimera.commander`、`chimera.sandbox` | 17 文件，2.2 k 行 | 封闭技能词表、锁步调用大模型、实时世界网关与决策桥、L2 光学与投影、遥测 | 大模型指挥接口 | `awr.agentsdk.env`、`awr.phys.adapter`、`awr.sensorsvc.camera` |
| `chimera.scoring`、`chimera.evalv3`、`chimera.checkpoints` 等 | 约 2.5 k 行 | 确定性检查点评测、见证门、记账 | 基准专有，不回收（评测内核归 M32） | 无 |

CHIMERA 的无人机是运行时的运动学机体：世界系速度指令、瞬时云台指向、无动力学与接触，由技能词表展开为逐 tick 的 `nav` 与 `gimbal` 动作（`chimera.commander.harness`）。它是演示的临时实现，粒度低于 ANet4D 的三档物理，不回收；回收的是技能词表与实时网关（§6.2、§7）。

### 2.3 差距

1. 场景表示绑定三角网格：装载成本随面数增长；从航拍得到网格需要摄影测量与减面；高斯表示的几何用途没有定义。
2. 碰撞代理 L1、L2 未交付，`walkable_height` 与 `nav_path` 未实现；车辆路网没有车道方向、路权、路缘与停靠位，行为引擎在 ANet4D 内无法承载真实交通。
3. 生活活动只有种子级；CHIMERA 的完整引擎在 ANet4D 之外，依赖自己的世界读取方式（实测支撑、导航栅格）。
4. 机体只有飞行器；云台没有独立的保真度档；指令层级只有速度 + 偏航率一档与若干 `CmdKind`。
5. 面向大模型指挥者的接口（技能词表、实时世界、决策桥）在 ANet4D 中没有对应物。

---

## 3. 目标分层

### 3.1 分层图

下图是 AWR-06 §3.2 中 L0 与 `hosts/hifi` 的内部展开；L1 至 L7 不变。

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 上层：AWR-06 L1 至 L7（内核、交互网络、协同算法、Benchmark、评测、榜单、SDK）                        │
│ 以及 D3 的 M26 Env 与 M27 Benchmark Runtime                                                          │
├──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ X 横切能力：指挥接口与实时网关（技能词表、决策桥）、资产与出处流水线、                               │
│ 场景准入诊断、Trace 与回放（M33、MCAP、Rerun）、场景创作助手                                         │
├────────────────────────────┬────────────────────────────────────┬────────────────────────────────────┤
│ E3 具身设备引擎            │ E2 目标行为引擎                    │ E1 场景引擎                        │
│ BodyAdapter（多旋翼、      │ 场景准入 → 语义 → 人口与日程       │ 高斯世界包（ply、splat、spz、sog） │
│ 固定翼、地面车、水面艇）   │ → 运动求解（native 或              │ 派生几何层（支撑面、占据、         │
│ 云台子设备（三档）         │ SUMO + JuPedSim）→ 磁带组装        │ ESDF、表面网格、NavMesh）          │
│ 三档物理、PoseHistory      │ 与 QA → Canonical Event Tape       │ 语义与可供性层                     │
│ M25 传感器                 │ Exogenous 与 Interactive 模式      │ 高斯相机与 LiDAR 后端、网页共渲染  │
│ 指令四层级、硬件桥         │                                    │ GLB 世界经同一接口服务             │
├────────────────────────────┴────────────────────────────────────┴────────────────────────────────────┤
│ 契约：ct:scene/affordance（E1 → E2）、磁带 TapeReader（E2 → E3、M25、M26）、                         │
│ PoseHistory 与 TierCommand（E3 与 M25、M26）、技能词表（X → E3）                                     │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

依赖方向：E2 依赖 E1 的可供性契约，不依赖 E1 的表示；E3 依赖 E1 的几何查询（净空、碰撞、支撑）与 E2 的磁带（动态实体的位姿与代理体），不依赖 E2 的内部；M25 同时读取 E1（静态场景）、E2（磁带）与 E3（PoseHistory）。

### 3.2 不变量

| 编号 | 不变量 | 来源 |
|---|---|---|
| I-E-01 | 行为引擎只经可供性契约读取场景；契约之外的场景数据（高斯、网格、点云、栅格）对行为引擎不可见 | 本文 |
| I-E-02 | 磁带是动态实体的唯一真值；渲染器、传感器与设备引擎从不推进演员时钟 | ADR-135；CHIMERA `chimera.life` 设计 |
| I-E-03 | 同一输入（世界包 `contentVersion`、配方、seed、锁定版本）重编译的磁带 `content_sha256` 相同，跨 AVX2 与 AVX-512 一致 | ADR-197 |
| I-E-04 | 场景表示（网格或高斯）的切换不改变可供性契约的字段，也不改变磁带格式 | 本文 |
| I-E-05 | 正式成绩的外观来源与排名约束按 ADR-187、ADR-188；高斯作为几何来源另设几何门 | 本文 D-01 |
| I-E-06 | 三档物理共用 `TierCommand`、`TickResult`、`VehicleStatus` 与 PoseHistory；新增机体类别不新增第四档 | ADR-137 |
| I-E-07 | D4 内核的 numpy-only 闭包不受本文影响：E1 至 E3 全部位于仓库环境或 `hosts/hifi` | AWR-06 §3.3 |

### 3.3 暂定模块号与包位置

| 暂定 | 名称 | 包路径（`python/awr/` 下） | 来源 |
|---|---|---|---|
| M35 | 高斯场景引擎 | `world/gaussian/`（入库、变换、编码）、`world/derive/`（派生几何层与可供性）、`sensorsvc/camera/gs/`（高斯相机后端） | ADR-192 的工具链；本文 §4 |
| M36 | 目标行为引擎 | `life/`（`scene`、`semantics`、`population`、`activity`、`motion`、`backends`、`tape`、`audit`） | `chimera.civil`、`chimera.life`、`chimera.ambient`、`chimera.envpack` |
| M37 | 具身设备引擎 | `phys/`（现有）扩展 `BodyAdapter`、`gimbal/`；`vehicles/` 新增类别 | ADR-137；本文 §6 |
| M38 | 指挥接口与实时网关 | `commander/`（技能词表、线束）、`runtime/gateway/`（实时网关、决策桥） | `chimera.commander`、`chimera.sandbox` |
| 契约 | `ct:scene/`、`ct:device/` | `packages/contracts/scene/`、`packages/contracts/device/` | 本文 §3.4 |

模块号与包名为暂定（D-09）；代码位置遵守 CONTRIBUTING 的路径所有权，新包的所有者在立项时登记。

### 3.4 契约职责（不定义字段）

| 契约（暂名） | 生产者 | 消费者 | 内容职责 |
|---|---|---|---|
| `ct:scene/affordance@1` | E1 `world/derive` | E2、场景准入诊断、Lab 页面 | 道路网（车道、方向、连接、路权类别、限速）、可走面（多边形与高度）、活动区域（类别、容量、开放时间）、设施锚点（入口、停靠位、泊位、容量）、支撑高度查询的离线产物、每项的来源（measured、osm、inferred）与确认状态；复用 `awr.anchors.v1` 与 `awr-zones@1` 的既有语义，不另起坐标与 POI 格式 |
| `ct:scene/geometry@1` | E1 | E3、M25、navsvc 烘焙 | 支撑面栅格（沿用 `f32-grid@1`）、占据体素与 ESDF（16 §6.4 的 L1、L2 定稿）、表面网格（Recast 输入与 LiDAR 求交）、每层的派生参数与质量指标 |
| `ct:scene/gaussian@1` | M22 构建 | M25、网页 | 高斯图层清单：源格式、splat-transform 版本、`fillRatio` 与 NaN 门限结果、坐标换算、瓦片与 LOD、编码（`spz@2`、`sog@1`）、出处 `pipeline`、`stats`、`edits`（ADR-192 决策 (1)） |
| `ct:device/body@1` | E3 | M25、M26、M30 适配器 | 机体类别、自由度、执行器限制、档位能力（kinematic、fast、realistic、sitl）、子设备（云台、传感器挂载）与其档位 |
| `ct:device/skills@1` | X 指挥接口 | M26 Env、M30 `adapters.drone_agent`、E3 | 封闭技能词表（名称、参数、前置条件、完成判据、展开规则），来源为 CHIMERA 的词表 |

---

## 4. E1 高斯场景引擎

### 4.1 数据与入库

- **输入格式**：3DGS 的 `ply`（原始训练产物，含球谐系数）、2DGS 的 `ply`（面元，带法向，几何更准，见 r05 §1.3）、`splat`（社区二进制）、`spz`、`sog`。入库统一为 ADR-192 的流程：锁定版本的 splat-transform 做体检（`fillRatio`、NaN）、清理、坐标变换与回读校验、瓦片、服务端 `spz@2`、网页流式 `sog@1`。
- **世界包图层**：新增 `gaussian` 图层，与 `appearance.glb` 并列；`world.json.layers[].format` 登记 `3dgs-ply@1`、`2dgs-ply@1`（新增，待 16 §2.2 登记）与既有的 `spz@2`、`sog@1`。一个世界可同时含网格与高斯；`contentVersion` 覆盖两者。
- **来源分档**：沿用 ADR-187 的 `appearance_source ∈ {aerial, mesh_distilled, generative}`；几何用途另记 `geometry_source ∈ {mesh, gaussian, gaussian_mesh}`（§4.2.4）。
- **配准**：高斯与世界坐标系的配准沿用 M22 的强制配准（OSM 先验、控制点、`coordinate.json`），配准残差写入清单；ADR-189 的公开参照集用于配准与几何核对。
- **编辑**：SuperSplat v3 为内部编辑工具（ADR-192 决策 (2)），编辑产生新的图层版本，不覆盖。

### 4.2 派生几何层与自适应碰撞边界

高斯本身没有表面，碰撞边界必须派生。本文定义四级派生产物，对应 16 §6.4 的 L0 至 L3，并给出"自适应"的含义：每一级的阈值不是全局常量，而是按语义类别、局部密度与实测支撑校准，并以质量指标门控。

| 级 | 产物 | 派生方法 | 自适应项 | 质量指标 |
|---|---|---|---|---|
| L0 支撑面 | DTM、DSM、HAG（`f32-grid@1`） | 自上而下正交渲染高斯的期望深度（gsplat 深度模式），按格网取中位数；地面类别用低分位数去除浮点 | 分位数与核宽按局部高斯密度（每格的 `fillRatio`）选取；与 CHIMERA `measured_support`（无人机实测支撑）对齐校准 | 与 LiDAR 或摄影测量参照的高度残差（中位数、p95）；地面连通分量数 |
| L1 占据 | 体素占据（`geometry/voxel/`） | 对每个体素积分高斯不透明度（NavGSim、Splat-Nav 的切片与积分思路，r05 §2），阈值化 | 阈值按类别（建筑、植被、细杆件）分别校准：植被取高阈值避免把树冠当实体，杆件取低阈值并做形态学闭运算 | 与网格 BVH 的射线命中一致率；细结构（杆、线、栏杆）的召回 |
| L2 距离场 | ESDF（`geometry/sdf/`，float16） | 由 L1 占据计算，或直接以高斯中心与协方差求近似距离（Splat-Nav 的椭球距离） | 距离截断与分辨率按机体尺寸（E3 的 `ct:device/body@1`）选取 | 净空查询与网格 BVH 的差值分布 |
| L3 表面网格 | 三角网格（Recast 输入、LiDAR 求交、阴影投射） | 从 2DGS 或表面对齐方法提取（2DGS、SuGaR、PGSR、MILo、GOF，r05 §1.3），再减面与分瓦片 | 提取方法按场景类型选：航拍大场景用 2DGS 或 PGSR，近景用 SuGaR 细化 | 与 L1 占据的一致性；可走面积与 OSM 的差；路廊连续性；路缘高度分布 |

#### 4.2.1 自适应的定义

"自适应"指三件事：(1) 阈值与核参数按语义类别和局部密度选取，不用全局常量；(2) 以实测数据校准：CHIMERA 的实测支撑（无人机实测的地面与路面）、ADR-189 的公开参照集的 LiDAR；(3) 派生结果以 QA 门控，不通过的图层不得声明为就绪（沿用 scene-target-engine §第一批行为定义的原则：缺数据、未确认不得显示为已就绪）。

#### 4.2.2 与网格世界的统一

GLB 世界经同一 `world/derive` 接口产出同样的四级产物（L0 由 raycast 网格自上而下求交，L1、L2 由网格体素化，L3 即网格本身）。上层只见 `ct:scene/geometry@1`，不见来源。这使 WP-E2 的行为引擎移植不等待高斯。

#### 4.2.3 运行时查询

- 批量净空、碰撞与支撑查询（E3 与 M25 使用）：优先 L2 ESDF（O(1) 查表），其次 L3 BVH（Open3D `RaycastingScene`，现有）。
- LiDAR：P-first 以 L3 网格求交（现有 MeshLidar，逐点时刻，ADR-198）；直接对高斯做射线求交（LiDAR-GS、SplatAD 的方法，r05 §3）作为可替换后端评估，差值以 L3 为参照。
- 动态实体：沿用 ADR-198 的实体局部代理网格；动态实体的高斯外观是研究项（§7 第 10 条）。

#### 4.2.4 几何门 G-GEO（D-01）

高斯派生几何作为世界真值（用于碰撞、LiDAR、L1 检测的遮挡）须通过几何门：

| 指标 | 门限（建议，立项后以参照集定稿） |
|---|---|
| G-GEO-1 支撑面残差 | 相对 LiDAR 参照中位数 ≤ 0.10 m，p95 ≤ 0.30 m（地面类别） |
| G-GEO-2 占据一致率 | 与同场景网格 BVH 的射线命中一致率 ≥ 97%（随机 10 万射线，含低空斜视） |
| G-GEO-3 细结构召回 | 人工标注的杆件、线缆、栏杆样本召回 ≥ 90% |
| G-GEO-4 可走面与路廊 | 可走面积与 OSM 道路面的对称差 ≤ 8%；路廊连续、无断裂 |
| G-GEO-5 确定性 | 同输入重派生逐字节一致 |

门限通过的世界记 `geometry_source = gaussian`；未通过但有网格的世界记 `mesh` 并照旧运行；两者都不通过的世界不得进入正式成绩。外观门 G-3DGS（ADR-186 至 188）独立于几何门，互不替代。

### 4.3 语义与可供性层

可供性层是 E1 到 E2 的唯一接口，内容来自三类来源并记来源与确认状态：

1. **测得**（measured）：CHIMERA 的实测支撑、路缘普查（`curb_survey`）、停车普查（`parking_survey`）、道路服务几何（`road_service_geometry`）、公共区域与户外场所，经本文移植为 `world/derive/affordance/`。这些模块当前读取 CHIMERA 自己的导航栅格与网格；移植时改读 `ct:scene/geometry@1`。
2. **先验**（osm）：OSM 车道图、建筑用途、设施 POI（M22 既有）。
3. **推断**（inferred）：从几何与先验推断的车道方向、路口连接、停靠位、入口（`lane_registration`、`junction_access`、`berth_guides`）。

可供性层的"原则性边界"由规则派生并可审计：道路边界 = 路面类别的支撑面 ∩ 路廊宽度约束；可走面 = 支撑面中坡度与台阶满足行人 profile 的区域 − 车道（JuPedSim 的几何约束）；活动范围 = 区域多边形 ∩ 可走面 ∩ 开放时间；停靠位 = 路缘段中满足长度、坡度、无障碍与规则的区间。每条规则的参数（行人 profile、车辆尺寸族、最小路廊宽度）取自 `ct:device/body@1` 与行为引擎的实体尺寸包络，不硬编码。

### 4.4 渲染

| 用途 | 后端 | 状态 |
|---|---|---|
| 服务端 L2 相机（无人机视角） | M25 新增 `GsCameraBackend`：gsplat（CUDA）光栅化静态高斯，输出 RGB 与期望深度；动态实体（E2 磁带的角色与车辆）以现有 EGL 光栅化绘制并按深度合成（RL-GSBridge、SplatAD 的网格与高斯混合思路，r05 §2、§3）；效果链（畸变、卷帘、运动模糊、噪声）沿用 ADR-140 §5.4.3 | ADR-188 已命名 `GsCameraBackend`，在 P3；本文提前到 WP-E1-04 |
| 服务端深度与实例 | 深度来自高斯期望深度与实体光栅化的合成；实例与语义掩膜来自实体光栅化与高斯的类别标注（入库时按 L1 占据的类别传播） | 新 |
| 网页主视口 | three.js 视口 + Spark 共渲染（SOG 流式 LOD），事件编辑器的选择交互仍基于网格（ADR-192 决策 (3)） | WP-D3-41 |
| 大场景 | Spark 的分页 LoD；城市级（十亿高斯）参考 Aholo Viewer 的分块 LOD 流式（r05 §1.4，B 级核实） | 评估 |
| 对拍 | splat-transform 光栅化器（慢一个数量级）只作离线对拍；nvdiffrast 不进入产品（ADR-140） | 沿用 |

### 4.5 从网格到高斯的迁移

1. 现有三份 GLB 世界不重建；它们经 §4.2.2 的统一接口继续服务，并作为 G-GEO 的网格参照。
2. 新世界优先以航拍影像训练 3DGS 或 2DGS 入库（r05 §1.2 的航拍大场景方法为候选），同批影像产出网格作对照（ADR-189 决策 (2) 的试点口径）。
3. 两种表示并存期间，Run 规格的 `appearance` 与 `geometry_source` 分别记录；榜单版本登记允许的组合。
4. 动态实体的高斯外观（4DGS 系，r05 §1.5）不在本文的工作包内。

### 4.6 开源选型

| 组件 | 用途 | 核实状态（r05） | 处置 |
|---|---|---|---|
| gsplat | 服务端光栅化（r11 实测 720p 291 fps） | A | 采用，锁版本 |
| splat-transform、SuperSplat、SuperSplat Viewer | 入库、编辑、检视 | A（r12） | 采用（ADR-192） |
| Spark | 网页共渲染、SPZ、分页 LoD | B | 采用，先做许可与 three.js 版本核对 |
| 2DGS、SuGaR、PGSR、MILo、GOF、2D-SuGaR | L3 表面网格提取 | B | 评估后选一主一备 |
| NavGSim、Splat-Nav、SplatGym | L1、L2 派生与导航可行域的方法参考 | B | 方法参考，不作依赖 |
| LiDAR-GS、GS-LiDAR、SplatAD | 直接高斯 LiDAR | B | 作为可替换后端评估 |
| Aholo Viewer | 城市级 LOD 流式 | B | 评估 |
| CityGaussian、VastGaussian、Horizon-GS | 航拍大场景训练 | C | 候选，未核实，不作依据 |

---

## 5. E2 目标行为引擎

### 5.1 定位与边界

目标行为引擎生成并推进场景中的非被测实体（居民、车辆、街道对象、事件演员），输出 Canonical Event Tape。它是 M31 生成器的一种（AWR-06 §1.8：`exogenous_path` 的来源），不是被测 Agent，不进入 D4 实验语义；其内部随机量只来自登记的随机域（m23 的流 24 `target_activity`，移植后按子域登记）。

边界：

- 读场景只经 `ct:scene/affordance@1` 与 `ct:scene/geometry@1` 的支撑高度（I-E-01）。
- 写只有磁带与审计产物；不写世界包。
- 运行形态：离线编译（Exogenous，缺省）；Interactive 模式沿用 ADR-199 的局部偏离模型，不在 sim-core 内嵌人群仿真。
- 实时演示（CHIMERA 的实时沙盒）是平台基础设施的消费者，不改变引擎语义。

### 5.2 流水线

沿用 CHIMERA `chimera.civil.build` 的阶段划分，每阶段产物内容寻址、缓存、可单独重跑：

| 阶段 | 输入 | 输出 | 来源模块 |
|---|---|---|---|
| 1 场景准入 | 世界包、可供性层 | 准入报告（阻塞、待验证、就绪），与 m23 `scene-audit` 合并 | `chimera.envpack`、`chimera.civil.grounded`、`py:sim/events/scene.py` |
| 2 语义 | 可供性层、OSM | 车道网与路廊、路缘与停靠位、停车场与规则、公共区域、设施容量；路线 A 时另产 SUMO net 与 JuPedSim 几何 | `chimera.civil.{lane_network, road_corridors, curb, parking_*, public_regions, outdoor_places}`、`chimera.ambient.{roads, walkway}` |
| 3 人口与身份 | 配方、seed、资产目录 | 固定身份、外观变体、住所与设施槽位、车辆所有权与泊位 | `chimera.civil.{population, body, appearance_variant}` |
| 4 活动与日程 | 人口、设施、开放时间 | 每人的日程（`LifeAgenda`：步行与驾车出行）、活动需求、出行证据 | `chimera.civil.activity.{director, recipes, driving_adapter, opportunities}` |
| 5 运动求解 | 日程、语义 | 逐 0.1 s 的位姿、状态、容器（上下车）、动作 | native：`chimera.life.engine`（步行、驾驶、人群、礼让、改道）；外部：`chimera.ambient`（SUMO + JuPedSim + 路口耦合） |
| 6 QA 与审计 | 位姿序列 | 早期运动检查、生活审计（进度、等待原因、空间跨度、边界进出、碰撞、身份连续性）、失败证据 | `chimera.civil.{qa, life_audit, journey_audit, fleet_audit}`、`chimera.life.failure_evidence` |
| 7 组装 | 位姿序列、身份、动作 | 磁带（`awr.sim.events` 写入器）、facts、出处 | `chimera.civil.assemble`、`chimera.civil.provenance` |

### 5.3 协议

- `BehaviorBackend`：`native`（移植的 `life_v1`）、`sumo_jupedsim`（路线 A）、`detour_crowd`（未来，仅离线）。后端只负责第 5 阶段；第 1 至 4、6、7 阶段共用。
- `Generator`（M31 §7.2）：行为引擎与 m23 `py:sim/activities/` 都实现它；M31 实例包记录生成器 id、版本与配方哈希。
- 确定性：固定步长、seeded、禁用 numba `fastmath`（ADR-197）；SUMO 与 JuPedSim 后端的跨机器一致性须单独验证（§10 风险 R-03），不满足时该后端只能 `ranked = false`。
- 磁带格式不变；需要新增的字段（容器关系、动作标签、身份连续性）以磁带的扩展段承载，由负责人决定是否升级 `tape` 版本（D-04）。

### 5.4 从 CHIMERA 的移植映射

| CHIMERA | ANet4D 目标位置 | 改造要求 |
|---|---|---|
| `chimera.civil`（语义、人口、活动、构建、QA、组装） | `awr.life.{semantics, population, activity, pipeline, audit, tape}` | 场景读取改为可供性契约与几何契约；`awr.sim.events` 的 11 处调用保留；`awr.world.mesh` 的 3 处改为几何契约；移除对 `chimera.schema` 与 `chimera.render` 的引用（外观变体改为资产目录 id） |
| `chimera.life`（`life_v1` 运行时、引擎、路由、驾驶、人群、记录） | `awr.life.motion.native` | 对 `awr.sensorsvc.camera` 的 1 处引用移除（属于演示的可见性检查）；`bullet.py`（PyBullet 验证工具）归 `tools/life/`，不进入包 |
| `chimera.ambient` | `awr.life.backends.sumo_jupedsim` | SUMO、JuPedSim 为附加包 `anet4d[life-ext]` 的依赖，缺省不装；后端在解析阶段按可用性拒绝 |
| `chimera.envpack` | `awr.life.scene.admission` 与 m23 `scene-audit` 合并 | 报告格式合并为一份准入报告（退出码 0、2、3 沿用） |
| `chimera.render`（骨骼角色、接触阴影、贴片） | `awr.sensorsvc.camera.actors` | 作为 M25 动态实体绘制的一部分；资产（Rocketbox 角色）按 ADR-156 的资产目录与许可登记，不入仓库 |
| `chimera.civil.backends.native_*`（动作、运动、人群的原生实现） | `awr.life.motion.native` | 与 `chimera.life` 合并去重 |
| 脚本 `scripts/{basemap, route_b, cast, envpack, physics}` | `tools/life/` | 只保留构建流水线需要的；一次性实验脚本不移植 |

依赖：numpy、shapely、scipy、numba、PyYAML、rsplan（Reeds-Shepp）、py_trees 为仓库环境依赖；sumolib、libsumo、jupedsim、contourpy 为 `anet4d[life-ext]`。全部不进入 D4 内核闭包（I-E-07）。

### 5.5 与 m23、M18、M23 的关系

- m23 `py:sim/activities/` 保持为轻量生成器（固定模板、无交通），适合 S2 网格族的快速实例生成；完整引擎用于需要交通、停车与生活连续性的世界。两者共用资产目录 `ct:target/assets.catalog.json` 与随机域登记。
- M18 `TargetRuntime`（D2 交互沙盒的群体推进）不变；完整引擎的磁带可经 M18 的 `exogenous_path` 绑定进入 D2 沙盒展示。
- M23 事件运行时不变：完整引擎的磁带与脚本事件演员在同一磁带内，脚本演员优先（r6 §0 第 6 条："若改用人群模拟推进，事件根本不会发生"）；人群只作背景，脚本演员的路径由事件编译器生成，人群对脚本演员让行而不是相反。

### 5.6 开源后端

| 组件 | 用途 | 核实状态（r05） | 处置 |
|---|---|---|---|
| Eclipse SUMO（sumolib、libsumo） | 车辆微观仿真、路网编译 | A（CHIMERA 在用） | `anet4d[life-ext]` |
| JuPedSim | 行人仿真（几何约束、社会力） | A（CHIMERA 在用） | `anet4d[life-ext]` |
| SUMO 与 JuPedSim 官方耦合 | 路口与人车交互 | B | 评估替换 CHIMERA 的自实现耦合 |
| rsplan、py_trees | Reeds-Shepp 机动、行为树 | A | 仓库环境依赖 |
| Recast/Detour（navsvc） | 行人与车辆 NavMesh | A（ADR-196） | 沿用 |
| DetourCrowd | 在线人群 | A（ADR-199 否决于 sim-core） | 只在离线后端评估 |
| MATSim、CARLA scenario runner | 活动日程与交通剧本 | C | 候选，未核实 |

---

## 6. E3 具身设备引擎

### 6.1 保留与推广

保留：三档物理与 `ranked` 规则（ADR-137 v1.3）、`TierCommand`、`TickResult`、`VehicleStatus` 与接触阈值、PoseHistory 与 IMU 合成（ADR-160）、M25 传感器链（相机、LiDAR、热成像、导航解与云台编码器噪声）、L0 至 L3 观测档、机型包与传感器目录（20）。

推广：

| 项 | 现状 | 目标 |
|---|---|---|
| 机体类别 | `DroneAdapter`，多旋翼为主 | `BodyAdapter`，按 `ct:device/body@1` 的类别参数化：多旋翼、固定翼（`awr_f1`）、地面车、水面艇；每类给出 kinematic、fast、realistic 三档实现，SITL 只对有自驾仪的类别 |
| 云台 | 作为相机位姿的一部分，噪声在 M25 | 云台为子设备，三档：ideal（瞬时指向，CHIMERA 现状）、servo（角速度与加速度限制、编码器噪声、延迟）、joint（gz 关节与控制器）；档位进入 Run 规格与 `ranked` 判定 |
| 指令 | 世界系速度 + 偏航率，另有 `goto`、`follow_path`、`orbit`、`hover`、`takeoff`、`land`、`kill` | 四层级（§6.2），层级之间的展开在 ANet4D 侧的同一路径跟踪器完成，三档物理只见层级 1 |
| 接触 | 与世界的碰撞、机间碰撞、触地 | 增加与动态实体（磁带代理体）的接触、地面车与路缘和车道边界的接触、水面艇与岸线的接触 |
| 能量 | `tasking/energy.py` | 按类别的能耗模型进入 `ct:device/body@1` |

### 6.2 指令层级

| 层级 | 内容 | 消费者 | 来源 |
|---|---|---|---|
| 0 执行器 | 电机、舵面、推进器指令 | realistic 与 SITL 档内部 | gz 模型、PX4 |
| 1 速度与偏航率 | 世界系速度 + 偏航角速度（ADR-137 的统一语义） | 三档物理 | 现有 |
| 2 技能 | `goto`、`orbit`、`follow`、`scan`、`hold`、`land`、`gimbal_track` 等封闭词表，带前置条件与完成判据 | M26 Env、M30 `adapters.drone_agent`、指挥接口 | CHIMERA 技能词表 + 现有 `CmdKind` |
| 3 语言 | 自然语言或结构化意图，由指挥接口翻译为层级 2 | 大模型指挥者（X 层） | CHIMERA commander |

层级 2 的词表以 `ct:device/skills@1` 登记；每个技能的展开是确定性的，展开结果（逐 tick 的层级 1 指令）进入 trace。

### 6.3 新机体类别

| 类别 | fast 档 | realistic 档 | 备注 |
|---|---|---|---|
| 地面车（Terra 测试床） | 运动学自行车模型 + 行为引擎的路廊与停车逻辑（`chimera.life.vehicles` 的 LiveDrive、Reeds-Shepp）复用为被测机体的局部规划 | gz 车辆模型（Ackermann）与接触 | 与 E2 的交通共享可供性层 |
| 水面艇（Aqua 测试床） | 三自由度平面模型，含风与流的扰动（`awr.environment.wind` 复用） | gz 水动力与浮力插件（r05 §4，C 级核实） | 需要水面类别的可供性（航道、岸线、泊位） |
| 固定翼 | 现有 `awr_f1` 与 `py:sim/fixedwing` | gz 固定翼模型 | 现有 |

### 6.4 硬件桥

用户对 ANet4D 的定义是"同一观测与行动接口，仿真与硬件相同"。L3 SITL/HIL 档是入口：

- MAVLink（PX4、ArduPilot）的指令与遥测映射到层级 1 与 `VehicleStatus`；云台协议（SIYI、Gremsy 等，具体型号立项时核对）映射到云台子设备的 servo 档。
- 硬件上的观测只走 L2 与 L3（真实相机），`bench/obs.v1` 白名单不变。
- 硬件运行不进入 `ranked`，只进入 sim2real 回归（ADR-189 决策 (2) 的用途扩展）。

### 6.5 开源选型

| 组件 | 用途 | 核实状态（r05） | 处置 |
|---|---|---|---|
| PX4 SITL、gz-sim Harmonic、`native/gz` 插件 | realistic 与 SITL 档 | A（ADR-137、138） | 沿用 |
| RotorPy | 多旋翼动力学与相机传感器参考 | A（r10） | 参考 |
| XTDrone | PX4 + ROS + Gazebo 多机 | B | 参考多机编组方式，不作依赖 |
| AirSim | 已归档 | B | 不采用 |
| Isaac Lab、ManiSkill3、Genesis | 向量化仿真与 RL 训练 | B | 批量推进的参考（§7 第 11 条） |
| Habitat、OmniGibson、TongSIM | 具身任务与室内场景 | B | 任务语义参考 |
| ArduPilot SITL、gz 水动力插件 | 其他自驾仪与水面 | C | 候选，未核实 |

---

## 7. 其他必需能力

| 序 | 能力 | 现状 | 建议 |
|---|---|---|---|
| 1 | 资产与出处流水线（高斯、网格、动态实体资产的来源、许可、哈希、版本） | M22 构建阶段、ADR-156 资产目录、ADR-192 出处清单 | 扩展到高斯图层与角色资产；`check-release` 扫描许可标识（ADR-190 的机制） |
| 2 | 场景准入诊断 | m23 `scene-audit`；CHIMERA `envpack` | 合并为一份准入报告（§5.2 阶段 1） |
| 3 | 指挥接口与实时网关 | CHIMERA `commander`、`sandbox`（技能词表、锁步调用、实时网关、决策桥、遥测） | 作为 M38，位于平台基础设施；技能词表入契约；实时世界只用于演示与开发，不用于正式成绩 |
| 4 | 场景创作助手 | m23 `designer.py`（DeepSeek）、CHIMERA `directed_extension`、`demo_intents` | 合并为受约束的配方生成器；模型调用冻结为实例输入（AWR-06 §1.8） |
| 5 | Trace 与回放 | M33 `trace.v1`、MCAP、Rerun（r8）、CHIMERA 的运行回放页 | 行为引擎的阶段产物与审计进入 trace；回放页复用 D3 Lab |
| 6 | 交互网络 | M29 | 不变；地面车与水面艇的链路模型参数进入 `ct:device/body@1` |
| 7 | 评测 | M27、M32；CHIMERA 的确定性检查点评测与见证门 | 检查点评测作为 M31 评测器插件的一种登记（基准专有，不回收代码） |
| 8 | 多域可供性 | 只有陆域 | 水域（航道、岸线、泊位）与空域（禁飞、高度层）进入可供性契约 |
| 9 | 硬件桥 | SITL 档 | §6.4 |
| 10 | 动态实体的高斯外观 | 无 | 研究项：4DGS 系（r05 §1.5）；先以网格角色合成 |
| 11 | 批量向量化推进 | 无（FleetSim 单进程） | 研究项：多实例并行的 fast 档与高斯渲染（r05 §2 的向量化仿真参考） |
| 12 | 数据集与 sim2real | ADR-189 参照集 | 高斯派生几何的几何门与 LiDAR 回归并入 |

---

## 8. 实施路线

### 8.1 工作包

估算为粗估（±40%），以一名熟悉两个代码库的工程师计；未计入评审与文档。

| 工作包 | 内容 | 依赖 | 估算 | 出口 |
|---|---|---|---|---|
| WP-E2-01 行为引擎移植 | `chimera.civil`、`chimera.life`、`chimera.envpack` 迁入 `awr.life`，命名空间与依赖改造，磁带写入经 `awr.sim.events`，单元与确定性测试 | 无 | 4 人周 | 在现有 GLB 世界（crop 与 CHIMERA 的小镇）上重编译出与 CHIMERA 发布磁带等价的磁带 |
| WP-E2-02 可供性契约与网格世界适配 | `ct:scene/affordance@1`、`ct:scene/geometry@1` 的 PRD 与 schema；`world/derive` 对网格世界的实现；CHIMERA 的实测支撑、路缘与停车普查改读契约 | WP-E2-01 | 3 人周 | I-E-01 的 lint 检查通过（行为引擎不 import `awr.world.mesh`） |
| WP-E2-03 外部后端 | `anet4d[life-ext]`：SUMO + JuPedSim 后端，跨机器一致性验证 | WP-E2-01 | 2 人周 | 同输入两台机器 `content_sha256` 一致，或登记为 `ranked = false` |
| WP-E1-01 高斯世界包 | `gaussian` 图层、格式登记、splat-transform 入库、出处清单、配准校验 | 无（可与 WP-D3-39 合并，D-09） | 3 人周 | 一个航拍高斯世界入库并经 `worldpkg validate` |
| WP-E1-02 派生几何 L0 至 L2 | 支撑面、占据、ESDF 的派生与自适应校准，G-GEO 工具 | WP-E1-01 | 5 人周 | 对 crop 世界的网格参照通过 G-GEO-1、2、3、5 |
| WP-E1-03 表面网格与 NavMesh | 从 2DGS 或 PGSR 提取 L3，减面与分瓦片，Recast 烘焙，navsvc 不变 | WP-E1-02 | 4 人周 | G-GEO-4；行为引擎在高斯世界上编译磁带 |
| WP-E1-04 高斯相机 | `GsCameraBackend`（gsplat）、动态实体合成、效果链、吞吐测量 | WP-E1-01 | 6 人周 | 720p 与 1080p 吞吐达到 ADR-140 的档位阈值；对拍差值登记 |
| WP-E1-05 网页共渲染 | Spark 共渲染、SOG 流式、Lab 检视 | WP-E1-01 | 4 人周 | WP-D3-41 的验收 |
| WP-E3-01 BodyAdapter 与云台子设备 | `ct:device/body@1`，类别参数化，云台三档 | 无 | 4 人周 | 现有多旋翼三档回归不变；云台 servo 档的噪声与延迟进入 M25 |
| WP-E3-02 指令层级与技能词表 | `ct:device/skills@1`、技能展开器、M26 与 M30 适配 | WP-E3-01 | 3 人周 | CHIMERA 的指挥线束改用 ANet4D 词表 |
| WP-E3-03 地面车与水面艇 fast 档 | 运动学模型、与可供性层的接触、机型包 | WP-E2-02、WP-E3-01 | 5 人周 | Terra 与 Aqua 测试床的最小运行 |
| WP-E3-04 硬件桥 | MAVLink 与云台协议映射，HIL 回归 | WP-E3-01、硬件 | 6 人周 | 一架真机的层级 1 与层级 2 闭环 |
| WP-X-01 实时网关 | `chimera.sandbox` 迁入 `awr.runtime.gateway`，决策桥与遥测 | WP-E3-02 | 3 人周 | CHIMERA 演示改由 ANet4D 网关驱动 |

合计约 52 人周。

### 8.2 批次

| 批次 | 工作包 | 说明 |
|---|---|---|
| 第一批 | WP-E2-01、02、03 | 不依赖高斯；交付后 ANet4D 自带完整的小镇生活引擎 |
| 第二批 | WP-E1-01、02、03 | 高斯作为几何来源；出口是行为引擎在高斯世界上运行 |
| 第三批 | WP-E1-04、05 | 高斯视角与网页 |
| 第四批 | WP-E3-01 至 04、WP-X-01 | 设备引擎推广；硬件桥取决于设备到位 |

### 8.3 分支与评审

- 第一批在模块分支 `m36/life-engine` 开发，按 CONTRIBUTING 的规则：契约先于实现、`make ci` 通过、路径所有权评审。
- 每批以一份实现报告（`docs/impl/`）与审计证据提交；本文随各批修订版本号。
- CHIMERA 仓库在移植完成后改为依赖 ANet4D 的 `awr.life`，删除自己的副本（CHIMERA 侧的工作，不在本文估算内）。

---

## 9. 建议登记的决策（待负责人裁定）

| 编号 | 事项 | 建议 | 影响 |
|---|---|---|---|
| D-01 | 高斯作为几何来源的地位 | 新增几何门 G-GEO；通过的世界以高斯派生几何为真值；外观门 G-3DGS 与排名规则不变 | 修订 R-D3-51"几何真值以网格为准"的适用范围；16 §6.4 定稿 |
| D-02 | 场景可供性契约 | 作为 E1 到 E2 的唯一接口，lint 强制（行为引擎禁止 import 场景表示包） | 新契约命名空间 `ct:scene/` |
| D-03 | 行为引擎的包位置与 m23 的关系 | `awr.life` 为完整引擎，m23 `py:sim/activities/` 保留为轻量生成器，共用 `Generator` 协议与资产目录 | AWR-06 §1.8 扩展；M31 生成器登记 |
| D-04 | 磁带扩展 | 容器关系、动作标签、身份连续性以扩展段承载；是否升级 `tape` 版本 | M23 磁带契约 |
| D-05 | BodyAdapter | `DroneAdapter` 推广为按类别参数化的 `BodyAdapter`，三档不变，云台为子设备 | ADR-137 扩展；M24 |
| D-06 | 指令层级 | 四层级；层级 2 的技能词表为契约 `ct:device/skills@1` | M26、M30 |
| D-07 | 新机体类别 | 地面车与水面艇进入机型目录；fast 档先行 | 20 号文档 |
| D-08 | 实时网关 | 作为平台基础设施（不进入实验语义），只用于演示与开发 | AWR-06 §3.5 |
| D-09 | 模块号与排期 | M35 至 M38 暂定；WP-E1-01 是否与 WP-D3-39 合并；本文工作包纳入 AWR-06 §10 的方式 | 路线图 |
| D-10 | 依赖政策 | gsplat（CUDA）只在服务端镜像；SUMO 与 JuPedSim 为 `anet4d[life-ext]`；D4 内核闭包不变 | ADR-D4-28 |
| D-11 | 许可登记 | Spark、gsplat、2DGS 系、SUMO（EPL-2.0）、JuPedSim 的许可进入 §14 登记与 `THIRD_PARTY_NOTICES` | 发布检查 |
| D-12 | 第一批的分支与所有者 | `m36/life-engine`，所有者 houqiii，评审 M22、M23、M25 的所有者 | CONTRIBUTING |

---

## 10. 风险与未决问题

| 编号 | 风险 | 缓解 |
|---|---|---|
| R-01 | 高斯派生几何在细结构（线缆、杆件、栏杆）上召回不足，导致低空碰撞判定失真 | G-GEO-3 门限；细结构类别用低阈值与形态学闭运算；不通过的世界保持网格真值 |
| R-02 | 航拍训练的高斯在低空斜视下有浮点与空洞（r11 的观察） | 入库体检（`fillRatio`、NaN）、SuperSplat 清理、L0 分位数去浮点；几何门独立于外观门 |
| R-03 | SUMO 与 JuPedSim 的跨机器确定性无保证 | WP-E2-03 验证；不一致时后端 `ranked = false`，只用于演示与数据生成 |
| R-04 | 移植规模（3.5 万行）与两个代码库的同步 | 分阶段移植，CHIMERA 在移植完成前冻结引擎改动；每阶段以 CHIMERA 发布磁带为回归基准 |
| R-05 | gsplat 与 EGL 光栅化的合成在深度边界有缝 | 期望深度与硬件深度统一为 reversed-Z 线性深度后合成；差值登记 |
| R-06 | 城市级高斯的显存与装载 | 瓦片与 LOD（Spark 分页、Aholo 分块）；服务端按视锥驻留 |
| R-07 | 新机体类别的可供性（水域、空域）没有数据 | 先以合成场景验证接口，真实水域世界待数据 |
| R-08 | 许可（Skyfall-GS、FLUX 的限制仍在；新组件许可未核） | r05 的核实状态升级为 A 之前不进入产品镜像 |

未决问题：

| 编号 | 问题 | 处理 |
|---|---|---|
| Q-01 | 高斯世界的 L1 检测（`mesh_los`）用什么做遮挡：L1 占据、L3 网格，还是高斯不透明度积分 | WP-E1-02 实测三者差异后定 |
| Q-02 | 行为引擎的实体尺寸包络与 E3 机体类别是否共用一份尺寸族 | 建议共用 `ct:device/body@1` 的尺寸段 |
| Q-03 | 指挥接口的大模型调用是否纳入 M30 的 B-LLM 预算档 | 由 M30 所有者决定 |
| Q-04 | 2DGS 与 3DGS 是否同时支持入库 | 建议两者都支持，L3 提取优先用 2DGS |

---

## 附录 A CHIMERA 引擎包的依赖扫描（2026-11-02，`b5bede3`）

扫描方法：对各包全部 `.py` 做 AST 解析，统计顶层第三方导入、`awr` 导入与跨包导入（次数为导入语句数）。

| 包 | 文件与行数 | 第三方 | `awr` | 其他 CHIMERA 包 |
|---|---|---|---|---|
| `civil` | 103，20,871 | numpy 76、shapely 73、scipy 56、sumolib 6、yaml 4、numba 3、jsonschema 1、libsumo 1、rsplan 1、py_trees 1 | `awr.sim.events` 11、`awr.world.mesh` 3 | `schema` 11、`render` 6、`life` 4、`envpack` 2 |
| `life` | 65，10,071 | numpy 56、shapely 20、scipy 9、numba 9 | `awr.sensorsvc.camera` 1 | `civil` 58、`schema` 8、`render` 2 |
| `ambient` | 13，4,332 | shapely 34、numpy 12、scipy 7、sumolib 3、contourpy 1、numba 1、jupedsim 1、libsumo 1 | `awr.sim.events` 1 | `civil` 9 |
| `envpack` | 9，1,656 | numpy 4、scipy 4、shapely 4 | `awr.sim.events` 1 | `civil` 4、`schema` 3 |
| `render` | 35，3,599 | numpy 30、scipy 6、PIL 6、numba 2、moderngl 2 | `awr.sensorsvc.camera` 6、`awr.sim.events` 4、`awr.agentsdk.env` 1、`awr.sensorsvc.timing` 1 | `schema` 9、`commander` 2 |
| `commander` | 10，1,432 | numpy 3、PIL 2、anthropic 1、openai 1 | `awr.agentsdk.env` 2、`awr.sim.events` 1 | `accounting` 6、`sandbox` 4、`tasks` 3 |
| `sandbox` | 7，769 | numpy 1、shapely 1 | `awr.sensorsvc.camera` 4、`awr.agentsdk.env` 3、`awr.phys.adapter` 2 | `render` 7、`truth` 4 |

结论：行为引擎（`civil`、`life`、`ambient`、`envpack`，约 3.7 万行）对 `awr` 的耦合只有磁带（`awr.sim.events`）与网格查询（`awr.world.mesh`）两处，前者保留，后者由几何契约替换；对 `chimera.schema` 与 `chimera.render` 的引用在移植时移除。

## 附录 B 术语

| 术语 | 含义 |
|---|---|
| 高斯场景、3DGS、2DGS | 以三维（或二维面元）高斯基元表示的场景；本文"高斯"泛指两者 |
| 可供性（affordance） | 场景对行为的许可与约束：可走、可驶、可停、可进、容量与开放时间 |
| 派生几何层 | 从场景表示派生的支撑面、占据、距离场与表面网格 |
| 磁带 | Canonical Event Tape（M23） |
| 三档物理 | Fast、Realistic、SITL（ADR-137） |
| 观测档 | L0 Oracle、L1 检测、L2 渲染、L3 SITL/HIL（M26） |
| 核实状态 A、B、C | r05 的定义：A 本项目已研读并实测；B 2026-11-02 检索核实（摘要级，版本与许可待核）；C 待核实 |
