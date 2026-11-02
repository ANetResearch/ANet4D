# 目标生活引擎 `awr.life`

| 项 | 内容 |
|---|---|
| 文档 | `python/awr/life/` 的说明：来源、包结构、运行方式、依赖、数据、测试与限制；本目录 `notes/` 收录引擎的设计笔记，`examples/` 收录配置样例 |
| 版本 | v1.0（2026-11-02，移植自 CHIMERA 基准仓库 `houqiii/chimera` 提交 `b5bede3`） |
| 维护 | houqiii；模块归属待负责人登记（当前按 M23 事件运行时的扩展提交，分支 `m23/life-engine`） |
| 移植报告 | [impl/life-engine-v1](../impl/life-engine-v1.zh-CN.md) |

## 1. 这是什么

`awr.life` 生成并推进一个世界里的非被测实体：居民（固定身份与外观）、私家车与车队（出租、公交、快递、警车、救护）、街道对象、日常活动（居家、上班、购物、散步、遛狗、聚集）、停车与路缘使用、交通（含事故与违规剧本），以固定 0.1 s 步长整组推进，输出 D3 的 Canonical Event Tape（`awr.sim.events`）、facts 表与审计产物。它是 CHIMERA 基准演示场景（30 ha 小镇、150 名居民、31 辆车）的目标引擎，在 ANet4D 内作为 M31 生成器的一种实现，与 m23 的轻量生活活动编译器 `awr.sim.activities`（固定模板、无交通）并列。

设计原则（详见 `notes/`）：

- 运行时拥有每一个位姿：决策不写位姿；渲染器与传感器从不推进演员时钟。
- 只在实测支撑上活动：未验证的室内、过街与机动一律拒绝，不以瞬移代替。
- 同一输入（世界包版本、配方、seed、锁定版本）重编译结果逐字节一致；生成器内核不启用 numba `fastmath`。
- 发布前经同一份最终 QA 与生活审计（进度、等待原因、空间跨度、边界进出、碰撞、身份连续性）。

## 2. 包结构

| 子包 | 来源（CHIMERA） | 内容 |
|---|---|---|
| `awr.life.civil` | `chimera.civil` | 场景语义（车道网、路廊、路缘、停车场与规则、公共区域、户外场所、实测支撑）、人口与身份、活动需求与日程（`civil.activity`：导演、配方、驾驶适配、停车准入、出行证据）、一次性构建流水线 `civil.build`（SUMO net → 语义 → 人口 → 日程 → 仿真 → QA → 组装，阶段产物缓存）、`civil.grounded`（实测支撑上的严格演员与 `life_v1` 运行时入口）、`civil.queries`（从 facts 生成可检查的任务候选） |
| `awr.life.motion` | `chimera.life` | `life_v1` 运行时：`LifeEngine` 每 0.1 s 推进决策、上下车、驾驶（出行准入与路廊预留、停车前瞻、Reeds-Shepp 机动）、步行（静态路由、让行、拥堵改道、礼让）、人群、记录、失败证据与生活审计 |
| `awr.life.ambient` | `chimera.ambient` | 路线 A 后端：Eclipse SUMO 车辆仿真于实测车辆支撑、JuPedSim 行人仿真于实测可走地面、路口耦合、转磁带与发布；本身无运动模型 |
| `awr.life.envpack` | `chimera.envpack` | 环境包对"环境 feature 接入原则 v1"的静态诊断（几何、语义、连通性、来源版本、历史测量），不授予运行时准入 |
| `awr.life.appearance` | `chimera.render.{assets, identity, wheels, human_pose}` 与 `assets/` | 外观身份（服装调色、车牌文字）、车轮装配、资产目录 `appearance/catalog/`（catalog.json、三份 lock 文件与 history；资产来源为 MakeHuman 与 Innerscene 的 CC0 资产，源包与编译缓冲不入仓库） |
| `awr.life.common` | `chimera.{schema, world_integrity, scene_entities, admission, truth}` | schema 标识映射、完整性哈希、准入错误、实体清单、真值几何 |

对 `awr` 其他包的依赖只有：`awr.sim.events`（磁带读写、规范 JSON、地面索引、raycast 哈希）、`awr.world.mesh`（网格查询）与 `awr.sensorsvc.camera.raster.ActorFrame`（一处类型引用）。`awr.life` 在导入边界检查中登记为 D3 层包（`tools/lint/check_py_imports.py`），D2 代码不得导入它。

## 3. 标识与兼容

本包写出的文档以 `awr.life.<name>.v<N>` 为 schema 标识。CHIMERA 基准在移植前发布的记录带 `chimera.<name>.v<N>`，更早的带 `d4ale.<name>.v<N>`；`awr.life.common.schema.schema_id` 把三种前缀映射为当前写法，所有检查经它比较。已发布的磁带、任务包与审计记录不改写。

环境变量前缀由 `CHIMERA_` 改为 `AWR_LIFE_`（`AWR_LIFE_ASSETS` 指向资产目录，缺省为包内 `appearance/catalog/`；`AWR_LIFE_SCENES` 指向生活场景夹具目录）。

## 4. 运行

### 4.1 依赖

```sh
pip install -e ".[life]"        # shapely、pillow、rsplan 1.0.10、py-trees
pip install -e ".[life-ext]"    # eclipse-sumo 1.28.0、jupedsim 1.4.2、contourpy（只有 awr.life.ambient 需要）
```

两组都不进入 D1 的 `requirements.lock`（与 `geo-worker`、`px4` 同一处置），也不进入 D4 内核的 numpy-only 闭包。`rsplan` 的版本被离线停车规划器锁定为 1.0.10（`civil.reeds_shepp`）。

### 4.2 民事场景构建（主入口）

```sh
PYTHONPATH=python python -m awr.life.civil.build docs/life/examples/civil-crop-city-002.yaml \
    --world <world dir> --nav <navgrid dir> --work <work dir> --out <event dir> \
    [--world-ref <tape.json>] [--ground grid|mesh] [--duration S] [--skip-sim] [--terrain <terrain dir>]
```

- `<world dir>`：D3 世界包目录（raycast 网格、`coordinate.json`、语义层）。
- `<navgrid dir>`：`awr-navgrid@1` 导航栅格（`*.npy` 与 `meta.json`）。
- 配置文件（`awr.life.civil.config.v1`）给出舞台范围、时间原点、OSM 世界文件、人口与车队、日程与事件剧本；样例见 `examples/civil-crop-city-002.yaml`。
- 配置中 `runtime: life_v1` 选择 `awr.life.motion` 的运行时（经 `civil.grounded.build`），否则使用构建器的编译推进。
- 阶段产物写入 `<work dir>` 并在下次构建时复用；删除对应目录即重建。
- 输出目录已存在时拒绝覆盖（已发布的事件不可变）。
- 最后一个阶段（`civil.assemble` 写磁带）与 `civil.validate_tape` 需要磁带 v2 记录（`awr.sim.events.tape` 的 `RECORD_V2`）；本仓库 `main` 尚无该记录时抛出带说明的 `RuntimeError`，见移植报告 §4 第 1 条。

### 4.3 其他入口

| 命令 | 用途 |
|---|---|
| `python -m awr.life.ambient.build CONFIG --support S --semantics SEM --stage STAGE_DIR --catalog C --out OUT` | 路线 A（SUMO + JuPedSim）场景构建，随后 `python -m awr.life.ambient.publish` 发布为磁带 |
| `python -m awr.life.envpack.validate <environment.json> --queries <queries.json> --out <report dir>` | 环境包静态诊断 |
| `python -m awr.life.civil.queries --event <event dir> --sim <sim dir> --semantics … --out <tasks.yaml>` | 从 facts 生成任务候选 |
| `python -m awr.life.civil.activity.headless` | 只用拓扑的活动集成运行器（无渲染、无 SUMO、无世界下载），用于在昂贵的落地 QA 之前检查需求、所有权与资源 |
| `python -m awr.life.civil.validate_tape` | 磁带对 `packages/contracts/event/tape.v2.schema.json` 的校验 |

### 4.4 数据

世界包、导航栅格、实测支撑、车道登记与停车普查等输入由 CHIMERA 仓库的脚本组产生（`scripts/basemap`、`scripts/envpack`、`scripts/route_b`、`scripts/sandbox`），本次未移植；CHIMERA 的小镇数据保存在 dmax 的 `/data/houqiii/chimera-work`。引擎的测试只用仓库内的合成数据（`tests/life/data/`）。

## 5. 测试

```sh
AWR_KERNEL=numpy .venv/bin/python -m pytest -q tests/life
```

缺少 `life` 附加包时整个目录按 `[dep]` 跳过。`tests/life/` 收录 CHIMERA 测试集中只依赖本包的 112 个文件（CHIMERA 的其余测试依赖其基准专有的包、PyBullet 校验脚本、角色渲染、小镇数据或磁带 v2，未移植；清单见移植报告 §3）；合成小镇夹具为 `tests/life/life_toytown.py`。

## 6. 限制与后续

1. 场景输入仍是 GLB 世界包与导航栅格；引擎经 `awr.world.mesh` 与自己的实测支撑文件读取场景。场景可供性契约（道路网、可走面、活动区域、设施锚点）作为引擎与场景表示之间的唯一接口尚未定义，属于架构讨论事项。
2. 磁带 v2 记录（容器、速度与角色字段）是 CHIMERA 对上游 `705a1be` 的补丁，尚未进入本仓库；采纳与否属于 M23 契约事项。
3. 角色与车辆的渲染（骨骼角色、接触阴影、车辆贴片、L2 相机合成）留在 CHIMERA 的 `chimera.render`，本次只移植了人口阶段需要的外观身份与资产目录。
4. SUMO 与 JuPedSim 后端的跨机器确定性未验证；在验证之前只用于演示与数据生成。
5. 与 m23 `awr.sim.activities` 的协议统一（M31 `Generator`）、随机域登记（流 24 的子域）与 M18 `TargetRuntime` 的绑定待后续工作包。
6. 代码经 `ruff format` 与安全修复，不做改变语义的自动改写；若干 ruff 规则对本包路径在 `pyproject.toml` 中豁免（见移植报告 §2），待清理工作包处理。行为与 CHIMERA `b5bede3` 一致，以 `tests/life` 与 CHIMERA 已发布磁带的重编译为回归基准。

## 7. 设计笔记（`notes/`）

按主题分组；日期为笔记写成的日期，内容以当时的 CHIMERA 代码为准，路径与模块名已改写为本包的名称。

| 主题 | 笔记 |
|---|---|
| 运行时与活动 | [生活引擎 life_v1：设计、验证与现状](notes/life-runtime-v1.zh-CN.md)、[通用活动引擎候选 activities_v1](notes/activity-engine.zh-CN.md)、[场景活动需求配置 v1](notes/activity-recipes-v1.zh-CN.md)、[生成多样活动需求，再验证可执行性](notes/diverse-activity-selection-v1.zh-CN.md) |
| 驾驶与停车 | [车辆活动全过程与出发约束](notes/car-intention-lifecycle-v1.zh-CN.md)、[驾车活动必须使用停车后的步行条件](notes/future-parking-access.zh-CN.md)、[停车变化触发的在途改道](notes/parking-replanning.zh-CN.md)、[停车布局与道路连通的联合检查](notes/parking-road-preservation.zh-CN.md)、[行程车辆的临时停车与行人避让](notes/vehicle-restart-envelope.zh-CN.md)、[混合交通制动原因与可重放证据](notes/drive-diagnostics.zh-CN.md) |
| 道路与车道 | [车道跟随实测路面与车道间换道](notes/carriageway-lanes-2026-10-25.zh-CN.md)、[实测车行带的可选车道生成](notes/measured-lanes-integration.zh-CN.md)、[共享道路交互：下一项通用机制的边界](notes/shared-road-interaction-plan.zh-CN.md) |
| 步行与拥堵 | [有预算的拥堵绕行](notes/congestion-rerouting-v1.zh-CN.md)、[新行程的观测拥堵成本接入主引擎](notes/observed-departure-routing-v1.zh-CN.md)、[质量图冲突与实际轨迹的关系](notes/quality-exposure.zh-CN.md) |
| 环境与后端 | [环境 feature 接入原则 v1](notes/environment-feature-principles-v1.zh-CN.md)、[路线 A：镇中心环境场景（SUMO 车辆 + JuPedSim 行人）](notes/ambient-route-a-2026-10-26.zh-CN.md) |
