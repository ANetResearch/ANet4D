# 目标行为引擎的三个扩展点

引擎（`awr.life`）在第 2 步之后只读 principle 包；本步把"引擎里有哪些目标、它们做什么、谁来推进它们"也从代码里拿出来，变成三处可配置的登记：

| 扩展点 | 回答的问题 | 登记在哪 | 模块 |
|---|---|---|---|
| 实体类别 | 一类目标是什么：磁带类别与导航剖面、身体与资产、身高与速度分布、导演角色、可做的活动 | 内置目录 `classes/default.json`；配方的 `classes`（内联）或 `entity_classes`（目录文件）；`population` 要几个 | `awr.life.classes` |
| 活动配方 | 一类目标做什么：目的、去哪类场所、停多久、多久想去一次、需要场所有什么能力、哪些角色做 | 配方的 `life.activity_engine.activities`（替换缺省规则）或 `additional_activities`（追加） | `awr.life.civil.activity.recipes` |
| 运动后端 | 谁逐 tick 推进已准入的世界 | 登记表 `awr.life.runtimes.REGISTRY`；配方的 `runtime` 选名字，`runtimes` 可按 `"name": "module:callable"` 加后端 | `awr.life.runtimes` |

完成标志（计划第 3 步）："用配置加一类目标，不改引擎"。文笔村上加了一类**跑步者**（8 人，2.6–3.05 m/s，在户外场所之间慢跑、短暂停留），只改了配方（见 §4），引擎代码没有为跑步者加一行；同时不加类别的配方复跑（run v11）与 v10 帧数据逐字节相同。

## 1. 实体类别（`awr.life.entity_classes.v1`）

```json
"classes": {
  "runner": {
    "label_zh": "跑步者",
    "tape_class": "person",            // 磁带类别：person vehicle dog machine object uav
    "profile": "person",               // 导航剖面：person / vehicle / null（静态）
    "body": {"kind": "articulated",    // articulated 步行者 / rigid 车辆 / static 静物
             "assets": "humans",       // 资产族（humans / vehicles / street）或资产 id 列表
             "height_m": {"base": 1.68, "step": 0.04, "cycle": 4}},   // base + step·(j mod cycle)
    "motion": {"max_speed_mps": {"base": 2.6, "step": 0.15, "cycle": 4}},
    "role": "runner",                  // 导演角色；活动配方用 roles 指向它
    "activities": ["jog"],             // 可选：只做这些目的
    "min_activity_kinds": 2,           // 可选：准入时家附近至少几类可达场所（缺省 3）
    "semantic_type": "pedestrian",
    "description": "跑步的人"
  }
},
"population": {"resident": 150, "runner": 8}
```

- 内置三类：`resident`（居民，`person-NNN`）、`local_traffic`（本地车辆）、`street_object`（街道设施）。它们的分布公式与引擎原先的硬编码完全相同（身高 1.60 + 0.04·(j mod 6)，步速 1.0 + 0.08·(j mod 4)，车速 3.0 + 0.2·(j mod 4)），所以旧配方的录制结果不变。内置类不能删除。
- `population` 缺省等于旧的 `people`（全是居民）；居民永远排在前面，索引、id 和随机流不变。本阶段只准入 `articulated` 的 `person` 类作为步行者；车辆与静物仍按原路径准入。
- 演员 id 用类别 id 做前缀（`runner-003`），`id_prefix` 可改；前缀不能重复。
- `python -m awr.life.classes check 配方.json` 打印解析后的类别与人口。契约 `packages/contracts/life/entity_classes.v1.schema.json`。

引擎里由类别决定的地方：步行网络的身体包络（所有步行者类）、准入时的资产/身高/速度/角色/描述、导演的角色与可做目的、准入门槛 `min_activity_kinds`。物理常量（紧急减速、运动限值、QA 阈值）仍按磁带类别取，不按实体类别。

## 2. 活动配方

```json
"life": {"activity_engine": {
  "additional_activities": [
    {"purpose": "jog",
     "kinds": ["park", "cafe", "place_of_worship", "townhall", "school", "post_office"],
     "roles": ["runner"],
     "dwell_s": [15, 40], "recurrence_s": 45, "readiness": 0.9,
     "requires": ["outdoor_stay"]}
  ]}}
```

- 字段：`purpose`（唯一）、`kinds`（场所类型，来自 principle 的语义）、`requires`（场所必须有的能力，来自场所证据，配方不能凭空赋予）、`dwell_s`、`recurrence_s`、可选 `weight`、`modes`（walk/car）、`readiness`、`tour`、`roles`（缺省 `resident`）。
- `activities` 替换缺省的设施规则（`itineraries.LIFE_FACILITY_PURPOSES`），`additional_activities` 追加；目的不能重复。
- 一个角色至少要有一条配方点名，否则该类别准入时拒绝（"No activity recipe names the role"）。
- 配方只说需求；路线、容量、身体、支撑、让行、门限审计都还是引擎和后端的事。

## 3. 运动后端

```json
"runtime": "life_v1",
"runtimes": {"my_backend": "my_package.backend:run"}
```

- 已登记：`grounded_v1`（构建器自带的编译循环，缺省）、`life_v1`（`awr.life.motion.runtime.run`，`LifeEngine`）。
- 后端是一个 `run(ctx)`：`ctx` 是构建器装好的字典（`cfg`、`jobs`、`actors`、`n`、`out`、`warm`、`nt`、`tracks`、`vel`、`yaw_track`、`state`、`container_track`、`events`、`occupancy`、`rest_r`、`surface`、`walking`、`ext`、`xy`、`yaw`、`activity_runtime`、`driving_binding`、`walk_radius` …），后端填满每个 tick 的 `tracks/vel/yaw_track/state/container_track`，返回生活上下文字典；不得改花名册。准入、导演、磁带装配、终检、生活审计、发布都在引擎。
- 代码里 `awr.life.runtimes.register(name, callable)` 也可登记。

## 4. 文笔村验证

| 运行 | 配方 | 结果 |
|---|---|---|
| v11 | 与 v10 相同（无 `classes`/`population`） | 帧数据与 v10 逐字节相同（`body_sha256` 相同） |
| v12 | 加 `runner` 类 8 人与 `jog` 配方 | 158 名步行者；跑步者以 2.6–3.05 m/s 在户外场所之间移动；QA、生活门限见引擎进展页 |

## 5. 还没做的

- 车辆类与静物类的 `population` 准入（现在仍走 `vehicles`/`props`）。
- 新的身体种类（骑行者、四足、无人机）需要资产与身体模型，目录能登记 `tape_class`，但步行网络与碰撞只认 articulated/rigid/static 三种。
- 运动后端的 `ctx` 契约还是字典，没有类型化；第 5 步设备引擎接入时再定。
