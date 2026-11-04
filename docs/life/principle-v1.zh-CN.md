# 场景 principle v1（`awr.life.principle.v1`）

目标行为引擎（`awr.life`）的**唯一场景输入**。场景建模进程交付的不是网格、点云或高斯，而是一个 principle 包：引擎只读这个包，不打开任何场景表示文件。

本文件是交付要求：场景进程按此产出，Anet4D 用 `python -m awr.life.principle check DIR` 验收，通过后才能进入 `civil.grounded --principle DIR`。

## 1. 为什么是这个边界

引擎推进目标（人、车、物）需要回答的只有五类问题：

| 问题 | 层 |
|---|---|
| 这里能不能站、能不能开，地面多高 | `support`（栅格） |
| 路在哪、房子在哪、设施在哪、出入口是否核实 | `semantics` |
| 车沿哪条车道走、路口怎么连 | `lane_network`（SUMO net） |
| 人在室外停留点在哪（到不了室内时的替代点） | `outdoor_places` |
| 磁带头引用的世界是哪一个 | `world`（见证文件，不含网格） |

这五层都可以从任何一种场景表示（纹理网格、点云、3DGS）派生。把派生留给场景进程，引擎就不依赖表示格式；换一种数据，引擎一行不改。

## 2. 包结构

```
<principle>/
  principle.json           清单（本规范）
  support/                 awr.life.support.v1：meta.json ground.npy walk_person.npy walk_vehicle.npy z_person.npy z_vehicle.npy
  semantics.json           awr.life.town_semantics.v1
  lanes/net.net.xml        可选，SUMO 路网
  places.json              可选，awr.life.outdoor_places.v1
  world/                   磁带头引用的世界：world.json、geometry/nav/{person,vehicle}.json、geometry/mesh/raycast/*
```

`world/` 只含见证文件：`world.json`（id、contentVersion、bounds），导航见证（`geometry/nav/person.json` 的 `bin.sha256` 必须等于 `support/walk_person.npy` 的 SHA-256），raycast 层（磁带头的 `raycast_sha256` 由它算出）。**网格不在包里**。

## 3. 清单字段

Schema：`packages/contracts/life/principle.v1.schema.json`。

| 字段 | 含义 |
|---|---|
| `id` | 场景 id，等于 `world/world.json` 的 `id`，进入磁带头 |
| `content_version` | 所有绑定文件 `layer:file:sha256` 行排序后的 SHA-256；任何一层改动都会改变它 |
| `frame` | `units: m`、`axes: ENU`、`z_up: true`、`bounds: [x0,y0,x1,y1]`；可选 `georeferenced`、`origin{lat,lon,alt_m,crs}` |
| `layers.<name>.href` | 包内相对路径 |
| `layers.<name>.sha256` / `files{}` | 每个文件的 SHA-256 |
| `layers.<name>.source` | `measured`（实测）/ `derived`（由规则从场景表示派生）/ `assigned`（指派，如设施类型）/ `manual`（人工标注） |
| `layers.<name>.verified` | 是否有人核实过 |
| `capabilities.<name>.status` | `supported` / `unsupported`（必须给 `reason`）/ `unknown` / `absent` |
| `provenance` | 生产者、输入、备注 |

能力名：`walking`、`vehicles_parked`、`driving`、`indoor`、`public_transport`、`cycling`、`water`。`walking` 与 `driving` 必须声明。

## 4. `check` 做什么

`check` 是准入门，不通过的包引擎拒收：

1. 清单：schema、必填字段、层的 `source`/`verified`、`frame`。
2. 文件：每个绑定文件存在且 SHA-256 与清单一致；`content_version` 与绑定层一致。
3. 支撑栅格：`meta.json` 的 `nx/ny/cell_m` 与五个数组形状一致；`walk_*` 为 bool；可走格必须有有限地面高；至少一个可走格。统计可走面积、连通块数、最大连通块面积、车格面积。
4. 语义：id 唯一；道路至少两点且宽度为正；多边形至少三点；POI 的 `building` 指向存在的建筑，`verified_entrance` 为布尔，`source` 合法；越界项计为告警。统计道路公里数、建筑/POI/开放区数、核实出入口数。
5. 世界见证：`world.json` 的 id 与清单一致；`contentVersion` 与 `layers.world.content_version` 一致；`raycast_sha256` 与 `geometry/mesh/raycast` 一致；导航见证的哈希与支撑数组一致。
6. 能力与证据：`walking: supported` 必须有可走格；`vehicles_parked: supported` 必须有车格；`driving: supported` 必须绑定 `lane_network`。

报告格式 `awr.life.principle_check.v1`：`pass`、`errors`、`warnings`、`stats`。

## 5. 引擎如何使用

```
python -m awr.life.civil.grounded --principle DIR --config recipe.json --out run/v<N>
```

- `--principle` 取代 `--world/--nav/--semantics`，两者不能同时给。
- 配方（`--config`）不再写场景文件路径：`lane_network`、`outdoor_places` 由包提供；配方若自带这两个键且指向别处，拒绝。
- 配方的需求必须在包声明的能力之内（`awr.life.principle.demand_errors`）：要走路的人需要 `walking: supported`；要停车需要 `vehicles_parked` 或 `driving`；要开车（`life.vehicles.population.curb_trip > 0`、`life.driving`、`delivery`）需要 `driving: supported`。
- 输出目录里多两个文件：`principle.json`（清单副本）与 `principle-check.json`（验收报告）；构建出处 `build-inputs.json` 记录 `principle{id, content_version, root}`。

## 6. 从旧输入打包

现有环境（世界引用目录 + 支撑目录 + 语义 + 配方）可以直接打包：

```
python -m awr.life.principle pack --world W --nav N --semantics S --config C \
    [--sources sources.json] [--capabilities caps.json] --out DIR
```

`sources.json` 形如 `{"support": {"source": "derived", "verified": false, "method": "..."}}`；缺省 `derived` / 未核实。能力缺省由栅格推出（有人格→`walking`，有车格→`vehicles_parked`），并把配方 `life.unavailable` 里的 `car_journeys`、`indoor_stays`、`public_transport`、`cycling` 转成 `unsupported` 加原因；`--capabilities` 可覆盖。打包后自动 `check`。

## 7. 对场景进程的要求（交付清单）

1. 一个目录，含 `principle.json` 与 §2 的层；`check` 通过。
2. 坐标：米、ENU、z 向上；`frame.bounds` 覆盖所有层；若有地理参考给 `origin`。
3. 支撑栅格的格距由场景分辨率决定（文笔村 0.5 m）；`unknown_is_blocked` 为 true。
4. 每一层如实填 `source` 与 `verified`；指派的东西（设施类型、出入口）写 `assigned`，不要写 `measured`。
5. 不支持的活动写 `unsupported` 并给原因，不要留空让引擎猜。
6. 包内不要放网格、点云或贴图；它们属于场景引擎（CHIMERA 的建模/修复进程），引擎不读。

## 8. 版本

v1：文笔村环境已打包并以 `--principle` 复跑（run v10），帧数据与旧输入（run v9）逐字节一致。后续新增层（水面、室内、空域）通过加层名与能力名扩展，不改已有字段。
