# 目标生活引擎 `awr.life` 移植报告（v1，2026-11-02）

| 项 | 内容 |
|---|---|
| 提交 | 直接提交 `main`（基线 `c79a3e0`，V0.3.4） |
| 来源 | CHIMERA 基准仓库 `houqiii/chimera`，提交 `b5bede3`（2026-11-02）的 `chimera/{civil, life, ambient, envpack}`、共享辅助模块与资产目录 |
| 说明文档 | [docs/life/README.md](../life/README.md)（包结构、运行、依赖、限制、设计笔记索引） |
| 作者 | houqiii |

## 1. 做了什么

1. 代码移入 `python/awr/life/`（202 个 Python 文件，约 5.2 万行，含注释与空行）：`civil`（103 文件）、`motion`（原 `chimera.life`，65 文件）、`ambient`（13）、`envpack`（9）、`appearance`（原 `chimera.render` 中人口阶段需要的 4 个模块与资产目录）、`common`（5 个辅助模块）。
2. 命名空间改写：`chimera.civil` 至 `awr.life.civil`，`chimera.life` 至 `awr.life.motion`，`chimera.ambient` 至 `awr.life.ambient`，`chimera.envpack` 至 `awr.life.envpack`，辅助模块至 `awr.life.common` 与 `awr.life.appearance`；代码、文档字符串与命令行示例一并改写。
3. 标识改写：输出文档的 schema 标识由 `chimera.<name>.v<N>` 改为 `awr.life.<name>.v<N>`；`awr.life.common.schema.schema_id` 把 `chimera.`、`d4ale.`、`drone4d-ale.` 三种旧前缀映射到新写法，已发布的磁带、任务包与审计记录不改写。环境变量前缀 `CHIMERA_` 改为 `AWR_LIFE_`。
4. 路径修正：源码哈希（构建出处）、磁带 schema、资产目录由 CHIMERA 的仓库布局（`chimera/`、`vendor/ANet-Drone4D/`、`assets/`）改为本仓库布局（`python/awr/life/`、`python/awr/sim/events/`、`packages/contracts/event/`、`appearance/catalog/`）。
5. 依赖：`pyproject.toml` 新增附加包 `life`（shapely、pillow、rsplan 1.0.10、py-trees）与 `life-ext`（eclipse-sumo、jupedsim、contourpy），两者不进入锁文件，在 `tools/lint/check-deps.mjs` 的 `PY_UNLOCKED_EXTRAS` 登记（与 `geo-worker`、`px4` 同一处置）。
6. 导入边界：`awr.life` 在 `tools/lint/check_py_imports.py` 登记为 D3 层包（`D3_PACKAGES` 与 `_D3_PATHS`），D2 代码不得导入它；`awr.life` 对 `awr` 的导入只有 `awr.sim.events`、`awr.world.mesh` 与 `awr.sensorsvc.camera.raster.ActorFrame`。
7. 测试：`tests/life/` 收录 CHIMERA 测试集中只依赖本包的 112 个测试文件、合成小镇夹具 `life_toytown.py` 与小型数据（`tests/life/data/`，约 330 KB）；缺 `life` 附加包时整个目录不被收集（conftest 中的 `collect_ignore_glob`；在 conftest 导入时跳过会成为收集错误，已在无附加包的环境中验证）。
8. 文档：`docs/life/README.md`、18 篇设计笔记（`docs/life/notes/`）、配置样例（`docs/life/examples/`）；`docs/README.md` 登记。

## 2. 为满足仓库规则所做的改动

原则：只做不改变语义的处理。第一次尝试把 ruff 的"unsafe"自动修复用于集合与数值规则（`RUF005`、`RUF046`、`RUF007`、`B007`、`RUF059`、`SIM1xx`、`PERF401`、`F841`）时，测试集暴露了行为变化（numpy 的广播加法 `ndarray + [a, b]` 被改写成列表拼接，循环变量重命名后遮蔽了同名辅助函数，共 22 个用例失败），因此这些规则一律不自动改写，而在 `pyproject.toml` 对 `python/awr/life/**` 与 `tests/life/**` 显式豁免，留给清理工作包逐处处理。

| 规则 | 处理 | 行为影响 |
|---|---|---|
| ruff `E701`、`E702`（一行多语句，共 3,363 处） | `ruff format` 全量格式化（307 文件） | 无 |
| `I001`、`F401`、`UP`、`RUF021`、`E401`、`W291` 等安全修复 | `ruff check --fix` | 无 |
| `E731`（lambda 改 def）、`SIM105`（`contextlib.suppress`）、`PERF102`（字典迭代）、`B905`（`zip` 显式 `strict=False`）、`SIM300`、`UP` 系列 | `ruff check --fix --unsafe-fixes` 限定这些机械改写 | 无 |
| `B007`、`B008`、`B020`、`B023`、`E741`、`F841`、`PERF401`、`RUF005`、`RUF007`、`RUF012`、`RUF046`、`RUF059`、`SIM102`、`SIM103`、`SIM108`、`SIM114`、`SIM115`、`SIM117`、`SIM201`（测试另含 `B017`、`F811`） | `pyproject.toml` `per-file-ignores` 对两个路径豁免，附说明 | 无 |
| `PY-CB-01`（bus 回调只能入队） | `civil/traffic.py` 中 libsumo 的 `vehicle.subscribe(vid, varIDs=[...])` 改为关键字实参，避免被当作 bus 回调 | 无 |

结果：`ruff check python tools tests`（与 `make lint` 的 Python 部分相同）通过，`ruff format --check` 对两个路径通过，`check_py_imports.py` 与 `check_py_callbacks.py` 通过，`check-deps.mjs` 通过，`no-emoji.mjs` 对新增文档通过。`make ci` 的完整链路（含 Node 侧）未在本机执行。

## 3. 测试结果

环境：`/data/projs/houqiii/.tools/anet4d-venv`（Python 3.12.3，`requirements.lock` 同步，附加 `life` 与 `life-ext`，`pip install -e .`）。

| 运行 | 结果 |
|---|---|
| 移植后、格式化前（129 个文件，含后来剔除的 17 个） | 763 通过、15 失败、20 错误、19 跳过、2 xfail，506 s |
| 格式化与修复后（112 个文件，最终提交的树） | 760 通过、0 失败、19 跳过（14 个需要 PyBullet 原生构建 `AWR_LIFE_BULLET_BUILD`，5 个需要原生桥与真实场景夹具）、2 xfail，404 s |

剔除的 17 个测试文件及原因：

| 文件 | 原因 |
|---|---|
| `test_access_rebuild`、`test_body_clearance`、`test_curb_maneuver`、`test_fleet_frontiers`、`test_mixed_bullet`、`test_native_road_windows`、`test_realism_contracts`、`test_regional_parking`、`test_road_review`、`test_scenario_overlay`、`test_support_chains`、`test_supported_frontiers`、`test_turn_experience` | 导入 CHIMERA 的 PyBullet 校验脚本（`scripts/physics/bullet/*`，未移植） |
| `test_life_capability` | 需要登记后的小镇环境数据（世界包、12 MB 路口端口、停车语义） |
| `test_long_scene_contracts`、`test_appearance_variant` | 需要磁带 v2 记录（见 §4 第 1 条）；在 CHIMERA 自己的环境中也分别失败与报错 |
| `test_native_actions`、`test_native_activity_execution` | DetourCrowd 原生后端的陈旧测试（`Director` 签名已变）；CHIMERA 环境缺 `py_trees` 而一直跳过 |

格式化前的 15 个失败中，12 个来自上述剔除文件，3 个来自源码哈希路径仍指向 `life/traffic_evidence.py`（已改为 `motion/`）。中间一轮（含集合与数值规则的自动改写）出现 22 个失败，是放弃这些改写的依据（§2）；安全修复 `F401` 曾删去三个测试文件对 autouse 夹具 `contacts` 的再导出，已加 `noqa` 恢复。

## 4. 待补能力与限制

1. **场景磁带生成待补**（负责人 2026-11-02 决定）。CHIMERA 的引擎对着"上游 `705a1be` + 四个补丁"开发，其中 `patches/realistic-civil-world.patch` 为 `awr.sim.events.tape` 增加磁带 v2 记录（`RECORD_V2`、`NO_CONTAINER_V2`：容器、速度与角色字段）与 `tape.v2.schema.json`，并改动 `gen.py`、`sensorsvc/camera/raster.py`、`capture.py`、`agentsdk/env/l2.py`。本仓库的场景尚未对齐目标引擎所依据的场景 principle，因此不移植这些补丁：场景引擎将先规定标准化的场景 principle，目标引擎依据它生成；磁带 v2 随高斯（3DGS）场景引擎一并定义后再补齐生成能力。当前 `civil.assemble.build` 与 `civil.validate_tape` 抛出 `RuntimeError(SCENARIO_TAPE_PENDING)`，第 1 至 6 阶段（准入、语义、人口、日程、运动求解、QA）可用。引擎与场景的接入点清单见 [docs/life/README.md](../life/README.md) §2。
2. 渲染耦合未移植：骨骼角色、接触阴影、车辆贴片与 L2 相机合成留在 CHIMERA（`chimera.render` 与对 `sensorsvc/camera` 的补丁）。
3. 场景输入的生产脚本（`scripts/basemap`、`scripts/envpack`、`scripts/route_b`、`scripts/sandbox`）与小镇数据未移植；它们产出的实测支撑、车道登记与停车普查将由场景 principle 的派生层取代。
4. SUMO 与 JuPedSim 后端的跨机器确定性未验证。
5. 与 m23 `awr.sim.activities` 的生成器协议、随机域登记（流 24 子域）、M18 绑定待后续工作。
6. 上游自 `705a1be` 以来的改动（`sim/events/appearance.py`、`camera/actorbuf.py`、`camera/encode.py`、`appearance_lib.v1`）与 CHIMERA 的角色外观路线存在重叠，随渲染部分一并处理。

## 5. 复现

```sh
# 环境
uv venv --python 3.12 .venv && uv pip sync requirements.lock --python .venv/bin/python
uv pip install --python .venv/bin/python -e ".[life,life-ext]"
# 检查
.venv/bin/ruff check python/awr/life tests/life && .venv/bin/ruff format --check python/awr/life tests/life
.venv/bin/python tools/lint/check_py_imports.py python/awr/life
.venv/bin/python tools/lint/check_py_callbacks.py python/awr/life
node tools/lint/check-deps.mjs
# 测试
AWR_KERNEL=numpy .venv/bin/python -m pytest -q tests/life
```
