# 00-index：研究总览、选型矩阵与整合结论

> 项目：ANet Drone / World Runtime（真实世界无人机数字孪生与多智能体仿真平台）
> 日期：2026-09-28 ｜ 作者：首席研究整合（基于 38 个研究单元笔记 + 本次整合补充实测）
> 上游：`docs/01-design.md`（原始设计）、`docs/02-refs.md`（参考仓库清单）
> 下游：系统架构说明书、技术选型说明书、业务逻辑设计说明书、UI 交互 PRD、产品 PRD、分模块 PRD，以及 MVP 实现
>
> 阅读顺序建议：§0（十条总结论）→ §3（按模块最终推荐）→ §8（冲突裁决）→ §9（缺口）→ §5（MVP 复用清单）。§2 的全仓库矩阵用于查表。

---

## 0. 十条总结论（先读）

1. **World 是核心资产，坐标、单位、时间契约必须在 V0.1 定死。** World ENU（米、Z-up、右手系）是唯一度量坐标系；GPU 上只允许相对坐标（节点局部量化或局部 ENU float32）；时间统一为 int64 `t_sim_ns`；姿态四元数统一为 `[x,y,z,w]`、表示 WORLD←BODY(FLU)；PX4 的 NED/FRD 只在网关边界换算。UrbanScene3D 六城的单位、上方向、倾斜各不相同（旧金山 10.15 m/单位且需绕 Z 旋转 +90°，芝加哥 ×1000 且需调平 2.03°，苏州 Y-up 且无地面），**ingest 规范化是 MVP 的必经阶段**（x01 为权威）。
2. **MVP 用双链路。** 主链路是"内置 UrbanScene3D 世界 → ingest → 八叉树 → Web 渐进加载 → Mock 机群 → WebSocket → 控制与回放"，在本机（无 GPU）完整可跑、可测。重建链路（LingBot-Map）只交付 Engine Adapter 接口、Mock 重建引擎和可选的 DA3-SMALL CPU 冒烟，真推理放 GPU 节点（V0.5）。
3. **点云运行时格式 = Potree 2.0 三文件容器 + `ANET_Q16` 节点编码（12 B/点，节点内点序打散）+ HTTP Range 静态服务。** 3D Tiles 1.1 只作为导出格式，COPC 作为归档和交换格式。本次整合已实测 Starlette 1.7.0 `StaticFiles` 正确返回 206 Range 响应。
4. **"疏密自动调节"是闭环，不是阈值表。** 做法是：设备像素 SSE（点间距投影）两级 best-first 选择 + 全局点预算 + 节点内前缀绘制与淡入 + FPS 反馈的离散画质阶梯（含两个软件档）+ 自适应点径 + EDL。原设计"远处 10K / 近处 1M"应删除。
5. **渲染器是全局架构决策。** WebGPU 的点图元恒为 1 px；`WebGPURenderer` 的 WebGL2 后端把 `gl_PointSize` 写死为 1.0。建议采用 `RenderBackend` 双后端：
   - 硬件 WebGPU 档：`WebGPURenderer` + TSL（点云用 vertex-pulling 四边形，V0.3 起评估 compute 软光栅）。
   - WebGL2 与软件档：经典 `WebGLRenderer` + GLSL 点材质（真 `gl_PointSize`）。其他图层的 TSL 材质通过 r186 的 `WebGLRenderer.setNodesHandler(new WebGLNodesHandler())` 运行。本次核对源码确认，这条路径不支持 MRT、WebGPU 后处理栈、storage texture 和 compute，因此该档的 EDL 用 GLSL，粒子用无状态顶点着色器。

   该方案仍需在本机做一次实测对比，见 §9 G1。
6. **仿真后端采用逐级提升的保真度阶梯。** MVP 用向量化 Mock FleetSim（PX4-lite 控制级联，已与 SIH 对照验证；1000 架 @250 Hz 单核 3.5 ms），所有机体在同一时钟下步进。V0.2 接 PX4 SIH 容器（无 GPU、每机约 0.22 核）；V0.4–V0.6 做 HIL 桥（由我方物理驱动 PX4）；Gazebo/Isaac 只用于 GPU 节点上的传感器仿真。所有后端统一在 `DroneAdapter` 接口之后，UI 不感知差异。
7. **实时协议自研（r27 称 `anet.rt.v1`，建议改名 `awr.rt.v1`）。** JSON 控制面 + 16 B 对齐的二进制 BATCH 数据面；DroneState 采用 Full64/Lite32 raw struct；credit 窗口背压；WebSocket 放在 Worker 里；按 tick 对齐取最新值并保证尾帧；TIME/epoch 时钟同步；MCAP 录制。大块数据走 HTTP。ROS2 不作为必经环节。
8. **环境只有一个物理真值。** 能见度 MOR → 消光系数 σ，由前端雾、相机退化、LiDAR 衰减共用；风用分级 L0–L3，每架机有独立湍流滤波器（Dryden），不存 4D 湍流场；服务端对天气状态有权威，客户端只做短阻尼。视觉按 Low/Med/High 分档：降水用无状态粒子、雾用解析高度雾、云分 2D 与体积、云阴影全档开启。浏览器 GPU 计算只服务可视化。
9. **设计体系四件套已可落地。**
   - 色卡 ANet Graphite：冷调科技灰 5 档数据阶加品牌红 `#E93024`，一张图只有一处红，状态靠形状而不是新色相表达。
   - 动效：transitions.dev token，对接 Base UI 的 `data-starting-style/data-ending-style`，用 motion tier 与点云共享 FPS 信号。
   - 图标：morphicons 1.7.1 + lucide@1.48 数据，只有白名单图标对做 morph，其余用 swap。
   - 组件：shadcn 4.21，base-mira（Base UI）风格。

   全站禁用 emoji 和 `backdrop-filter`。
10. **ANet 是秒级协作平面，不是控制平面**（实测一次委派往返约 1 s）。每架机一个 AID，能力 id 用裸形式；采用合同网流程 find → quote → score → delegate；效果状态与任务状态是两条独立的轴；使用自建 hub。V0.6 在进程内 Mock ANet 语义，V1.0 接入真 daemon。

---

## 1. 研究总览与方法

### 1.1 范围

| 项 | 数量 / 说明 |
|---|---|
| 研究单元 | 38 个：r01–r27（设计文档与 02-refs 指定仓库）、d01–d05（设计体系与 ANet）、x01（UrbanScene3D 数据）、n01–n05（2025–2026 新项目发现） |
| 本地克隆 | `refs/` 下 121 个仓库，分 12 组：backend 4、data 1、design 5、discovery 44、dynamic4d 3、lidar 13、recon 7、sim 11、swarm 5、weather 8、web3d 13、world 7。另有 `.cache/research/*` 下的补充克隆，包括 Lichtblick、foxglove-sdk、R3F v10、drei v11、zenoh-ts、3DTilesRendererJS 等 |
| 评估条目 | 约 260 条（含子模块、仅做快速评估未克隆的 discovery 项目、npm 包）；去重后约 230 个独立项目 |
| 笔记总量 | 约 2.6 MB，均在 `docs/research/` |
| 实测脚本与原始数据 | `/data/projs/anet-drone/.cache/research/<unit>/`（每份笔记都注明了脚本位置） |

### 1.2 评估维度与结论口径

| 维度 | 口径 |
|---|---|
| **2026 活跃度** | 以默认分支最近一次提交为准（2026-09-28 实测）。2026 年仍有提交的加分；2024 年及以前停更的，只取算法（port/reference） |
| **star** | GitHub star（实测）。同类仓库中 star 高者优先，但契合度与实测结论优先于 star |
| **契合度** | 与本项目栈（React + three r186 + TSL + shadcn、FastAPI、无 GPU 服务器、UrbanScene3D）的兼容程度，以及能否替换 |
| **MVP 价值** | 能否直接进入 V0.1–V0.2 主链路：内置世界、渐进加载、疏密调节、Mock 机群、实时通信、UI |
| **本机实测** | 能跑的都在本机跑过（CPU、SwiftShader、headless Chromium 151）。SwiftShader 下的绝对 FPS 不代表真 GPU，**只采信相对量级与行为结论** |

**结论四档**：

- **adopt**：作为依赖或工具直接使用，锁定版本。
- **port**：移植算法、公式或协议语义，重写进我方代码。
- **reference**：只读参考，不移植代码。
- **skip**：不采用，笔记中写明原因。

按用户要求，**所有 license 限制一律忽略（科研用途）**，但权重来源和许可信息仍记录在 `engine.json` 等元数据里，便于追溯。

### 1.3 本机环境与实测口径

- 硬件：8 核 Xeon E5-2603 v4（1.7 GHz），无 GPU、无 CUDA。研究期间机器与其他单元共享，load average 在 5–98 之间波动，**所有绝对耗时偏悲观**。
- 浏览器：Chrome for Testing 151（Playwright 缓存 rev 1234）。默认 headless 下 WebGPU adapter 为 null，three 自动回退到 WebGL2（ANGLE + SwiftShader）。加上 `--enable-unsafe-webgpu --enable-features=Vulkan --use-webgpu-adapter=swiftshader`（可选加 `--enable-unsafe-swiftshader`）可以得到 SwiftShader WebGPU fallback adapter，功能可测，性能按软件渲染对待。
- 运行时：Node 22.12.0，Python 3.12（项目 venv `/data/projs/anet-drone/.venv`，现有 numpy 2.5.3、open3d 0.20.0、laspy 2.7.0、plyfile 1.1.5；**尚未安装** fastapi、scipy、pyproj、msgpack、zenoh 等）。
- Open3D 0.20 在本机 import 需要 libEGL（r06 的做法是本地解包 deb 并设置 `LD_LIBRARY_PATH`，见 `.cache/research/r06/run.sh`）。

**本次整合补充的两项实测或核对**：

| 项 | 结果 | 意义 |
|---|---|---|
| Starlette 1.7.0 `StaticFiles` 的 HTTP Range（在 `.cache/research/r27/venv` 中起 uvicorn，请求 `Range: bytes=100-1099`） | 返回 **206 Partial Content**，`accept-ranges: bytes`、`content-length: 1000`，字节与源文件逐字节一致 | 关闭 r09/r12 的"Starlette Range 未验证"疑点。MVP 用 FastAPI 挂 `StaticFiles` 直接服务 `octree.bin`/`hierarchy.bin` 即可，不需要 nginx |
| three r186 `examples/jsm/tsl/WebGLNodesHandler.js` 文件头的 Limitations 注释 | 不支持 VSM 阴影、MRT、Transmission、WebGPU 后处理栈（RenderPipeline）、storage texture；fog/environment 不会自动更新；实例化几何不能共享 | 经典 `WebGLRenderer` 档可以运行 TSL 材质，但 EDL、云等后处理必须有 GLSL 版本，粒子不能依赖 compute |

### 1.4 研究单元索引

| 单元 | 主题 | 笔记 | 一句话结论 |
|---|---|---|---|
| r01 | LingBot-Map + viser | `r01-lingbot-map-viser.md` | LingBot 是主重建引擎，但输出无尺度、不对齐重力，需要 ≥16–24 GB GPU；GNSS Sim3 前移到 V0.1；viser 移植协议与性能模式 |
| r02 | VGGT + COLMAP/GLOMAP | `r02-vggt-colmap.md` | 以 COLMAP 4 数据模型为骨架建 Recon IR；VGGT（W2C）与 LingBot（C2W）方向相反；直线航带需要重力增强 Umeyama |
| r03 | nerfstudio / 3DGS / gsplat | `r03-nerfstudio-3dgs-gsplat.md` | Visual World 走五步升级路径；gsplat 只在离线 GPU 上用；移植量化、Morton、自适应密度控制器 |
| r04 | Livox SDK2 / driver2 / LI-Init | `r04-livox-driver-calib.md` | 以 MID-360 数据语义作为内部 LiDAR 契约；参数化扫描模式；虚拟 MID-360 放 V0.2 |
| r05 | FAST-LIO / LIVO2 / R3LIVE | `r05-fastlio-livo-r3live.md` | LIO 是度量骨架；移植 map delta 流式与可观测性指标；伪 V0.5 演示 |
| r06 | Open3D | `r06-open3d.md` | 用 RaycastingScene 构建 Geometry World（DSM → BVH），提供碰撞、净空、LiDAR 查询；需要单独进程（持有 GIL） |
| r07 | GICP / 图优化 | `r07-gicp-graph-slam.md` | 视觉到 LiDAR 必须用 Sim(3)；small_gicp、Open3D、gtsam 分工；GNSS 用两阶段鲁棒 |
| r08 | UAV LIO ROS2 / 重定位 | `r08-uav-lio-ros2.md` | 引入 Session 与 Map Release；重定位状态机；MID-360 能力边界 h_max |
| r09 | PotreeConverter / PDAL / LAStools | `r09-potreeconverter-pdal-lastools.md` | numpy 切片器 `worldpkg tile`；定义 ANET_Q16；首屏 BFS 前缀用一次 Range 取回；PDAL 放 V0.5 Docker |
| r10 | 3D Tiles | `r10-3dtiles.md` | SSE 公式与参数；隐式瓦片客户端；3D Tiles 作为 GIS 与 Cesium 导出 |
| r11 | three.js WebGPU/TSL | `r11-threejs-webgpu.md` | 点只有 1 px；`gl_PointSize` 补丁；12 B/点布局；每对象 CPU 开销；PerfProbe 与 QualityController |
| r12 | potree / potree-core / three-loader | `r12-potree-core-loader.md` | PointCloudEngine 总体设计、参数分档、FPS 控制律；回退路径用经典 WebGLRenderer |
| r13 | Potree-Next / spark / supersplat | `r13-potree-next-splats.md` | 预算约束的贪心 LOD；统一页池加 DrawTable；reversed-Z；SVO 碰撞；渲染器生态割裂 |
| r14 | R3F / drei | `r14-r3f-drei.md` | "R3F 宿主 + 命令式引擎"（B′）；帧序契约；drei 白名单；遥测不进 React |
| r15 | Cesium / deck.gl / Foxglove | `r15-cesium-deckgl-foxglove.md` | 坐标规则 R1–R10；回放 Player；Hermite 插值；性能 HUD；foxglove-sdk 做调试旁路 |
| r16 | Web 天气视觉 | `r16-web-weather-fx.md` | 无状态降水、扁平四边形、解析高度雾、三档云、积分相位、质量档与预算 |
| r17 | WindNinja / OpenFOAM / OpenVDB | `r17-wind-cfd-vdb.md` | 风场分级 L0–L3；L2 质量守恒法移植（CPU 几秒到几十秒/方向）；按流向对齐的扇区插值 |
| r18 | 4DGS | `r18-4dgs.md` | 时间窗不透明度与回放 SoA 结构进入 MVP；动态外观重建放 V1.x |
| r19 | Prometheus（P600） | `r19-prometheus-amov.md` | P600 实际是 ROS1 + 私有地面站 TCP/UDP JSON 协议；移植控制权状态机与 Mock 动力学 |
| r20 | PX4 | `r20-px4.md` | SIH 容器多机；PX4-lite 控制级联移植并与 SIH 对照验证；端口与 namespace 规则；保真度阶梯 |
| r21 | MAVSDK / MAVROS / MAVLink | `r21-mavsdk-mavros-mavlink.md` | DroneAdapter 抽象；MAVSDK v4 原生 Python；命令微协议；NED<->ENU 闭式公式 |
| r22 | PX4 多机 / XTDrone | `r22-px4-multi-sim-xtdrone.md` | 每机一个容器的编排与生命周期状态机；编队律加前馈；SIH 风注入 |
| r23 | AirSim / gz-sim | `r23-airsim-gzsim.md` | 带减速约束的速度限幅与抗饱和；风力模型；gz world 导出器；AirSim 天气是反面教材 |
| r24 | MRS UAV | `r24-mrs-uav.md` | Safety & Health 模块、Flight FSM、围栏、电量 RTL、多机避让 |
| r25 | EGO / Fast-Planner | `r25-ego-fastplanner.md` | Planning Service：A* + B-spline + ESDF；三层互避；虚实混合 UDP 报文 |
| r26 | PX4 swarm controllers | `r26-px4-swarm-controllers.md` | 虚拟结构编队；CAPT 分配；区域覆盖规划；点云"扫描揭示"演示 |
| r27 | rosbridge / Foxglove / Zenoh | `r27-realtime-bridges.md` | 自研 rt 协议全规范；raw struct；credit 背压；zenoh 做内部总线（V0.5） |
| d01 | lieflat-charts | `d01-lieflat-charts.md` | ANet Graphite 色卡；图表规范；CPU canvas 加 SVG；表格用 table.log |
| d02 | transitions.dev | `d02-transitions-dev.md` | motion token；32 个配方；motion tier；遥测数字分 4 类处理 |
| d03 | morphicons | `d03-morphicons.md` | 静态 `<Icon>` 与动态 `<StateIcon>`；morph 白名单；并发预算 K=8 |
| d04 | shadcn/ui | `d04-shadcn-ui.md` | base-mira；`anet-base.json`；布局 block 组合；Base UI toast |
| d05 | ANet | `d05-anet.md` | 协作平面；一机一 AID；效果状态；合同网；品牌使用规范 |
| d06 | 渐进暴露 | `d06-progressive-exposure.md` | 传感器、边缘节点、完整智能体三档；四档耦合与搜救流程对照；复合体留在控制平面；四个可测实验 |
| x01 | UrbanScene3D | `x01-urbanscene3d-data.md` | 六城规范化矩阵（权威）；着色方案；Height_map；任务剧本 S1–S6 |
| n01 | 新 Web 点云 / 3DGS | `n01-discover-web-pointcloud.md` | voxelkloud 两级预算选择器、画质阶梯、cut 点径、单缓冲单 draw、compute 光栅 |
| n02 | 新重建 / SLAM | `n02-discover-recon-slam.md` | Engine Adapter v2；DA3-SMALL CPU 冒烟；RKO-LIO；CSF 规则语义 |
| n03 | 新仿真 / 集群 / Agent | `n03-discover-uav-sim-agents.md` | RotorPy 气动；Crazyflow pipeline；Dryden 按机建模；droneserver 守卫；CBBA |
| n04 | 新环境 / 孪生 | `n04-discover-environment-twin.md` | takram 大气只进 High 档；无状态流线虚线；LBM 瞬态风（L2.5）；moq QoS 语义 |
| n05 | 前端工程栈核实 | `n05-discover-web-stack.md` | 精确版本矩阵；Vite 8、TS 7、oxlint；trial 沙盘；性能测试基础设施 |

---
## 2. 全仓库选型矩阵

说明：
- stars 与"最后提交"来自各单元 2026-09-28 的实测。`—` 表示未查询或不适用。
- "落点模块"采用 §3.0 的统一目录命名。
- "版本"指落点里程碑。
- 结论写作"A + B"时表示分部分处理，例如整体 adopt、部分算法 port。

### 2.1 重建（Reality → Geometry）

| 仓库 | stars | 最后提交 | 定位 | 结论 | 落点模块 | 版本 |
|---|---|---|---|---|---|---|
| Robbyant/lingbot-map | 17140 | 2026-09-08 | 流式前馈视觉重建，主引擎，需 CUDA（估计峰值 16–20 GB） | adopt（GPU worker 以库方式调用，锁 commit 849e690） | `reconstruction/engines/lingbot` | V0.1 接口与 Mock；V0.5 真推理；V1.0 在线流式 |
| nerfstudio-project/viser | 2794 | 2026-09-24 | Python→Web 流式可视化 | port（混合二进制帧、合并缓冲、客户端插值、泄漏安全的 BufferGeometry）+ adopt（仅内部 Recon QA 查看器） | `apps/api/rt`、`tools/recon_qa` | V0.1 |
| facebookresearch/vggt | 14435 | 2026-05-18 | 前馈多视几何，接口规范源头（W2C） | port（pose_enc、518 预处理映射、COLMAP 导出、方向自检） | `reconstruction/engines/vggt` | V0.1 规范；V0.5 可选引擎 |
| colmap/colmap（pycolmap 4.2） | 12828 | 2026-09-27 | SfM 度量基准、Recon IR 骨架、地理配准、QA | adopt（CPU wheel） | `reconstruction/ir`、`reconstruction/georef`、`reconstruction/qa` | V0.1 IR；V0.5 基准 |
| cvg/glomap | — | — | 已并入 COLMAP 4.x（`global_mapper`） | skip | — | — |
| facebookresearch/map-anything | 3771 | 2026-08-07 | 以 RTK 位姿和 LiDAR 稀疏深度为条件的度量重建 | adopt | `reconstruction/engines/mapanything` | V0.1 定接口；V0.5 |
| ByteDance-Seed/Depth-Anything-3 | 6402 | 2026-07-27 | DA3-SMALL CPU 冒烟（8 帧 336px 约 27 s）；DA3-Streaming 离线长序列 | adopt | `reconstruction/engines/da3_cpu`、`jobs/recon_long` | V0.1 冒烟；V0.5 |
| facebookresearch/vggt-omega | 4588 | 2026-09-22 | VGGT 后继（权重需申请） | reference（经 MapAnything 包装评测） | — | V0.5 |
| yyfz/Pi3 | 2181 | 2026-05-18 | 无参考帧的前馈重建 | reference | — | V0.5 |
| MIT-SPARK/VGGT-SLAM | 1133 | 2026-06-29 | 开放词汇目标发现流程 | reference | — | V1.0 |
| DengKaiCQ/VGGT-Long | 903 | 2026-02-09 | 工程已并入 DA3-Streaming（教训：度量输出应做 SE3 对齐） | reference | — | V0.5 |
| zju3dv/Scal3R ／ HengyiWang/amb3r | 538 ／ 501 | 2026-09-15 ／ 2026-06-05 | 千米级、度量 VO | reference（观察） | — | V0.5+ |
| HengyiWang/amb3r-slam | 41 | 2026-09-17 | 尚未发布代码 | skip | — | — |
| CUT3R ／ TTT3R ／ StreamVGGT ／ InfiniteVGGT ／ MASt3R-SLAM | 1497 ／ 734 ／ 975 ／ 391 ／ 3193 | 2025-08 至 2026-05 | 与 LingBot-Map 角色重复 | skip | — | — |
| Tencent-Hunyuan/HunyuanWorld-Mirror | 1210 | 2026-05-27 | 条件化引擎的备选 | reference | — | V0.5 |
| OpenDroneMap/ODM | 6494 | 2026-09-15 | 经典航测基线，可解析 DJI SRT/EXIF | reference | `reconstruction/qa` | V0.5 |
| microsoft/MoGe | 2984 | 2026-08-19 | 单目度量先验 | reference | — | V0.5 |
| mistletoe235/OpenFlyScan | 2 | 2026-09-27 | "低质量区域补采"Agent 任务蓝本 | reference | — | V1.0 |
| DekuLiuTesla/CityGaussian | 1263 | 2026-08-16 | 大尺度 Visual World | reference | — | V1.0 |
| nerfstudio-project/gsplat | 5736 | 2026-09-19 | 3DGS 训练、导出、神经传感器渲染 | adopt（离线 GPU）+ port（节点量化、Morton、四元数打包、surfel 足迹） | `reconstruction/gaussian`、`world/pipeline` | V0.1 算法；V0.6–V1.0 训练 |
| nerfstudio-project/nerfstudio | 12032 | 2025-07-28 | 渲染状态机、transforms.json | reference | `engine/pointcloud/core/controller`（思路） | V0.1 |
| graphdeco-inria/gaussian-splatting | 24003 | 2024-10-30 | 3DGS PLY 字段契约、深度尺度对齐 | reference | `world/visual/gaussian/io` | 全程 |

### 2.2 LiDAR、定位与融合

| 仓库 | stars | 最后提交 | 定位 | 结论 | 落点模块 | 版本 |
|---|---|---|---|---|---|---|
| Livox-SDK/livox_ros_driver2 | 853 | 2026-09-21 | MID-360(S) 驱动；其数据契约作为 mock 标准 | adopt（锁 1.2.8 加本地补丁） | `packages/contracts/sensor_lidar`、Sensor Gateway | V0.2 契约；V0.5 真机 |
| Livox-SDK/Livox-SDK2 | 516 | 2026-09-21 | 协议真源 | adopt（间接依赖）+ port（codec、虚拟设备 SIL） | `sensors/lidar/livox_codec.py`、`simulation/sil` | V0.5 / V0.6 |
| hku-mars/LiDAR_IMU_Init | 1515 | 2026-04-30 | LiDAR-IMU 外参与时延标定 | adopt（ROS1 Docker）+ port（数学移植到 scipy） | `tools/calibration`、`services/calibration` | V0.5 |
| hku-mars/FAST_LIO | 5225 | 2024-07-23 | MID-360 LIO 事实标准 | adopt（机载）+ port（ikd 下采样规则、立方体迟滞、可观测性指标） | `reconstruction/lidar`、`engine/pointcloud` 驻留集 | V0.1 算法；V0.5 |
| hku-mars/FAST-LIVO2 | 4689 | 2026-03-08 | 带颜色的度量地图、COLMAP 导出 | reference（硬同步时有条件 adopt） | `reconstruction/fusion/colorize` | V0.5 / V1.0 |
| hku-mars/r3live | 2460 | 2022-08-25 | map delta 流式、贝叶斯着色 | port | `apps/api` 的 `lio.mapDelta` | V0.2 / V0.5 |
| isl-org/Open3D | 14005 | 2026-09-16 | 清洗、RaycastingScene Geometry World、配准、TSDF | adopt（0.20，需 libEGL） | `world/ingest`、`world/geometry`、`reconstruction/registration` | V0.1 / V0.2 / V0.5 |
| koide3/small_gicp | 1045 | 2026-08-31 | 下采样、GICP/VGICP、Sim(3)-GICP 基础 | adopt | `world/pointcloud/tiler`、`reconstruction/registration` | V0.1 / V0.5 |
| koide3/glim | 1848 | 2026-09-06 | 融合图架构蓝本 | reference | — | V0.5 |
| koide3/hdl_graph_slam | 2339 | 2024-07-16 | GPS 边、信息矩阵公式 | port（公式） | `reconstruction/fusion/graph` | V0.5 |
| koide3/fast_gicp | 1700 | 2025-04-24 | 已被 small_gicp 取代 | skip | — | — |
| liangheming/FASTLIO2_ROS2 | 761 | 2026-08-10 | 建图、PGO、HBA、重定位全链 | adopt（fork 后修补） | `reconstruction/lidar`、`localization` | V0.5a / V0.5b |
| Liansheng-Wang/faster_lio_localization | 212 | 2024-11-28 | iVox LRU、瓦片驻留调度 | port | `engine` 实时扫描层、`world/voxel` | V0.1–V0.5 |
| PRBonn/rko_lio | 660 | 2026-09-24 | pip 安装、无需 ROS 的离线 LIO（CPU 实测 p50 9–11 ms/扫描） | adopt（锁 0.4.0） | `tests/test_lio_mock.py`、`reconstruction/lidar/lio_job.py` | V0.2 / V0.5 |
| Liansheng-Wang/Super-LIO | 602 | 2026-07-13 | 机载 ROS2 LIO；OctVoxMap | adopt（机载）+ port（OctVoxMap） | `sim/mapping/octvox.py` | V0.2–V0.5 |
| jianboqi/CSF | 649 | 2026-09-11 | 布料模拟地面滤波 → DTM/HAG/规则语义 | adopt | `world/ingest/semantic_rules.py`、`world/geometry/terrain` | V0.1 |
| Pointcept/Pointcept ／ Pointcept/Utonia ／ IGNF/myria3d ／ meidachen/STPLS3D | 3237 ／ 751 ／ 294 ／ 290 | 2025–2026 | 学习型点云语义（需 GPU 与标注） | reference | — | V0.5+ |
| facebookresearch/sam3 | 11818 | 2026-09-18 | 开放词汇 2D 掩码提升到 3D | reference | — | V1.0 |
| hku-mars/Point-LIO ／ Swarm-LIO2 ／ MARSIM ／ UMI-3D ／ Voxel-SLAM | 1338 ／ 455 ／ 583 ／ 278 ／ 687 | 2025–2026 | 对照、多机、仿真、标定 | reference | — | V0.2–V1.0 |
| APRIL-ZJU/Gaussian-LIC ／ rpng/MINS ／ PRBonn/kiss-icp | 648 ／ 784 ／ 2330 | 2026 | LiDAR 3DGS、RTK 紧耦合、ICP 基线 | reference | — | V0.5–V1.0 |

### 2.3 World Package、切片与格式

| 仓库 | stars | 最后提交 | 定位 | 结论 | 落点模块 | 版本 |
|---|---|---|---|---|---|---|
| potree/PotreeConverter | 818 | 2026-09-23 | Potree 2.0 格式与算法标准（本机 GCC 13 无法编译 master） | port（numpy `worldpkg tile`，5M 点约 5–6 s）+ adopt（可选，>1 亿点时走 Docker） | `world/pointcloud/tiler.py` | V0.1 |
| PDAL/PDAL | 1416 | 2026-09-21 | 重投影、地面、DEM、HAG、着色、COPC | adopt（Docker `pdal/pdal`）；V0.1 用 numpy + pyproj 替代 | `world/georef`、`world/terrain` | V0.4–V0.5 |
| LAStools/LAStools | 1070 | 2026-09-21 | LAZ QA 与互操作（只用开源部分） | reference | `world/qa` | V0.5 |
| CesiumGS/3d-tiles | 2612 | 2026-08-17 | 3D Tiles 1.1 规范；SSE 语义 | adopt（作为**导出**规范；运行时不直接使用，见 §8 C3） | `world/export/tiles3d` | V0.5 / V1.0 |
| CesiumGS/3d-tiles-tools | 539 | 2026-07-24 | 隐式瓦片客户端代码、CLI（打包、合并、升级） | port（隐式客户端）+ adopt（CLI） | `world/packaging` | V0.5 |
| CesiumGS/3d-tiles-validator | 475 | 2026-09-16 | 导出产物的 CI 门禁 | adopt | `tools/ci/validate-world.sh` | V0.5 |
| CesiumGS/cdb-to-3dtiles ／ yeyan00/potree23dtiles | 92 ／ 58 | 2024-05 ／ 2021-10 | REPLACE 网格规则；Potree<->3D Tiles 子节点位序映射 | reference | — | V0.5–V0.6 |
| py3dtiles | 234 | 2026-09-18 | 其 1.1 输出不合法 | reference（仅做对照） | — | — |
| connormanning/copc.js + hobuinc/laz-perf + hobuinc/untwine | 63 ／ 103 ／ 78 | 2026-08 至 2026-09 | COPC 读写（归档与交换） | adopt | `engine/pointcloud/io/copc`、`world/export/copc` | V0.5 |
| visgl/loaders.gl（copc 模块） | 855 | 2026-09-26 | copc.js 的替代实现（v5 alpha） | reference | — | V0.5+ |

### 2.4 Web 3D 引擎与点云渲染

| 仓库 | stars | 最后提交 | 定位 | 结论 | 落点模块 | 版本 |
|---|---|---|---|---|---|---|
| mrdoob/three.js | 116010 | 2026-09-28 | 唯一核心渲染引擎（`three/webgpu`、`three/tsl`，也包含经典 WebGLRenderer） | adopt（`~0.186.1`） | `viewport/renderer.ts`、全部 engine | V0.1 起 |
| tentone/potree-core | 255 | 2026-09-14 | 库化 Potree 内核 | port（hierarchy、LRU、DEFAULT 解码、GLSL 点材质、EDLPass）+ adopt（dev 交叉验证页） | `engine/pointcloud/{io,core,render/glsl}` | V0.1 |
| pnext/three-loader | 285 | 2026-05-28 | TS 类型、有界 worker 池、3DGS LOD | reference / port（worker 池） | `engine/pointcloud/io/workerPool.ts` | V0.1 / V1.0 |
| potree/potree | 5622 | 2026-01-08 | 着色器数学、密度修正、HQ splat | reference | `engine/pointcloud/render` | V0.1 / V0.3 |
| voxelkloud-*（view/loader/core/format-*/react/wasm） | 0 | 2026-09-25 | 两级预算选择器、画质阶梯、流式策略、cut 点径、单缓冲单 draw、compute 光栅 | port（核心算法）+ adopt（`@voxelkloud/react` 仅作 dev 对照） | `engine/pointcloud/core/*`、`render/*` | V0.1 / V0.3 / V0.5 |
| Aurtechmx/openlidarviewer | 23 | 2026-09-28 | 帧预算 governor、自适应 DPR、Weyl 淡入、驱逐迟滞、后端探测 | port（策略函数） | `render/FrameGovernor.ts`、`engine/pointcloud/core/Fade.ts` | V0.1–V0.3 |
| NASA-AMMOS/3DTilesRendererJS | 2476 | 2026-09-28 | 3D Tiles 网格/地形底座；SSE、LRU、队列参考 | adopt（网格层）+ reference（公式） | `viewport/layers/TilesLayer.ts` | V0.1 公式；V0.8 |
| m-schuetz/Potree-Next | 124 | 2025-10-07 | WebGPU 点云技巧：reversed-Z、点 ID 拾取、GPU timestamp | port（技巧）；遍历逻辑不看预算，不照搬 | `render/camera.ts`、`engine/perf/gpuTimer.ts` | V0.1 / V0.2 |
| davbau/Potree-Next（Bauer 2025）／ m-schuetz/compute_rasterizer ／ SimLOD | 0 ／ 749 ／ 520 | 2026-09 ／ 2023 ／ 2024 | compute 光栅参数、算法出处 | reference | — | V0.3 / V0.6 |
| sparkjsdev/spark | 3660 | 2026-09-25 | 预算贪心 LOD、页池、3DGS 高保真 | port（算法）+ adopt（仅限独立 WebGLRenderer 预览路由） | `routes/splat-preview` | V0.1 算法；V0.8 |
| playcanvas/supersplat-viewer | 579 | 2026-09-26 | SVO 体素碰撞、compute 剔除 | port | `world/voxel/svo.py`、`engine/geometry/VoxelCollision.ts` | V0.4 / V0.6 |
| playcanvas/engine（gsplat-unified）／ playcanvas/splat-transform | 16944 ／ 1339 | 2026-09-28 | 全局预算性价比分配器；3DGS LOD 离线构建 | port ／ adopt | `engine/RenderBudgetArbiter.ts`、`pipeline/visual/splat_lod` | V0.6 / V0.8 |
| antimatter15/splat ／ manycoretech/aholo-viewer ／ Visionary ／ cg-tuwien/lidarscout | 3072 ／ 1076 ／ 526 ／ 27 | 2025–2026 | 3DGS 渐进、chunk LOD、动态高斯接口、LAZ 秒开 | reference | — | V0.5–V1.0 |
| WilliamLiu-1997/3D-Tiles-RendererJS-3DGS-Plugin ／ nianticlabs/spz ／ rsasaki0109/CloudAnalyzer | 128 ／ 924 ／ 14 | 2026 | 3DGS 与 3D Tiles 互操作；SPZ 格式；融合 QA | reference | — | V0.5–V0.8 |
| pmndrs/react-three-fiber v9 | 32582 | 2026-09-27 | 3D 视口宿主与组合层 | adopt（9.8.1） | `viewport/` | V0.1 |
| react-three-fiber v10 alpha ／ drei v11 alpha | — | 2026-09-26/27 | 原生 WebGPU、相位调度 | reference（peer 要求 react <19.3，等 rc 再迁移） | — | V0.3+ |
| pmndrs/drei | 9900 | 2026-09-25 | 选择性使用：CameraControls、GizmoHelper（renderPriority=2）、Html（只用于选中对象）、Detailed、meshBounds、Bvh、AdaptiveDpr、View | adopt（10.7.9 白名单） | `viewport/` | V0.1 |
| CesiumGS/cesium | 15781 | 2026-09-25 | 大地测量数学、点衰减、EDL、优先级、时钟 | port（约 150 行数学与 shader）+ adopt（可选，作为独立的地球 GIS 视图） | `engine/geo/wgs84.ts`、`views/globe` | V0.1 port；V0.5+ |
| visgl/deck.gl | 14615 | 2026-09-25 | TripsLayer 轨迹 shader、软范围过滤 | port（shader 思路）+ adopt（可选的 2D 态势图） | `engine/drones/trail-material.ts`、`panels/map2d` | V0.2 / V0.5+ |
| foxglove/studio | 92 | 2024-03（已归档） | 空仓库 | skip | — | — |
| lichtblick-suite/lichtblick | 1141 | 2026-09-25 | 回放 Player 状态机、渲染屏障、Panel API、M4 降采样、TF buffer | port（逻辑与接口，不用 MUI） | `time/player`、`net/pipeline.ts`、`panels/*` | V0.1–V0.2 |
| iTowns ／ GaussianSplats3D ／ gsplat.js ／ Reall3dViewer ／ web-splat ／ gaussian-splatting-webgpu ／ three-loader-3dtiles ／ Babylon.js ／ vtk-js ／ pygfx | — | — | 与本栈重叠、已停更或引擎不同 | skip | — | — |

### 2.5 环境视觉与物理风场

| 仓库 | stars | 最后提交 | 定位 | 结论 | 落点模块 | 版本 |
|---|---|---|---|---|---|---|
| Token-Gremlin/natural-disasters | 273 | 2026-08-31 | WebGL2/软件档天气算法主来源：无状态雨、GPGPU、Bayer 摊销云、质量闭环 | port（GLSL 改写为 TSL，另保留 GLSL 版本） | `engine/environment/*` | V0.3–V0.4 |
| SkyeShark/Eanpa-Sky | 56 | 2026-09-12 | 世界锚定降水、积分相位、确定性闪电、过渡、湿润 | port（算法与约定） | `engine/environment/*`、`services/sim/environment` | V0.3–V0.5 |
| CK42BB/procedural-weather-threejs ／ jeantimex/procedural-clouds | 11 ／ 6 | 2026-02 | 天气预设词汇与路由；compute 写 3D 缓存 | reference（代码有 bug，不照抄） | `environment/state/presets.json` | V0.3 / V0.5 |
| takram-design-engineering/three-geospatial | 1697 | 2026-05-27 | 物理大气与云（npm 0.19.1 与 r186 不兼容） | adopt（High 档，需 three r184 或修复）+ 离线烘焙天空全景 + port（云算法改写为 TSL） | `environment/atmosphere`、离线工具 | V0.3 烘焙；V0.5–V0.6 |
| mapbox/webgl-wind ／ NOC-OI/cesium-wind-layer ／ Rawcloud/cesium-wind-arrows-3d | 1108 ／ 0（上游 125）／ 0 | 2026 | 粒子更新公式、几何拖尾、3D 箭头 | port | `engine/environment/wind/WindViz.ts` | V0.3 |
| weatherlayers/weatherlayers-gl | 161 | 2026-09-28 | 按年龄分块的环形拖尾 | reference | — | V0.5 |
| anvaka/wind-lines 等（ektogamat、joshbrew、nff747） | 65 等 | 2026 | 无状态流线与 TSL 示例 | reference | — | V0.3 |
| firelab/windninja | 186 | 2026-09-22 | 廓线、粗糙度、质量守恒诊断风（L2） | port（移植到 Cartesian MAC + scipy/pyamg）+ adopt（CLI Docker，仅山地 DEM） | `environment/wind/{profile,turbulence,library}.py`、`tools/wind/build_l2_library.py` | V0.2 L0/L1；V0.3 L2 |
| OpenFOAM/OpenFOAM-dev | 2247 | 2026-09-25 | L3 RANS 扇区库（离线） | adopt（Docker，锁 OpenFOAM 13） | `tools/cfd/openfoam` | V0.6–V1.0 |
| AcademySoftwareFoundation/openvdb | 3418 | 2026-09-23 | 点云转水密网格与 SDF；.vdb 交换 | adopt（工具）+ port（采样器、RK4 流线） | `world/geometry/sdf`、`tools/export` | V0.5+ |
| NCAR/FastEddy-model | 129 | 2026-08-12 | LES（需 GPU） | reference | — | V1.0+ |
| arch85-km/urban-wind-lab | 1 | 2026-09-26 | D3Q19 LBM 瞬态城市风（L2.5） | port（numba 实测 3.8–5.8 MLUPS） | `environment/wind/lbm.py` | V0.4–V0.6 |
| rbischof/windinet ／ NVIDIA/physicsnemo ／ neuraloperator ／ thuml/Transolver | 12 ／ 3297 ／ 3918 ／ 415 | 2026 | 风场验收损失；3D 代理模型训练框架 | reference（验收指标可立即移植） | `environment/wind/validate.py` | V0.3 / V1.0 |
| RaymanNg/3D-Wind-Field ／ sakitam-fdd/wind-layer ／ cambecc/earth ／ leaflet-velocity ／ three-volumetric-clouds ／ procedural-clouds-threejs ／ deck.gl-particle ／ webgl-streamline-visualizer ／ WebGPU-Ocean | — | 多数停更 | 2D 地图或 demo 级 | skip | — | — |

### 2.6 动态世界（4D）

| 仓库 | stars | 最后提交 | 定位 | 结论 | 落点模块 | 版本 |
|---|---|---|---|---|---|---|
| fudan-zvg/4d-gaussian-splatting | 1038 | 2026-01-12 | 时间窗不透明度与时间预滤波；4D 切片 | port | `engine/materials/TemporalWindowNode` | V0.2 回放；V1.0+ |
| hustvl/4DGaussians | 3950 | 2024-10-27 | HexPlane；训练改走 gsplat.contrib.dynamic | reference | — | V1.0+ |
| JonathonLuiten/Dynamic3DGaussians | 2299 | 2023-12-22 | 固定槽位回放 SoA、恒速外推、OpenCV K 到投影矩阵 | reference + port（回放与内参相关部分进入 MVP） | `engine/replay/ReplayBuffer.ts`、`engine/camera/intrinsics.ts` | V0.2 |

### 2.7 仿真与飞控

| 仓库 | stars | 最后提交 | 定位 | 结论 | 落点模块 | 版本 |
|---|---|---|---|---|---|---|
| amov-lab/Prometheus | 3265 | 2025-11-21 | P600 软件栈；私有地面站协议（TCP 55555 / UDP 8889 / TCP 55556，JSON 加 CRC-16/ARC） | port（codec、控制权状态机、fake_uav 级联、健康规则） | `sim/backends/prometheus`、`sim/core/{authority,command,health}.py` | V0.1 语义；V0.2 模拟器；V0.5 真机 |
| PX4/PX4-Autopilot | 12707 | 2026-09-27 | SIH 容器多机、控制级联、轨迹平滑、状态语义 | adopt（`px4io/px4-sitl` 镜像）+ port（PX4-lite 级联，约 250 行 numpy） | `sim/fleet/px4lite.py`、`sim/backends/px4_sih` | MVP 控制律；V0.2 SIH；V0.4–V0.6 HIL |
| mavlink/MAVSDK | 939 | 2026-09-28 | v4 原生 Python（无 gRPC） | adopt（`mavsdk==4.0.0`，每进程不超过 16 机） | `sim/adapters/mavsdk` | V0.2 |
| mavlink/mavlink | 2442 | 2026-09-28 | 语义真值表、fake_px4 测试替身 | adopt | `sim/common/modes.py`、`tests/fake_px4.py` | V0.1 / V0.2 |
| mavlink/mavros | 1226 | 2026-09-27 | frame_tf、Router、timesync 滤波 | reference / port（公式） | `sim/common/frames.py` | V0.1 / V0.5 |
| robin-shaun/XTDrone | 1725 | 2025-08-02 | 编队、避让、命令语义、MID-360 扫描 CSV | port | `swarm/formation`、`assets/lidar` | V0.5 / V0.6 |
| px4io/px4-sitl 镜像 | — | 2026-09-27 | 每机一个容器（`px4 -i 0`，sysid 由参数注入） | adopt | `sim/orchestrator/drivers/docker_sih` | V0.2 |
| AntonSHBK/px4_multi_drone_sim ／ TannerGilbert/PX4-Multiagent-Simulation ／ andy-zhuo-02/XTDrone2 | 26 ／ 11 ／ 115 | 2025–2026 | 轨迹原语与命令队列 ／ gz 传感器宏 ／ session schema（三者都有已知 bug） | port ／ reference ／ reference | `sim/mission/primitives.py` | V0.1 / V0.6 |
| gazebosim/gz-sim | 1520 | 2026-09-25 | Level C 后端、世界导出、阵风发生器、Lee 控制器 | adopt（仅 GPU 节点，或 headless 纯物理）+ port（阵风、Lee、旋翼阻力） | `sim/gazebo/exporter`、`environment/wind` | V0.2+（导出）；V0.5+ |
| microsoft/AirSim | 18516 | 2026-09-15（已归档） | carrot 跟随、watchdog、时钟、FastPhysics | port（算法）；天气设计是反面教材 | `sim/mock/*`、`sim/clock` | V0.1 / V0.3 |
| ctu-mrs/mrs_uav_system（总仓）+ 子仓 | 640（总仓） | 2026-08 至 09 | 多机科研栈；子仓 `mrs_multirotor_simulator`（向量化动力学）、`mrs_uav_managers`（安全阈值）、`mrs_uav_trackers`（LineTracker）、`mrs_lib`（围栏）、`status/autostart` 可移植；`mrs_uav_controllers` 为参考；`mrs_mpc_solvers` 跳过 | reference（总仓）+ port（子仓） | `sim/safety/*`、`sim/reference/line_tracker.py` | V0.1–V0.6 |
| spencerfolk/rotorpy | 311 | 2026-09-07 | 气动力矩（风仅通过空速进入）、Dryden、SE3 控制、HIL 客户端 | port（numpy SoA 版本，L2） | `sim/fleet/aero.py`、`environment/wind/turbulence.py` | V0.3–V0.6 |
| learnsyslab/crazyflow | 180 | 2026-09-26 | SoA 加命名 step pipeline（单一世界时钟）；so_rpy x500 参数 | port（架构） | `sim/fleet/fleet.py` | MVP 起 |
| PegasusSimulator/PegasusSimulator | 887 | 2026-07-24 | Backend 接口、传感器噪声参数、HIL lockstep | port（接口与参数） | `sim/backends/base.py`、`sim/sensors/noise.py` | V0.2 / V0.4 |
| learnsyslab/gym-pybullet-drones | 2146 | 2026-09-06 | 下洗流与地效公式 | port | `sim/fleet/interaction.py` | V0.6 |
| alireza787b/mavsdk_drone_show | 325 | 2026-09-25 | Boustrophedon 覆盖规划 | port（贪心分配改为 Hungarian） | `swarm/coverage/boustrophedon.py` | V0.6 |
| isaac-sim/IsaacLab ／ genesis-world ／ aerial_gym_simulator ／ Cosys-AirSim ／ PX4/Hawkeye ／ OpenFly ／ rl-tools ／ quad-swarm-rl ／ crazyswarm2 | 8240 ／ 29992 ／ 776 ／ 433 ／ 84 ／ 369 ／ 1036 ／ 239 ／ 259 | 2025–2026 | GPU 或 RL 路线、回放 UI、评测数据生成 | reference | — | V1.x |
| OmniDrones ／ flightmare ／ agilicious ／ CrazySim ／ lsy_drone_racing | — | 停更或不相关 | — | skip | — | — |

### 2.8 多机、规划与任务分配

| 仓库 | stars | 最后提交 | 定位 | 结论 | 落点模块 | 版本 |
|---|---|---|---|---|---|---|
| ZJU-FAST-Lab/ego-planner-swarm | 2193 | 2025-03-08 | B-spline 轨迹契约、FSM、swarm 代价、跨机 UDP 报文 | port（master）+ adopt（ros2_version 分支，Docker 部署的可选规划后端） | `sim/planning/*`、`sim/planning/backends/ego_*` | V0.2 契约；V0.6 |
| HKUST-Aerial-Robotics/Fast-Planner | 3414 | 2024-10-24 | ESDF 距离代价、EDT、states2pts、时间拉伸 | port（scipy EDT 加 L-BFGS-B） | `sim/planning/{grid,bspline}.py` | V0.2 |
| spirit0609/Fast-LIO2_Ego-Planner ／ ZJU-FAST-Lab/Primitive-Planner | 1 ／ — | 2024-12 ／ 2026-08 | 真机链路与外部目标协议 ／ 百机规模后继 | reference | — | V0.5 / V1.0 |
| artastier/PX4_Swarm_Controller | 97 | 2024-03-02 | PrC 拓扑权重、一致性律（有 bug）、offboard 生命周期 | port（修复后作为对照律） | `swarm/topology.py`、`swarm/formation/laws.py` | V0.6 |
| Apoorv-1009/PX4-Aerial-Swarm-Reconstruction | 16 | 2024-12-11 | 多机城市扫描场景、屏障同步、高度分层 | reference | `swarm/mission/executor.py` | V0.6 |
| zehuilu/CBBA-Python ／ keep9oing/consensus-based-bundle-algorithm | 130 ／ 45 | 2021 ／ 2022 | 去中心化分配（需修正 DMG） | port ／ reference | `swarm/allocation/cbba.py` | V1.0 |
| google/or-tools ／ PyVRP/PyVRP | 14115 ／ 702 | 2026-09 | 带电量约束的多机 VRP | reference（先做 PoC） | — | V0.6 |
| aerostack2/aerostack2 ／ nubot-nudt/dynamic_task_allocation | 390 ／ 164 | 2026-09 ／ 2024 | ROS2 行为架构 ／ ROS1 停更 | reference ／ skip | — | — |

### 2.9 实时通信与后端基础设施

| 仓库 | stars | 最后提交 | 定位 | 结论 | 落点模块 | 版本 |
|---|---|---|---|---|---|---|
| eclipse-zenoh/zenoh | 3216 | 2026-09-15 | 内部总线；liveliness 与 queryable 用于能力发现；key-expr 命名语法 | reference（V0.1 起沿用命名与 QoS）+ adopt（V0.5 起，`eclipse-zenoh 1.10.x`） | `apps/api/rt/bridges/zenoh_bus.py`、`agent/anet` | V0.1 / V0.5 / V1.0 |
| foxglove/ws-protocol | 150 | 2025-07（已归档） | channel 广告、订阅、TIME、Playback、懒生产 | port（语义；不采用其线格式） | `apps/api/rt/protocol` | V0.1 |
| foxglove/foxglove-sdk | 311 | 2026-09-24 | 调试旁路与 MCAP 录制 | adopt（`foxglove-sdk==0.27.0`） | `apps/api/rt/recorder.py`、`bridges/foxglove_debug.py` | V0.2 |
| RobotWebTools/rosbridge_suite | 1246 | 2026-08-17 | P600 ROS1 适配器（ros1 分支 0.11.18 支持 cbor）；自研 asyncio 客户端，不用 roslibpy | adopt（仅作适配器） | `apps/api/rt/bridges/rosbridge_client.py` | V0.5 |
| eclipse-zenoh/zenoh-ts（remote-api） | 49 | 2026-09-15 | 浏览器直连总线（反例） | skip | — | — |
| moq-dev/moq | 1546 | 2026-09-28 | QoS 语义（priority/order/maxAge）、Snapshot 轨道；WebTransport 中继 | port（语义）+ adopt（可选 moq-relay） | rt 协议的 advertise 元数据、`net/rt/transport.ts` | V0.2 / V1.0 |
| wtransport/pywebtransport ／ aiortc/aioquic ／ BiagioFesta/wtransport | 42 ／ 2006 ／ 709 | 2026 ／ 2025-10 ／ 2026 | Python WebTransport 不可用；Rust 备选 | skip ／ skip ／ reference | — | V1.0 |
| rerun-io/rerun | 11503 | 2026-09-27 | 研发记录仪（`.rrd`）；迟到订阅者缓冲语义 | adopt（仅 dev sidecar 与 `/debug` 路由） | `tools/rerun_sidecar.py` | V0.2 |
| eclipse-ditto/ditto | 928 | 2026-09-25 | reported/desired 状态模型、能力定义 | reference | DroneState 模型设计 | V0.2 / V1.0 |
| ertis-research/opentwins ／ iTwin/itwinjs-core ／ digitaltwincityviewer | 276 ／ 732 ／ 18 | 2025–2026 | 过重或不对口 | skip | — | — |

### 2.10 Agent 与 ANet

| 仓库 | stars | 最后提交 | 定位 | 结论 | 落点模块 | 版本 |
|---|---|---|---|---|---|---|
| ANetResearch/ANet | 6 | 2026-09-06（本地有 2026-09-27 未推送的 v0.2 检查点） | 身份、能力发现、签名委派、证据；效果状态模型；TSIR 验收；黑板 | port（语义，V0.6 做 Mock）+ adopt（V1.0：每机一个 daemon + 自建 hub，service 模块接入） | `agent/runtime`、`agent/anet_bridge`、`agent/capabilities`、`tools/anet` | V0.2 效果状态；V0.6；V1.0 |
| XDEI-Group/AerialClaw | 132 | 2026-07-08 | SkillSpec、单步闭环、审批分级 | port（语义） | `agent/runtime/loop.py`、`agent/skills` | V1.0 |
| PeterJBurke/droneserver | 7 | 2026-09-20 | "LLM 视为不可信指挥官"的守卫管线：分级、一次性确认令牌、围栏前推、审计 | port | `apps/api/gateway/guard.py`、`agent/mcp` | V0.6 / V1.0 |
| robotmcp/ros-mcp-server ／ ion-g-ion/MAVLinkMCP ／ typefly/TypeFly ／ learnsyslab/swarmGPT ／ naver-ai/DroneCATS | 1479 ／ 24 ／ 117 ／ 32 ／ 2 | 2026 | LLM 接入模式参考 | reference | — | V1.x |
| alireza787b/dronesphere ／ deepak61296/mavlink-mcp | 14 ／ 4 | 2025 ／ 2026 | 已被取代，或社区规模可忽略 | skip | — | — |

### 2.11 设计体系

| 仓库 | stars | 最后提交 | 定位 | 结论 | 落点模块 | 版本 |
|---|---|---|---|---|---|---|
| larashero3-dotcom/lieflat-charts | 5754 | 2026-09-05 | 图表与表格视觉语言（61 个模板、token、table.log） | port（全部重写为 React 组件；不引入 ECharts、Chart.js、Recharts） | `ui/lf/*`、`lib/lf/tokens.ts`、`scripts/lint-lf.mjs` | V0.1–V1.0 |
| Jakubantalik/transitions.dev | 4382 | 2026-09-21 | motion token 与 32 个免费配方；refine 算法；Pro 源码不可用 | adopt（token 与配方 CSS）+ port（hooks）+ reference（refine，用于 motion-lint）+ skip（Pro、CLI） | `styles/motion/*`、`ui/motion/*` | V0.1 / V0.2 |
| guillermolg00/morphicons | 2705 | 2026-08-28 | 图标 morph 引擎（2026 新项目） | adopt（精确锁 1.7.1）+ port（Procrustes，用于编队预览） | `ui/icons/*`、`swarm/formation-preview.ts` | V0.1 / V0.6 |
| lucide（数据包 1.48.0） | — | 2026-09-24 | 唯一的图标几何来源 | adopt（精确锁版本） | `ui/icons/registry.ts` | V0.1 |
| lucide-react 1.48.0 | — | 2026-09-24 | 组件包 | 按 §8 C7 裁决：只保留在 shadcn 组件内部（与 lucide 版本一致），应用层不使用 | — | V0.1 |
| shadcn-ui/ui（CLI 4.21.0，base-mira） | 124743 | 2026-09-28 | 全量 UI 组件（Base UI） | adopt（组件源码提交进仓库；`anet-base.json` 一次完成 init） | `ui/components/ui/*` | V0.1 |

### 2.12 前端工程栈（n05 实测版本）

| 包 | stars | 版本 / 日期 | 结论 | 说明 |
|---|---|---|---|---|
| react / react-dom | 250798 | 19.3.0（2026-09-09） | adopt | 与 R3F 9.8.1 兼容（peer <19.4）；R3F v10 alpha 要求 <19.3 |
| vite + @vitejs/plugin-react | 83061 | 8.3.1 / 6.1.1 | adopt | Rolldown 构建约 1 s；需配置 COOP/COEP 响应头 |
| typescript | 111253 | 7.0.2（Go 原生） | adopt（只用作 tsc） | 没有 JS API；需要 TS API 的工具走 `@typescript/typescript6` 别名 |
| oxlint + oxlint-tsgolint | 22904 | 1.86.0 / 7.0.2003 | adopt | type-aware lint；用 `no-restricted-imports` 约束 engine 与 net 层边界 |
| tailwindcss + @tailwindcss/vite | 97723 | 4.3.3 | adopt | 与 shadcn 配套 |
| zustand | 58763 | 5.0.15 | adopt | UI 摘要态（4–10 Hz） |
| @tanstack/react-query | 50371 | 5.104.0 | adopt | REST 服务端状态 |
| @msgpack/msgpack | 1558 | 3.1.3 | adopt（控制面与低频数据） | 与 Python msgpack 互通；高频数据用 raw struct |
| msgpackr | 695 | 2.1.0 | reference | records 扩展与 Python 不互通 |
| @playwright/test | 96804 | 1.63.0 | adopt | 用 `executablePath` 指向本地 Chrome 151 |
| vitest + @vitest/browser-playwright | 17167 | 5.0.2 | adopt | unit、browser、bench 三个项目 |
| @pmndrs/scheduler | 8 | 0.2.0（2026 新仓库） | adopt（V0.2）+ port（fixed-step） | R3F v10 的帧循环内核 |
| stats-gl | 280 | 4.2.3 | adopt | 无 DOM 的 GPU 计时数据源 |
| partysocket | 1276 | 1.3.0 | port（重连逻辑，约 60 行） | 默认值需改为 arraybuffer 与有界队列 |
| react-scan | 21858 | 0.5.7 | adopt（仅 dev） | 发现重渲染风暴 |
| msw | 18227 | 2.15.0 | adopt（V0.2，可选） | 离线演示与测试 |
| koota ／ detect-gpu ／ TanStack/router ／ biome ／ rolldown | 744 ／ 1213 ／ 15136 ／ 25871 ／ 13956 | 2026 | reference | 各自的触发条件见 n05 |
| r3f-perf ／ comlink ／ jotai ／ TanStack/pacer | 781 ／ 12797 ／ 21284 ／ 777 | — | skip | 已停更、用不上或与现有方案重复 |

### 2.13 演示数据

| 资产 | 定位 | 结论 | 落点 | 版本 |
|---|---|---|---|---|
| UrbanScene3D 六城采样点云（`data/raw/urbanscene3d/*_sampled_5m.ply`，每城约 5M 点，xyz 加法线，无颜色） | 内置演示世界，同时是回归与性能基准 | adopt（必须先规范化） | `datasets/urbanscene3d`、`world/ingest/urbanscene3d.py` | V0.1 |
| UrbanScene3D 航线文件（`paths/*`、`Oblique.log`，UE 厘米坐标、左手系） | 视点分布与云台参数模板、回放测试集 | port（连接段需重新规划） | `datasets/urbanscene3d/paths.py` | V0.2 / V0.6 |
| Linxius/UrbanScene3D `src/`（156 stars，2023-12） | Height_map、重建评测 | port（numpy 约 40 行；评测需修 4 处 bug） | `sim/world/heightmap.py`、`world/qa/recon_eval.py` | V0.2 / V0.5 |
| Simulator.zip ／ Evaluation.zip ／ polytech1k、artsci1k | UE4 工程、Windows exe、1k 点预配准集 | skip | — | — |

---
## 3. 按模块的最终推荐

### 3.0 统一落点目录（消除各笔记落点命名不一致）

各单元给出的目标路径不一致，例如 `apps/web/src/engine/pointcloud`、`apps/web/src/world/pointcloud`、`world/pointcloud`、`services/sim`、`apps/simulator`。下文和后续文档统一使用以下命名：

```text
anet-drone/
├── apps/
│   ├── web/src/
│   │   ├── ui/            # shadcn 源码(components/ui)、lf 图表、motion、icons、面板(panels)、布局
│   │   ├── viewport/      # R3F 宿主 + 薄适配层(每个 ≤150 行)、renderer.ts(RenderBackend 选择)、layers/
│   │   ├── engine/        # 命令式、无 React 依赖：pointcloud/ drones/ environment/ labels/ picking/ perf/ geo/ camera/ time/ loop.ts
│   │   ├── net/           # rt.worker.ts、codec/(raw/msgpack)、api.ts(TanStack Query)
│   │   ├── stores/        # zustand vanilla store(4–10 Hz 摘要态、选择集、偏好)
│   │   └── styles/        # tokens.css、motion/、lf.css
│   ├── web/tests/ web/perf/  # vitest(unit/browser/bench)、playwright 流畅性用例
│   └── api/               # FastAPI：REST、rt/(Gateway：协议、调度、会话、录制、桥接)、静态 World 服务(Range)
├── sim/                   # 仿真内核(独立进程)：fleet/(FleetSim 与 pipeline)、backends/(mock/px4_sih/prometheus/replay)、
│                          # core/(authority/command/health/frames)、safety/、planning/、sensors/、orchestrator/
├── environment/           # E(x,y,z,t)：wind/(L0–L3、湍流、library)、weather/(presets/derive/transitions/lightning)、field.py
├── world/                 # 离线与查询：ingest/、pointcloud/(tiler/format)、georef/、terrain/、geometry/(dsm/collision/sdf)、voxel/、semantic/、qa/、export/
├── reconstruction/        # ir/、engines/(lingbot/vggt/da3/mapanything/colmap/mock)、georef/、lidar/、registration/、fusion/、gaussian/
├── agent/                 # runtime/(tasks/allocator/tsir/blackboard/evidence)、capabilities/、anet_bridge/、mcp/
├── swarm/                 # formation/、coverage/、allocation/、topology/
├── packages/contracts/    # 唯一真源：rt/layouts.json、env/presets.json、schemas(world/coordinate/drone_state/command)
├── datasets/urbanscene3d/ # 数据适配器与航线解析
├── scenarios/             # 任务剧本 JSON(S1–S6)
├── vehicles/p600/         # Vehicle Package：model.yaml、params.yaml、sensors/、calib/
├── worlds/<id>/           # World Package(生成物，不入库)
└── tools/                 # shadcn 镜像、codemods、lint-lf、motion-lint、rerun sidecar、ci
```

### 3.1 跨模块契约（所有模块共同遵守）

| 契约 | 规定 | 依据 |
|---|---|---|
| 世界坐标 | World ENU（X 东、Y 北、Z 上），米，右手系。CPU 与服务端用 float64；GPU 只放节点局部量化值或局部 ENU float32（10 km 内误差 ≤0.7 mm） | r15 R1–R10、x01 §3.5、r09、r11 |
| 渲染坐标 | Three.js 场景保持 Y-up；WorldLayer 根节点整体 `rotation.x = −π/2`，内部全部是 ENU；映射为 three(x,y,z) = (E, U, −N)（裁决见 §8 C4） | x01 §3.5(c)、r14、r10 |
| 机体与传感器 | 机体 FLU（REP-103）；四元数 `[x,y,z,w]`，表示 WORLD←BODY；相机 optical 为 RDF；three 相机相对 FLU 的旋转为 `R_flu_cam = [[0,0,−1],[−1,0,0],[0,1,0]]` | r15、r21、x01 |
| PX4 边界 | `p_ned = (N, E, −U)`；`q' = (1/√2)(w+z, x+y, x−y, w−z)`（这是一个对合变换）；`yaw_ned = 90° − ψ_enu = heading`。只在网关换算 | r21（数值验证）、r23、x01 |
| 高程 | 三个字段并存：`h_ellipsoid`、`h_msl`、`z_world`；另加 `agl = z − dtm(x,y)` | r15 R3、n02 |
| 时间 | 线上和存储一律 int64 `t_sim_ns`；渲染用 float64 秒（相对会话起点）；GPU 用 float32 秒（相对块起点，块 ≤2 h）；绝对纳秒不能放进 JS Number | r15 R8、r04、r27 |
| 单位 | 米、m/s、弧度；航向显示用"北为 0、顺时针"；IMU 加速度在网关统一换算为 m/s²（MID-360 原始单位是 g） | r04、r15 |
| 命名 | 使用 zenoh key-expression 子集：`uav/p600-01/state`；ROS 名为 `/` + key，其中 `-` 换成 `_` | r27 §3.6 |
| 可视化与物理分离 | 碰撞、规划、传感器只读 Geometry World（全分辨率 DSM、体素、SDF），**绝不读 LOD 瓦片或 3DGS** | r03、r09、r13 |
| 浏览器职责 | 浏览器 GPU 计算只服务可视化（粒子、流线、剔除、排序），结果不回流物理 | r11、r13、01-design §13 |
| 未知值 | 二进制通道写 NaN；JSON 通道写 `null` | r27 |

### 3.2 Reconstruction（视频 → 几何）

- **最终选型**：采用 Reconstruction Engine Adapter v2（n02 §3.1，在 r02 Recon IR 基础上扩展），四类引擎共用一个输出契约。
  - LingBot-Map：在线与流式，主引擎。
  - DA3-Streaming：离线长序列加回环。
  - MapAnything：以 RTK 位姿加 LiDAR 稀疏深度为条件的度量重建，V0.5 与后配准方案做 A/B。
  - COLMAP 4.2：度量基准、兜底与 QA。
  - Mock：本机与 CI。
- **统一输出契约**：
  - 位姿：`T_world_cam`（C2W）加 `q_xyzw`。适配器负责把 VGGT、VGGT-Ω、DA3 输出的 W2C 求逆，并把 DA3 的 saddle 参考帧重新规范到第 0 帧，CI 中做方向自检。
  - 置信度：u8，按 `conf_u8 = round(255·(1−1/conf))` 计算。
  - 附带 `scale_status`，取值 `relative | gnss | rtk | lidar`，UI 必须显示。
  - `engine.json` 记录引擎、版本、权重和许可。
- **地理配准从 V0.5 前移到 V0.1**：做法是 RANSAC-Umeyama Sim3 对齐 GNSS/RTK 的 ENU 轨迹。直线航带要做共线退化检测并用 IMU 重力增强：实测纯 Umeyama 旋转误差 60–117°，增强后为 0.19°。质量门禁为内点率 <60% 或 RMSE >5 m 时转人工确认。
- **MVP 形态**：
  - Mock Reconstruction Engine（r01）：用 UrbanScene3D 加程序化航线做 z-buffer splat，按真实 worker 的输出契约产出结果，可注入逐窗口的尺度或 yaw 漂移。
  - 可选的 DA3-SMALL CPU 冒烟：8–16 帧生成真实点云，实测 8 帧 504 px 约 37.5 s，峰值 RSS 1.9 GB。
  - Job 状态机：`QUEUED→PREPARING→SEGMENTING→INFERRING→FUSING→GEOREFERENCING→TILING→PACKAGING→SUCCEEDED/FAILED/CANCELLED`，每个阶段幂等，完成时写 `.complete` 标记，进度事件节流到 250 ms。
- **V0.5 起**：GPU worker 用 Docker 部署，要求 24 GB 以上显存、CUDA 12.8，FlashInfer 可选。长视频按帧流式读取，逐帧落盘。天空掩码需要修复补零 bug，并把 skyseg.onnx 打进镜像。
- **来源单元**：r01、r02、r03、n02。

### 3.3 LiDAR 与融合（V0.5 为主，契约在 MVP 冻结）

- **LiDAR 契约**：从第一天起采用 Livox 的数据语义。
  - 时间：`(sec,nsec)` 时间基加 u32 `offset_ns`。
  - 点：FLU 系 f32 xyz（米），加 u8 反射率、tag、line。
  - 帧：10 Hz，对齐绝对 100 ms 栅格；IMU 200 Hz。
  - 这样真 P600 替换 mock 时，前端和后端都不需要改（r04）。
- **虚拟 MID-360**：
  - 扫描模式：参数化非重复扫描，FOV 为 −7° 到 +52°（r04 §3 给出拟合常数）。
  - 光线求交：V0.2 用 Open3D RaycastingScene（DSM 网格加 BVH，20k 射线约 6 ms，r06）；备选是 numpy 球面 z-buffer（9–80 ms，r04/r05）。
  - 物理效应：Beer-Lambert 双程衰减，σ 来自 E 场；最大量程随反射率变化 `r_max = 70·(ρ/0.8)^0.269`；杂波按 Livox tag 打标。
  - 挂载：必须建模挂载预设。正装时在 30 m AGL 只有约 34% 的射线命中，倒装约 75%；正装加 20° 前倾时建图高度上限约 18 m（r08）。
- **融合链路**：采用轨迹优先。
  1. LIO 加 RTK 做两阶段鲁棒的 GNSS 因子图（Huber → χ² 门限 → L2），得到 ENU 轨迹。
  2. 视觉轨迹与它做 Sim3 对齐。
  3. 用 Sim(3)-GICP 金字塔 [2,1,0.5] m 精化（必须带鲁棒核）。
  4. 用 gtsam 4.3 做联合因子图。
  5. 融合，然后 QA。

  工具分工：small_gicp 负责预处理、GICP 和 Sim3-GICP；Open3D 负责 FPFH+RANSAC、QA 和 RaycastingScene；gtsam 负责因子图。PCL 和 cuVSLAM 移出融合链路（r06、r07）。
- **V0.5 拆分**：
  - V0.5a：建图 Session（LIO+PGO+HBA+ENU 对齐），产出 Map Release。验收：RTK 下 ATE ≤5 cm。
  - V0.5b：重定位（瓦片加载、状态机、PX4 EV）。
  - V0.5c：视觉-LiDAR 融合 QA 与人工对齐闭环。

  机载只跑 FAST-LIO2 或 Super-LIO；后端离线 LIO 用 RKO-LIO（pip 安装，不依赖 ROS）。
- **MVP 附带**：mock MID-360 加 IMU 合成，再加 RKO-LIO 回归测试。实测 424 m 轨迹 ATE 0.12 m、1410 m 为 0.39 m，只能作为回归门禁，不能代表真实精度。可选 CPU 演示"伪 V0.5"：实时建图增量（int8 体素块，约 3 B/点）加 LIO 健康面板。
- **来源单元**：r04、r05、r06、r07、r08、n02。

### 3.4 World Package 与切片

- **目录**：在 01-design §41 基础上补齐。

```text
worlds/<id>/
├── world.json                    # manifest：版本、层、统计、工具参数、校验和、LOD 参数
├── coordinate.json               # 锚点(kind: rtk|survey|synthetic)、datum、geoid、enu_to_ecef、
│                                 # source{unitsToMeters, upAxis, leveledDeg, yawDeg, T_enu_src, evidence}、ground、northConfidence
├── geometry/
│   ├── pointcloud/source/        # 全分辨率 PLY/LAZ(物理权威)
│   ├── terrain/                  # dtm_*.f32、dsm_2m.npz、terrain.json
│   ├── collision/                # DSM 网格、简化网格、index.json
│   ├── voxel/                    # 占据键(numpy sorted int64)、可选 occupancy.voxel.{json,bin}(SVO)
│   └── sdf/                      # ESDF chunk(V0.6)
├── visual/
│   ├── pointcloud/               # Potree 2.0 容器：metadata.json(+anet 扩展)、hierarchy.bin、octree.bin(ANET_Q16)、hierarchy_ext.bin
│   ├── copc/                     # 归档(V0.5)
│   ├── tiles3d/                  # 导出(V0.5/V1.0)
│   └── gaussian/                 # manifest、master、web/spz、archive(V0.6+)
├── semantic/                     # zones.geojson(禁飞区/限制区)、footprints.geojson、类别统计
├── localization/                 # 重定位瓦片、manifest(V0.5b)
├── environment/                  # wind/(manifest + 扇区库 f16+zstd + solid mask)、presets、atmosphere(天空全景)、noise/*.bin
├── reconstruction/<session>/     # engine.json、session.json、cameras、traj_c2w.txt、trajectory.bin、alignment.json、qa.json、lidar/<session>/
├── recordings/                   # MCAP(仿真与真飞)
├── missions/  scenarios/         # 任务 spec/plan/logs、剧本
├── derived/                      # 规划栅格 plan_r{res}.npz、nofly_*.bin
└── qa/                           # ingest QA、registration.json、metrics.json、report.json
```

- **切片器**：`worldpkg tile` 是 numpy 移植版（r09）。
  - 做法：一次 Morton 排序，每层用 `reduceat` 做网格竞选，中心优先；加法式 LOD；G=64，LEAF=20000，叶节点小于约 5k 点时并入父节点。
  - 耗时：5M 点单线程 5–6 s。
  - 节点内用固定种子 Fisher–Yates 打散点序，因此任何前缀都是均匀子采样。
  - 写出 `levelsByteEnd`，保证首屏 L0–L2 能用一次 Range 请求取回（NY 为 181,872 点、2.18 MB）。
  - 写出 `hierarchy_ext.bin` 紧包围盒，避免城市扁平数据被过度细分。
- **ANET_Q16 v1**：
  - 位置：`unorm16x4`，其中 xyz 为节点局部坐标，w 为 oct16 法线。
  - 颜色：`unorm8x4`，rgb 加 1 字节 class（LAS 编码：地面 2、立面 64、屋顶 65、其他 1、水面 9）。
  - 合计 12 B/点；两个后端可以共用同一份 buffer（WebGPU 只支持 x2/x4 顶点格式）。
  - **禁止用 float16 存世界坐标**：实测在纽约尺度误差 1 m，在上海尺度误差 4 m。
- **ingest 规范化**：`world/ingest/urbanscene3d.py`，x01 §3.3 为权威配置，下表中的矩阵已实测。

| 城市 | 单位→米 | 上方向 | 调平 | yaw | 北向 | 特别处理 | 首屏层级 |
|---|---|---|---|---|---|---|---|
| Shenzhen | 1 | +Z | 否 | 0 | 未验证 | 17% 法线朝下需翻正；默认演示城市 | L≤2 |
| Shanghai | 1 | +Z | 否 | 0 | 已验证 | 24% 法线朝下，0.15% 零法线 | L≤3 |
| New York | 1 | +Z | 否 | 0 | 已验证 | 立面占 52% | L≤2 |
| San Francisco | **10.15** | +Z | 否（地形真实起伏，禁止调平） | **+90°** | 已验证 | 着色必须用 HAG 模式 | L≤2 |
| Suzhou | 1 | **+Y** | 否 | 0 | 未验证 | 无地面，需合成 z=0 地面；建议多根森林 | L≤4 |
| Chicago | **1000** | +Z | **是（2.03°）** | 0 | 已验证 | — | L≤3 |

  流程顺序：单位 → 上方向 → 调平 → 北向 → 原点（XY 取包围盒中心，Z=0 取 DTM 中位数）→ 法线修正（faceforward 加水平面翻正）→ DTM（10 m 栅格取最小值，再做 9×9 开运算）→ HAG → 规则分类 → 写出 `coordinate.json` 与 `qa.json`（门禁：最高建筑应在 100–700 m、地面平面残差、倾角、地标证据）。
- **语义**：MVP 用 CSF 加 HAG 加法线规则（n02，NY 5M 点 112 s）。分类存 1 B/点，GPU 用 classMask uniform 控制图层开关，不需要重新加载。
- **服务**：FastAPI `StaticFiles` 直接服务，已验证支持 Range。要求 CORS、`Cache-Control: immutable`，目录带版本号。
- **来源单元**：r09、r10、x01、n01、n02、r03、r15。

### 3.5 Web 点云引擎（PointCloudEngine）

- **架构**（r12 §4.2 为骨架，n01 替换选择器和控制器）：
  - Controller（FPS 闭环）→ Selector（每帧，<1 ms，零分配）→ Fetcher（Range、合并、取消、退避）→ WorkerPool → Uploader（每帧字节预算）→ Cache（点数与字节双约束 LRU）→ RenderBackend（Glsl 与 Tsl 两个实现）→ Picker（深度回读，异步）→ Stats（4 Hz）。
  - 全部是纯 TS，放在 `engine/pointcloud/`，不引用 React。
- **选择器**：采用 n01 的两级预算 best-first。
  - 投影误差 `e = spacing_L · pf(d)`，其中 `pf = 0.5·H_dev / (tan(fovY/2)·d)`，`d = max(|cam−c| − r, near)`。
  - 统一使用设备像素。
  - τ 默认 1.35 px；τ_min = τ/4；headroom h = 0.15；pop 顺序直接作为下载优先级。
  - 输出 `limitedBy`（budget/nodes/headroom/error/complete）和 `achievedScreenError`，接入 HUD。
  - 保留 r12 的改进：tight 包围盒；L≤1 免裁剪；最后一个放不下的节点按比例画前缀；±10% 迟滞；屏幕中心加权与焦点（选中无人机）加权。
- **画质阶梯**：n01 的 7 档，以相邻两次呈现的间隔判定。

| idx | 名称 | renderScale | 点预算 | τ（设备 px） | 用途 |
|---|---|---|---|---|---|
| 0 | soft-min | 0.5 | 40,000 | 4.0 | SwiftShader 与 CI 起步档（r12 实测） |
| 1 | soft | 0.6 | 150,000 | 3.0 | 软件渲染上限 |
| 2 | minimum | 0.6 | 750,000 | 2.7 | 集显低档 |
| 3 | low | 0.75 | 1,500,000 | 2.0 | 集显 |
| 4 | medium | 1.0 | 3,000,000 | 1.35 | 默认 |
| 5 | high | 1.0 | 6,000,000 | 1.0 | 自动模式上限 |
| 6 | ultra | 1.0 | 12,000,000 | 0.7 | 只能手动选择 |

  判定规则：
  - p50 > 1.35T 时快速降一档；p95 < 1.1T 且稳定 5 s 时升一档；每次"升上去又掉下来"会让下次升档的等待时间翻倍，最长 120 s。
  - 解码积压期间不采样；超过 400 ms 的帧丢弃。
  - 档内由 n05 已实测的 `AdaptiveBudget` 在该档的 `[Bmin, Bmax]` 之间连续微调：按时间每 250 ms 评估一次，输入是 `useFrame` 的 delta，不是 `clock.getDelta()`。
  - 交互期间预算 ×0.6，或按角速度降 DPR（OLV `adaptiveDpr`）。运动降载必须作用于 drawRange 或 DPR，不能只缩小选择集（voxelkloud 实测后者对 INP 无效）。
- **流式调度**：
  - 首个节点落地前并发宽度为 1，之后 4–8（HTTP/2）。
  - 离开视锥满 2 帧就取消请求。
  - 失败退避 30/120/480 帧，最多重试 3 次，然后标记 failed。
  - 每帧上传不超过 8 MiB，软件档不超过 20k 点。
  - 驱逐规则：从 1.5×B 开始、释放到 1.15×B；不碰本帧选择集和根节点；可见节点有 1 s 驻留保护，但"被更细层取代"的节点除外。
  - `hierarchy.bin` 用一次不带 Range 的 GET 整体拉取（这样 CDN 能压缩）。
  - 请求头不带 `content-type: multipart/byteranges`，避免 CORS 预检。
- **密度与过渡**：节点首次驻留和父子替换时，用 Weyl 抖动溶解做 220 ms 淡入（不透明、写深度，EDL 仍然正确）；预算下降时 drawCount 线性收缩。
- **点径**：
  - 默认（WebGL2 与 WebGPU 档）：octree cut 纹理（n01），每项 4 B，按 BFS 顺序；只统计已驻留节点；缩小幅度最多 1 级；点径 clamp 到 [1, 8] px，覆盖系数 1.7–2.0。
  - 软件档：Lite 档（r12，每节点 8 位 childDrawnMask，O(1)）或按节点 uniform。具体取舍由 §9 G2 的实测对比决定。
- **着色**（UrbanScene3D 没有颜色）：
  - 默认 Height 模式：p1–p99 范围，γ=0.6，Graphite 5 档渐变。
  - 叠加 faceforward Lambert 与半球环境光；桌面档加 EDL（strength 0.3–0.5，半径 1.4–2 px）；软件档关闭 EDL。
  - 其他模式：HAG（旧金山必须用）、Normal、Class、伪 Intensity。
  - 高亮只用 `#E93024`。
  - 雾以 uniform 注入点材质：Low 档逐顶点计算，与环境模块共用 σ。
- **首屏**：metadata、`hierarchy.bin` 与一次首屏 Range 请求返回后，1 s 内出首帧（TTFP）。首屏加载进度用 shadcn `Progress` 显示，进度值为 `|R∩resident|/|T|`。
- **来源单元**：r12、n01、r11、r09、r13、n05、x01、r03、r10、r15。

### 3.6 Web 3D 框架与渲染后端

- **宿主**：R3F 9.8.1，采用"R3F 宿主 + 命令式引擎"（r14 B′）。
  - 点云、环境、无人机实例、轨迹、标签、拾取都是 `engine/` 里的命令式类。
  - R3F 只负责三件事：挂载（`<primitive>`）、按相位驱动引擎、声明式管理低频可编辑对象（航点、区域、视锥、Gizmo）。
  - 实测：把 LOD 节点增删做成声明式时，一次 tick 耗时 2–154 ms，命令式只要 0.02–0.3 ms。
  - 遥测严禁进入 React state：500 架无人机时每次 commit 要 6.5–8.4 ms。
- **帧序**：telemetry swap → 仿真时钟（-3）→ 无人机插值（-2）→ 相机（-1）→ LOD、环境、标签（0）→ 渲染管线（1）→ HUD/Gizmo（2，renderPriority=2）→ PerfGovernor（afterEffect）。V0.2 起由 `@pmndrs/scheduler` 驱动：Canvas 设 `frameloop="never"`，在 scheduler 里调用 `advance()`（n05 已实测），与 R3F v10 的相位语义一一对应。
- **RenderBackend**：
  - Tier A（硬件 WebGPU）：`WebGPURenderer` + TSL。
  - Tier B（WebGL2 独显或集显）与 Tier S（软件渲染）：经典 `WebGLRenderer`。
  - 探测方法：`requestAdapter()` 返回 null 或 `isFallbackAdapter`，或者 renderer 字符串匹配 `swiftshader|llvmpipe`，则判为 Tier S。
  - 设置页可以强制指定后端；启动后不在运行中切换材质模式。
  - reversed-Z 优先于对数深度；首帧前调用 `compileAsync` 预热；处理设备丢失（device lost）与上下文丢失（context lost）。
- **drei 白名单**：CameraControls、GizmoHelper（只在 WebGL 下）、Html（只用于 1–3 个选中卡片）、Detailed、meshBounds、Bvh、AdaptiveDpr/AdaptiveEvents、View（FPV 画中画）。禁止使用：`Points`/`PointMaterial`（每帧整块重传）、`<Html occlude>`、`Trail`，以及在点云祖先节点上挂指针事件（R3F 会递归 raycast，每次点击遍历全部点）。
- **无人机层**：InstancedMesh 按屏幕半径分三档：<4 px 用标记 sprite；4–48 px 用低模实例；>48 px 用 P600 glTF（最多 6 个）。插值延迟 D = 2/f_ws（20 Hz 时 100 ms），位置用 Hermite 插值（利用速度），姿态用 slerp，外推不超过 3/f_ws，之后 HOLD 并显示"信号延迟"徽标。轨迹用环形缓冲加 `addUpdateRange`，或 deck.gl TripsLayer 式的时间窗 shader。
- **标签**：LabelLayer 在单个 DOM 覆盖层上最多放 48 个标签，按网格去重叠，不做场景 raycast。
- **来源单元**：r14、r11、r12、n05、r15、r18。

### 3.7 环境视觉（EnvironmentLayer）

- **模块拆分**按 r16 §4.1：wind、precip、atmosphere、clouds、surface、lightning、lighting、quality、state。所有效果都是 simTime 的纯函数；下落和风的位移在 CPU 上用 float64 积分成相位，服务端下发累积位移 `dispFall`、`dispWind`，因此 Timeline seek 可以精确复现。
- **档位**：

| 效果 | Low（软件档 / 集显 WebGL2） | Med（WebGL2 独显） | High（WebGPU 独显） |
|---|---|---|---|
| 雨 | 无状态扁平四边形 3k–8k（`vertexIndex/6`，不用实例化） | 20k，双层，加遮挡 | 60k，加雨区门控 |
| 沙尘与雪 | 无状态 2k–3k | float RT 乒乓 16k | compute 100k |
| 风可视化 | 箭头网格，或无状态流线虚线（n04：服务端 RK4 流线加飞行时间 τ，可精确 seek） | GPGPU 迹线 8k | compute 迹线 64k，环形尾迹 |
| 雾 | 解析高度雾，在点材质里逐顶点算 | 逐像素 | froxel（V1.0） |
| 云 | 2D 天空穹顶云层 | 体积云，0.35 分辨率，1/16 Bayer 摊销 | 半分辨率加光照缓存（takram 算法改写为 TSL） |
| 天空 | 解析渐变，或预烘焙全景（takram 离线，r184 工具工程） | SkyMesh | takram 大气（需要 r184 或修复） |
| 云阴影 | 天气图解析，逐顶点，全档开启 | 同左 | 同左，或阴影图 |
| 环境 GPU 预算 | 不超过帧时间的 20%；实测软件档粒子合计约 15 ms | ≤3 ms | ≤5 ms |

- **SwiftShader 实测要点**：实例化绘制每个实例约 50–100 µs；扁平四边形同样数量下快 15–25 倍；每多一个全屏 pass 约 15–20 ms。因此 Low 档不加任何全屏 pass，雾在点材质里做。
- **状态**：天气状态以服务端为权威。`presets.json` 前后端共享，共 12 种物理单位预设，各轴有独立时间常数，按路由表过渡。客户端只做 τ≈0.3 s 的阻尼。闪电节律是 `(seed, 事件序号)` 的纯函数，用于回放。
- **沙盘相机锚点**：降水锚点在"相机"与"轨道焦点"之间按 AGL 插值；盒子尺寸按八度离散分档（20/60/180/540 m）并交叉淡化；远景把 σ_rain 并入雾。
- **来源单元**：r16、n04、r11、r15。

### 3.8 物理风场与环境场 E(x,y,z,t)

- **查询契约**：`environment.query(x,y,z,t, frame)` 返回：
  - 均值风（ENU，m/s）；
  - 湍流谱参数 (σ_u, σ_v, σ_w, L)；
  - MOR（米）与 σ；
  - 降水强度（mm/h）、粉尘、温度、气压；
  - `valid/solid` 标志；
  - `source_level`。

  四种帧模式来自 gz：GLOBAL、LOCAL、ADD_VELOCITY_LOCAL、ADD_VELOCITY_GLOBAL。查询按 N 个位置向量化。
- **分级**：
  - L0：均匀风加对数或幂律廓线（写明参考高度与 z0）。
  - L1：L0 加 1-cos 泊松阵风、每机 Dryden（MIL-F-8785C，按精确离散）、垂直风；或者群机共享冻结的 von Kármán 湍流盒（64³×4 m 约 0.25 s 生成，1.6 MB f16）。
  - L2：质量守恒诊断风（WindNinja 算法移植）。在 UrbanScene3D DSM 体素上用 scipy+pyamg 求解：SF 1.34M 格每个方向 7–12 s，Shenzhen 4.64M 格约 30 s；每城存 6 个 30° 扇区，按流向对齐插值，误差比朴素 lerp 小 28 倍。
  - L2.5：LBM 瞬态序列（V0.4+，离线）。
  - L3：OpenFOAM 13 RANS 扇区库（V0.6+，离线 Docker）。
- **力耦合**：`F = −½ρ·CdA·|v_rel|·v_rel − (Σ|Ω_i|)·c_rd·v_rel⊥`，其中 `v_rel = v − W(p,t)`，推力按 ρ(h)/ρ0 缩放（ISA）。**禁止使用 gz WindEffects 的默认 k=1**（实测会把 1 kg 机体吹飞 143 m）。环境按 50–100 Hz 批量采样，在物理子步之间做零阶保持。
- **存储**：运行时为 dense f16 加 zstd，另存 u8 solid mask。前端拿到的是 2 倍降采样后的 RGBA16F 3D 纹理（约 1 MB）。VDB 只作为交换与导出格式。
- **验收**：每个风场入库前计算散度残差、壁面穿透、入口廓线误差（WinDiNet 损失改写为 3D），不达标拒绝入库。UI 标注"诊断风场"。
- **来源单元**：r17、n03、r23、n04、r16、r20。

### 3.9 无人机仿真与飞控

- **保真度阶梯**（统一在 `DroneAdapter` 或 `SimBackend` 接口之后，按 capability flag 区分能力）：

| 档 | 后端 | 能力 | 版本 |
|---|---|---|---|
| L0 | 运动学插值（回放用） | 轨迹回放 | V0.1 |
| **L1（MVP 默认）** | FleetSim + PX4-lite 级联（r20，已与 SIH 对照：50 m 阶跃 vmax 11.89/11.89、t90 4.85/5.0 s、8 m/s 风中俯仰 7.9°/7.9°） | 1000 架 @250 Hz 单核 3.5 ms | V0.1 |
| L2 | L1 加 RotorPy 气动力矩 stage（n03）或 MRS 刚体 RK4 移植（r24），加下洗流与地效 | 50 架 @250 Hz 或 300 架 @100 Hz | V0.3–V0.4 |
| SIH | `px4io/px4-sitl` 每机一个容器（每机约 0.22 核、10 MB；8 核机 8 机以内 RTF≥0.92） | 真 PX4 语义、failsafe、故障注入 | V0.2 |
| HIL | World 统一步进的 lockstep（TCP 4560+i），由我方物理驱动 PX4 | 真 PX4 加空间风场与地形碰撞 | V0.4–V0.6 |
| Prometheus | 地面站协议后端（纯 Python codec）与模拟器 | 真 P600 与 Prometheus SITL | V0.2 模拟器；V0.5 真机 |
| gz/Isaac | 仅 GPU 节点 | 相机、LiDAR 传感器 | V0.5+ / V1.x |

- **FleetSim 架构**：采用 Crazyflow 式 SoA 加命名 step pipeline。stage 依次为 mission → controller → env_wind → aero → interaction → integration → collision → sensors → faults → battery → safety → telemetry_tap，全部在同一世界时钟下步进，从而避开 SIH 多进程的时钟漂移。控制律采用 PX4-lite 级联，另加：
  - 带减速约束的限速 `v_lim = min(vmax, Kp·d, √(2·0.7·amax·d))`；
  - 条件积分抗饱和（r23，实测超调 0.34 m，8 m/s 侧风下零稳态误差）；
  - LineTracker 重定目标前先刹停（STOP_MOTION，r24）；
  - 命令 watchdog：250 ms 收不到控制命令即切 Hold。
- **语义对齐**：Mock 暴露与 PX4 相同的 `nav_state/arming/landed` 枚举和模式名（Takeoff→AUTO_TAKEOFF、GoTo→DO_REPOSITION、Orbit→DO_ORBIT、RTL）。控制权状态机沿用 Prometheus（INIT/MANUAL/COMMAND/LAND，加 DEFAULT/OVERRIDE/RELEASE）。Safety FSM 沿用 r24。三者关系需统一，见 §9 G4。
- **多机编排**（V0.2）：每机一个容器 `px4 -i 0`，参数 `PX4_PARAM_MAV_SYS_ID=k+1`，显式设置 `PX4_UXRCE_DDS_NS`，`PX4_HOME_*` 取自 World 锚点。Gateway `udpout` 到 14580+i，这样能绕开 i>9 时端口固定为 14549 的问题。用 `SET_MESSAGE_INTERVAL` 把每机报文从 414.6 条/s 降到约 70 条/s。生命周期状态为 PENDING→…→READY→ACTIVE/DEGRADED/LOST→RESTARTING；心跳超过 2.5 s 判 DEGRADED，超过 5 s 判 LOST。
- **P600 参数**：`vehicles/p600/params.yaml` 是唯一参数源，带来源与置信度列。现有的 1.505 kg、推重比 5.2 是 Iris 级占位值，V0.4 之前必须用真实 ULog 辨识；在此之前 Mock 默认用 x500 参数（2.064 kg）。
- **来源单元**：r19、r20、r21、r22、r23、r24、n03。

### 3.10 多机、规划与安全

- **Planning Service**（放在仿真内核进程内）：
  - 全局 26 邻域 A*（numba，4 m 全城 ESDF，EDT 5.6 s，70 MB）→ LOS 剪枝 → 用 states2pts 生成 B-spline 初值 → L-BFGS-B 优化（光滑项、ESDF 距离项、逐轴可行性，45–390 个控制点耗时 25–78 ms）→ 时间拉伸 → TOPP-lite 降级路径。
  - **规划 worker 必须 `OMP_NUM_THREADS=1`**（多线程 BLAS 下慢约 40 倍）。
  - 轨迹契约：均匀三次 B-spline（order、ts、ctrl_pts、t0、traj_id），与 EGO Bspline.msg 一一对应。
- **三层互避**（V0.6）：
  - 战略层：优先级加 4D 预约（100 架随机任务 0.63 s）。
  - 轨迹层：椭球代价，z 方向要求 2 倍间距。
  - 战术层：ORCA-3D（100 架 1.6 ms/tick），加刹停与错层兜底。
- **编队**：虚拟锚点加 p/v/a 前馈加 2 阶航向滤波（跟踪 RMS 0.01–0.09 m）；变换队形用 CAPT（Hungarian 加同步 smoothstep，n=50 时 10 ms）。PrC 一致性律只作对照。
- **覆盖**：DSM（SF 2 m 栅格 0.9 s）→ 扫描线与 BCD → 按代价均衡连续切分 → Hungarian 分配；配合"扫描揭示"shader，coverage 瓦片约 1.1 kB/s。
- **Safety & Health**（r24）：
  - 四层：FastGuard（100 Hz，向量化）、MissionGuard（10 Hz）、FleetGuard（2–5 Hz）、HealthGraph（1 Hz）。
  - Flight FSM 共 13 态；自动转移只允许升级；ELAND 与 FAILSAFE 锁存到解锁为止；模式切换后 1 s 宽限。
  - 命令先过围栏校验（棱柱、禁飞区），违规返回 409 加原因码。
  - 电量 RTL 条件为 `t_rem < 1.3·t_rtl`（PX4 阈值 0.15/0.07/0.05）。
  - 三类链路丢失分别处理。
  - SafetyEvent 走可靠有序通道。
- **Control Lease**（r21）：owner、优先级、TTL。UI、任务引擎、ANet agent 不会同时控制同一架机；Land、Hold、RTL 永远放行。
- **Agent 安全**（n03 droneserver）：把 LLM 或远程 agent 视为不可信指挥官。守卫顺序：authenticate → state → tier → authorize → rate-limit → confirm-token → bounds → geofence → preconditions → execute → audit。
- **来源单元**：r25、r26、r24、r22、n03、x01。

### 3.11 实时通信

- **协议**：r27 `anet.rt.v1`，建议改名为 `awr.rt.v1`（d05：与 ANet 的 `anet.*` 事件名冲突）。
  - 端点 `wss://host/api/rt`，子协议协商，token 放在子协议里，校验 Origin。
  - 控制面：JSON（serverInfo、advertise、subscribe、ack、ping、call、playback、event、status），用有界 FIFO，满了就以 1013 断开。
  - 数据面：二进制 BATCH（16 B 帧头 + 16 B 记录头，payload 8 字节对齐），用单槽 mailbox，新帧覆盖旧帧。
  - TIME：24 B，10 Hz。
  - 客户端发布：`CLIENT_DATA`（遥操作 setpoint，deadline 200 ms）。
- **载荷**：
  - `uav/{id}/state`：Full64，30–50 Hz。
  - `swarm/state`：Lite32，10–20 Hz。
  - `state_ext` 与 `safety`：msgpack，1–10 Hz。
  - `env/weather`：msgpack。
  - `env/wind/field`：typed-blob，超过 256 KB 时改为下发 URL。
  - `event`：JSON，可靠通道，带全局 seq，断线后从 ring 或 REST 补拉。
- **调度与背压**：60 Hz tick；rate class 为 {1,2,5,10,15,20,30,60} Hz；对齐 tick 取最新值并保证尾帧；按 (channel, seq) 只编码一次；懒生产；credit 窗口 W = clamp(ceil(max_rate·srtt)+2, 2, 8)，默认 3；ack 合并为每 3 帧一次或 ≤20 Hz。
- **客户端**：WebSocket 放在 Worker 内（重连策略参考 partysocket）；按 rAF 拉取每个 channel 的最新 SoA，用 transferable 零拷贝传递；在 COI 下可改为 SharedArrayBuffer 环形缓冲（V0.3）；Zustand 摘要态 ≤10 Hz。
- **编解码实测**：1000 架时 raw struct 编码 5.7 µs、解码 15 µs；JSON 编码 40.7 ms。permessage-deflate 一律关闭。
- **时钟与回放**：服务端时钟为权威。ping/pong 取最小 RTT 样本估计偏移。seek 时 epoch 加 1，先按 `≤t` 下发 backfill 快照。MCAP 按原生频率录制，`log_time = t_sim_ns`。foxglove-sdk 提供调试旁路。
- **分层**：MVP 在进程内（Gateway 与仿真内核的 IPC 见 §9 G6）；V0.5 起接 zenoh 总线，rosbridge（ROS1 P600）和 rmw_zenoh（ROS2）作为适配器。
- **来源单元**：r27、r15、n04、n05、r21、d05。

### 3.12 Agent 与 ANet

- **分层**：
  - Agent Runtime：自研 Python，负责任务状态机（A2A 七态加 `awr.phase` 物理子相位）、合同网分配、TSIR 验收、黑板、证据链、Control Lease 申请。
  - ANet：外部覆盖网络，每个 agent 一个 daemon，外加自建 hub。
  - Drone Agent Adapter：连接两者。
- **分配**：
  - 打分函数 `U = 1.0·conf − 0.6·eta/120 − 0.3·E/10 − 0.2·load − 0.5·risk`；返航后电量低于 20%，或环境超出传感器限值时，判为不可行。
  - 三档分配器共用这一打分函数：V0.6 集中式 Hungarian/SSI；V1.0 ANet 合同网；断网时用 CBBA（打分修正为 Δ = S(path⊕j) − S(path)，满足 DMG）。
- **效果状态**：`OK / UNVERIFIED / FAILED / UNAVAILABLE / PAYMENT_REQUIRED`，从 V0.2 起用于所有命令回执。例如 goto 已被接受记为 UNVERIFIED，到达后读回确认才记为 OK。`OK` 且 `verify_trust ≥ 2` 才算"已验证"。
- **长任务**：飞行与观测类能力走两阶段：先立即返回 UNVERIFIED、eta、task_ref，之后再取结果。原因是 daemon 对非 LongRunning 调用有 60 s 上限。
- **能力 id**：裸形式，例如 `thermal.imaging`、`rgb.zoom`、`lidar.mapping`、`relay.communication`、`flight.*`、`mission.insert_leg`；每架机都要提供 `agent.describe`、`agent.state`、`task.quote`。
- **部署**：V1.0 使用自建 ANetHub，不用公网 hub；配置 AID 白名单；飞行节点不配置 auto_reply。版本在 `tools/anet/versions.lock` 中锁定。
- **来源单元**：d05、n03、r19、r27。

### 3.13 设计体系四件套

- **色卡 ANet Graphite**（d01 §3.2，已用 dataviz 校验器验证）：

| 类别 | token 与值 |
|---|---|
| 冷灰阶 | g950 `#0A0B0D`、g900 `#111214`、g850 `#16181B`、g800 `#1D1F23`、g700 `#2A2D32`、g600 `#3E4249`、g500 `#5C616A`、g400 `#81868F`、g300 `#A7ABB3`、g200 `#CACDD3`、g100 `#E4E6E9`、g50 `#F2F3F5`、white `#FBFBFC` |
| 品牌红阶 | r700 `#B4241B`（浅色主题的红色文字）、r600 `#D12A20`、**r500 `#E93024`（品牌红与数据标记）**、r400 `#FF5242`（暗色主题的红色文字）、r300 `#FF8374` |
| 数据阶 | 只用 5 档。暗色主题 g500→g50；浅色主题 g300→g900 |
| 默认主题 | 暗色（数字沙盘）；浅色用于报告导出 |

  红色规则：
  - 一张图只有一处红，优先级为"严重告警 > 当前选中的无人机 > 数据主角"。
  - 列表中的选中态不用红，改用 `bg-muted` 加 2 px 前景色竖条。
  - 状态不引入新色相：critical 为红色实心，warning 为红色描边空心，stale 为虚线，并且一律配图标和文字。
- **动效**：transitions.dev `_root.css` 是唯一 token 源（7 个 duration、6 个 easing、5 个 distance、4 个 scale、3 个 blur），按用途匹配取值。
  - 对接 Base UI 的 `data-starting-style/data-ending-style`，并用 codemod 去掉 tw-animate 的 keyframe 类。
  - motion tier（full/lite/reduced）与点云控制器共用 FPS 信号，先降 UI 动效再降点云。
  - 3D 画布始终全屏，面板是浮层，禁止用 width 动画触发 canvas resize。
  - 遥测数字分 C/D/E/S 四类：连续量不做动画，离散量用 pop-in，事件型用 counter，状态文字用 swap。
  - 相机飞行时长 `clamp(0.4 + 0.15·ln(1 + d/20), 0.4, 1.2)` s，曲线用 smooth-out。
- **图标**：`<Icon>` 为静态单 path（canonicalD）；`<StateIcon>` 在白名单图标对之间 morph，其余用 swap；航向和风向这类连续角度用 CSS rotate。约束：并发 morph 不超过 8 个，空闲时预热，状态分档带迟滞 1.5 s，`reducedMotion="user"`。共 208 个图标：lucide 202 个加自定义 6 个（PointCloud、SandDust、OctreeLod、DroneHexa、Formation、CameraFrustum）。
- **组件**：shadcn 4.21 base-mira（preset `b1D0dv96`）。
  - 用 `anet-base.json` 一次完成 init，约装 45 个 MVP 组件，源码提交进仓库。
  - 布局组合 sidebar-16、sidebar-07、sidebar-15 三个 block，中间用 Resizable v4。
  - 通知用 Base UI toast manager（`toast.add()` 可以在 React 之外调用）。
  - 移除模板 ThemeProvider 的裸 `d` 键绑定（与 WASD 冲突）。
- **图表**：用 lieflat 风格的自研组件。流式图用 CPU canvas（`willReadFrequently: true`，实测 12 图 × 1200 点 @10 Hz 保持 60 fps）；静态图用独立合成层 SVG。全局只有一个 LfScheduler：HUD 4 Hz，聚焦图 ≤10 Hz。MVP 约需 16 个图型。表格采用 table.log 规范。
- **品牌**：完整徽章只用于宽度 ≥480 px 的场合（启动页、登录页、关于页、报告封面）。顶栏用 24 px 头像标加产品名。暗色 UI 上徽章下方要垫实心 g950 底板，不能直接浮在点云上。
- **来源单元**：d01、d02、d03、d04、d05。

### 3.14 演示数据与剧本

- **六城角色**：六个城市分别对应六种世界形态，固化为 World Ingest 的回归与性能基准。默认演示城市为深圳。
- **剧本**（x01 §3.11）：
  - S1：深圳，双机立面螺旋扫描（默认演示）。
  - S2：上海陆家嘴，编队加覆盖。
  - S3：纽约港，搜救；ANet 热成像确认，置信度从 0.42 提到 0.9，作为 V1.0 验收用例。
  - S4：芝加哥湖岸，编队。
  - S5：旧金山，地形跟随。
  - S6：苏州，走廊飞行加中继。
- **航线**：UrbanScene3D 航线直接相似变换到城市上时，有 1.5%–48% 的视点不安全，因此连接段必须用 Height_map `safe_transit` 或规划器重新生成。定点 120–150 m 的覆盖任务会撞上每座城市的超高层，高度必须按航段查询。
- **Mock 任务生成器**：割草机（按重叠率定间距 `s = 1.155h(1−side_ov)`）、HelixScan、Orbit、Oblique5Grid、ExpandingSquare、Corridor、TerrainFollow、Formation。
- **分发**：数据条款禁止再分发。公开演示或公开仓库应在首次启动时于本地生成 World Package。

---
## 4. MVP 技术栈（锁定版本）

| 层 | 选型（精确版本） | 理由与依据 |
|---|---|---|
| 前端构建 | Vite 8.3.1 + @vitejs/plugin-react 6.1.1；`build.target = es2023`；dev 和 preview 都加 COOP/COEP 响应头 | n05 trial 构建耗时约 1 s；跨源隔离后计时精度为 5 µs，并可使用 SharedArrayBuffer |
| 语言与质量 | TypeScript 7.0.2（只用作 tsc）+ oxlint 1.86 / tsgolint 7.0.2003 + Prettier 3.9.9 | TS 7 没有 JS API，typescript-eslint 无法使用；oxlint 边界规则已实测生效 |
| UI 框架 | React 19.3.0 + Tailwind 4.3.3 + shadcn 4.21.0（base-mira，Base UI 1.8.0）+ `cn` 0.4.0 | 用户要求 shadcn；Base UI 是 shadcn 4.13 起的默认 base（d04） |
| 3D | three `~0.186.1` + @react-three/fiber 9.8.1 + @react-three/drei 10.7.9（白名单）+ three-mesh-bvh 0.9.x | r11、r14、n05；R3F v10 等 rc 后再迁移 |
| 渲染后端 | RenderBackend 双实现：经典 `WebGLRenderer`（GLSL 点材质，加 `WebGLNodesHandler` 承载 TSL 图层）和 `WebGPURenderer` + TSL | §3.6；§8 C1；§9 G1 需要实测确认 |
| 状态与数据 | zustand 5.0.15（摘要态）、@tanstack/react-query 5.104.0（REST）、@msgpack/msgpack 3.1.3（控制面） | n05 |
| 帧循环与性能 | R3F `useFrame` 帧序契约（V0.1）→ @pmndrs/scheduler 0.2.0（V0.2）；stats-gl 4.2.3；页内 FrameSampler 与 LoAF，暴露为 `window.__perf` | r14、n05 |
| 设计体系 | lucide@1.48.0（数据）+ morphicons 1.7.1 + transitions.dev token 与配方（vendor）+ lieflat 自研组件；自托管 Inter Variable 与 JetBrains Mono | d01–d04 |
| 测试 | Vitest 5.0.2（unit、browser、bench）+ Playwright 1.63.0（用 `executablePath` 指向本地 Chrome 151）+ SwiftShader 标志组合；只做相对断言 | n05 |
| 后端 API | Python 3.12 + FastAPI + uvicorn（`--ws websockets`）+ Starlette `StaticFiles`（已验证 Range） | r27、本次实测 |
| 实时协议 | 自研 `awr.rt.v1`（r27 称 `anet.rt.v1`）；`packages/contracts/rt/layouts.json` 同时生成 numpy dtype 与 TS 访问器 | r27 |
| 仿真内核 | numpy SoA FleetSim：PX4-lite 级联 + pipeline；Height_map 碰撞与 AGL；L0/L1 风；Safety FSM；Prometheus 与 PX4 语义对齐 | r20、r23、r24、n03、x01 |
| 世界处理 | numpy + plyfile + laspy + pyproj；CSF（cloth-simulation-filter）；small_gicp（下采样）；Open3D 0.20（可选，需要 libEGL） | r09、x01、n02、r07、r06 |
| 重建（MVP） | Mock Reconstruction Engine + Engine Adapter v2 接口；可选 DA3-SMALL CPU 冒烟 | r01、n02 |
| 调试与录制（V0.2） | foxglove-sdk 0.27.0（MCAP 录制与调试旁路）；Rerun（仅 dev） | r15、n04 |
| 后续进入 | V0.2：MAVSDK 4.0.0、`px4io/px4-sitl`。V0.5：zenoh 1.10.x、pycolmap 4.2、gtsam 4.3、PDAL Docker。V0.6：OpenFOAM 13 Docker。V1.0：ANet daemon + ANetHub | 各单元 |

---

## 5. MVP 实现直接复用清单（算法与代码 → 我们的文件或模块）

标注：方式 A 为 adopt 依赖，P 为移植，D 为按笔记自研（公式和参数已给出）。"实测"指该单元在本机跑过原型。

### 5.1 世界与点云（服务端和离线）

| # | 复用项 | 来源（仓库 / 文件 / 本地原型） | 目标文件 | 方式 | 状态 |
|---|---|---|---|---|---|
| W1 | UrbanScene3D ingest 规范化（六城 CFG、`T_enu_src` 矩阵、单位初判启发式、上方向判定、调平、法线修正、DTM 开运算、HAG、规则分类） | x01 §3.3、§3.7；`.cache/research/x01/analyze.py` | `world/ingest/urbanscene3d.py`、`world/ingest/normalize.py`、`world/ingest/units.py` | P | 实测，六城正确 |
| W2 | Potree 2.0 容器读写（22 B 层级记录 `<BBIqq>`、BFS、proxy 分页、childIndex = x<<2\|y<<1\|z） | PotreeConverter `indexer.cpp`、`HierarchyBuilder.h`；`.cache/research/r09_potree2_writer.py` | `world/pointcloud/format.py`、`engine/pointcloud/io/hierarchy.ts` | P | 实测，Potree 1.8 viewer 可加载 |
| W3 | 向量化加法式八叉树构建（Morton 排序、逐层 reduceat 网格竞选、中心优先、G=64、LEAF=20000） | `.cache/research/r09_octree_fast.py`；PDAL `copcwriter/Processor.cpp::sample` | `world/pointcloud/tiler.py` | P | 实测，5M 点 5–6 s |
| W4 | ANET_Q16 编码（`unorm16x4` 位置加 oct16 法线、`unorm8x4` 颜色加 class、`levelsByteEnd`、`hierarchy_ext.bin`、节点内 Fisher–Yates 打散） | r09 §3.3；r11 §3.2.1 | `world/pointcloud/tiler.py`、`engine/pointcloud/io/q16.ts` | D | 规范已定；v1 字段需收敛（§9 G3） |
| W5 | 点间距与密度估计（nnMedian）、首屏层级规则（cum ≥ 1e5） | x01 §3.4、§3.6；small_gicp `KdTree.batch_knn_search` | `world/pointcloud/stats.py` → `metadata.anet` | P | 实测 |
| W6 | CSF 地面 → DTM → HAG → 法线规则语义（ASPRS u8） | jianboqi/CSF（pip）；n02 `n02_semantic.py` v2/v3 | `world/ingest/semantic_rules.py`、`world/geometry/terrain/` | A + P | 实测，NY 112 s |
| W7 | Height_map（2.5D 最大高度 + 膨胀 + 安全距离）与 `safe_transit` | UrbanScene3D `src/model_tools.h::Height_map`；x01 §3.8 | `sim/world/heightmap.py`、`sim/planning/transit.py` | P | 实测 |
| W8 | WGS84<->ECEF<->ENU 精确公式与帧转换全集（UE→ENU、ENU→three、ENU/FLU<->NED/FRD） | cesium `Ellipsoid.js`、`FixedFrameTransforms.js`；colmap `gps.cc`；MAVROS `ftf_frame_conversions.cpp`；x01 §3.5 | `world/georef/frames.py`、`engine/geo/frames.ts` | P | 实测，与 pyproj 差 1e-9 m |
| W9 | Range 静态服务 | Starlette 1.7 `StaticFiles` | `apps/api/main.py` 挂 `/worlds` | A | 本次实测 206 正确 |

### 5.2 Web 点云引擎

| # | 复用项 | 来源 | 目标文件 | 方式 | 状态 |
|---|---|---|---|---|---|
| P1 | 两级预算 best-first 选择器（limitedBy、achievedScreenError、入堆时裁剪、放不下时跳过而不是 break、有限的 nearFloor） | `refs/discovery/voxelkloud-view/src/lod/select.ts`、`heap.ts`、`frustum.ts`、`metric.ts` | `engine/pointcloud/core/Selector.ts` | P | 源码精读（n01） |
| P2 | 目标集语义、分段前缀绘制、迟滞、中心与焦点加权、tight bbox | potree-core `source/potree.ts:165-350`；r12 §4.4 | 同上 | P | 设计 |
| P3 | 画质阶梯 QualityController（按呈现间隔、升档后掉档则等待加倍） | `voxelkloud-view/src/quality.ts` | `engine/pointcloud/core/QualityController.ts` | P | 源码精读 |
| P4 | AdaptiveBudget（AIMD，按时间 250 ms 评估，带迟滞） | `.cache/research/n05/trial/src/engine/adaptiveBudget.ts` | `engine/pointcloud/core/budget.ts` | P | 实测，约 2 s 收敛 |
| P5 | 流式策略（取消、退避、上传预算、驱逐不碰选择集）与驱逐迟滞 1.5/1.15 | `voxelkloud-view/src/stream-policy.ts`、`view.ts`；`openlidarviewer/src/render/streaming/evictionPolicy.ts` | `engine/pointcloud/core/{Fetcher,Cache,Uploader}.ts` | P | 源码精读 |
| P6 | hierarchy.bin 整体 GET | `voxelkloud-format-potree/src/hierarchy-fetch.ts` | `engine/pointcloud/io/HierarchyLoader.ts` | P | 源码精读 |
| P7 | 有界 worker 池（空闲终止） | three-loader `src/utils/worker-pool.ts` | `engine/pointcloud/io/workerPool.ts` | P | — |
| P8 | DEFAULT 解码 worker（兼容路径） | potree-core `loading2/decoder.worker.js` | `engine/pointcloud/io/decode.worker.ts` | P | — |
| P9 | GLSL 点材质（节点局部位置解码、Height/Normal/Class 着色、faceforward、雾 uniform、裁剪） | potree-core `materials/shaders/pointcloud.vs/fs`；x01 §3.7 着色器 | `engine/pointcloud/render/glsl/` | P | — |
| P10 | octree cut 局部点径（每项 4 B BFS 纹理，缩小封顶 1 级） | `voxelkloud-core/src/cut.ts`、`voxelkloud-view/src/points-glsl.ts` | `engine/pointcloud/core/OctreeCut.ts` + 着色器片段 | P | 实测数据来自上游 |
| P11 | Lite 点径（每节点 8 位 childDrawnMask） | r12 §3.5 | 着色器片段（软件档） | D | — |
| P12 | EDL（GLSL 与 TSL 两版） | potree-core `rendering/edl-pass.ts`、`edl.fs`；r11 §3.5 TSL 版（两后端实测可编译） | `engine/pointcloud/render/{glsl,tsl}/edl.ts` | P | 实测 |
| P13 | Weyl 抖动淡入（220 ms） | `openlidarviewer/src/render/streaming/fadeDither.ts` | `engine/pointcloud/core/Fade.ts` | P | — |
| P14 | 渲染档位探测（adapter 与 renderer 字符串）+ 设备丢失处理 | r11 §3.1、r12 §4.5 `detectTier`；OLV `renderBackendChoice.ts` | `viewport/renderer.ts`、`engine/perf/tier.ts` | P | 实测 |
| P15 | WebGL2 后端 `gl_PointSize` 补丁（仅作为 WebGPURenderer 路线的备选） | r11 §3.2.3（`GLSLNodeBuilder._getGLSLVertexCode`） | `viewport/patches/glPointSize.ts` | P | 实测可用，依赖私有 API |
| P16 | near/far 按被接纳节点的最小间距在帧首计算；reversed-Z | `voxelkloud-view/src/lod/metric.ts::suggestNearFar`；Potree-Next `Camera.js` | `engine/camera/NearFar.ts` | P | — |
| P17 | dev 对照页（potree-core npm、@voxelkloud/react） | npm `potree-core@2.0.15`、`@voxelkloud/react` | `apps/web/dev/oracles/*` | A | — |

### 5.3 视口、无人机与交互

| # | 复用项 | 来源 | 目标文件 | 方式 | 状态 |
|---|---|---|---|---|---|
| V1 | EngineLayer 适配模式（`<primitive>` + `useFrame` + needs-render→invalidate） | 3DTilesRendererJS `src/r3f/components/TilesRenderer.jsx` | `viewport/layers/*` | P | — |
| V2 | 帧序契约与 frameloop always/demand 状态机 | R3F `core/loop.ts`、`core/store.ts`；r14 §3.1–3.2 | `viewport/WorldCanvas.tsx`、`viewport/useFrameloopPolicy.ts` | P | — |
| V3 | async `gl` 工厂接入 WebGPURenderer | R3F `core/configuration.ts` | `viewport/renderer.ts` | A | 实测 |
| V4 | 无人机快照插值（prev/cur SoA）+ InstancedMesh 分档 + meshBounds | r14 §3.6；n05 `droneLayer.ts`；drei `meshBounds.tsx` | `engine/drones/DroneLayer.ts` | P | 实测，200 架 20 Hz |
| V5 | 固定槽位回放缓冲 + 恒速外推 + 尾迹 | Dynamic3DGaussians `helpers.py`、`visualize.py` | `engine/replay/ReplayBuffer.ts`、`engine/drones/TrailRing.ts` | P | — |
| V6 | 时间窗轨迹 shader | deck.gl `trips-layer.ts` | `engine/drones/trail-material.ts` | P | — |
| V7 | OpenCV 内参转 three 投影矩阵（含主点偏移） | Dynamic3DGaussians `helpers.setup_camera`；colmap viewer `cameraGeometry` | `engine/camera/intrinsics.ts` | P | — |
| V8 | CameraControls、GizmoHelper、AdaptiveDpr、View（FPV 画中画） | drei 10.7.9 | `viewport/controls`、`viewport/hud`、`viewport/views` | A | — |
| V9 | LabelLayer（单个 DOM 覆盖层、网格去重叠） | r14 §3.8（参考 drei `Html.tsx`） | `engine/labels/LabelLayer.ts` | D | — |
| V10 | 相机模式与参考系（Bird Eye = ENU，Third Person = 速度系，FPV = 姿态锁定）+ 目标位置预加载 | cesium `EntityView.js`、`TrackingReferenceFrame.js`；r15 §3 | `engine/camera/modes.ts` | P | — |
| V11 | 跟随相机平滑、低头视角的鸟瞰路径 | lingbot-map `demo_render/rgbd_render/camera.py` | `engine/camera/follow.ts` | P | — |
| V12 | JS cubic-bezier 采样器（3D 与 DOM 共用曲线） | transitions.dev `13-input-clear-dissolve.md` 中的 `bezier()` | `engine/anim/bezier.ts` | P | — |

### 5.4 仿真内核、安全与规划

| # | 复用项 | 来源 | 目标文件 | 方式 | 状态 |
|---|---|---|---|---|---|
| S1 | PX4-lite 级联（位置 P → 速度 PID 抗饱和 → 推力矢量与倾角限制 → 姿态 P → 速率限幅），向量化 | PX4 `mc_pos_control/PositionControl.cpp`、`ControlMath.cpp`、`AttitudeControl.cpp`；`.cache/research/r20/mock_px4lite.py` | `sim/fleet/px4lite.py` | P | 实测，与 SIH 一致 |
| S2 | jerk 受限平滑（VelocitySmoothing、转弯限速、按剩余距离限速） | PX4 `lib/motion_planning/*`、`TrajMath.hpp` | `sim/fleet/setpoint.py` | P | — |
| S3 | FleetSim step pipeline（SoA + 命名 stage + 单一世界时钟） | `refs/discovery/crazyflow/crazyflow/sim/pipeline.py`、`sim/data.py` | `sim/fleet/fleet.py` | P（架构） | — |
| S4 | 减速约束限速 + 条件积分抗饱和；carrot GoTo；命令 watchdog | r23 `.cache/research/r23/exp2.py`；AirSim `MultirotorApiBase.cpp`、`OffboardApi.hpp` | `sim/fleet/px4lite.py`、`sim/mission/goto.py` | P | 实测 |
| S5 | LineTracker 梯形参考 + STOP_MOTION | mrs_uav_trackers `line_tracker.cpp` | `sim/reference/line_tracker.py` | P | 实测 |
| S6 | 控制权状态机（INIT/MANUAL/COMMAND/LAND，DEFAULT/OVERRIDE/RELEASE）+ 零速轴保持 + 健康规则 | Prometheus `uav_controller.cpp`、`uav_estimator.cpp` | `sim/core/{authority,command,health}.py` | P | — |
| S7 | Safety FSM、FastGuard 阈值、围栏棱柱、起降条件、按油门推算质量的着陆检测、电量 RTL、链路丢失策略、预检 | mrs_uav_managers `control_manager.cpp`、`uav_manager.cpp`；mrs_lib `safety_zone`；PX4 battery 参数；`.cache/research/r24/mrs_mock.py` | `sim/safety/{flight_fsm,fast_guard,mission_guard,geofence,battery,health}.py` | P | 实测，6 个故障场景通过 |
| S8 | PX4 状态语义（nav_state、arming、landed、custom_mode 编解码） | PX4 `msg/versioned/VehicleStatus.msg`；MAVSDK `px4_custom_mode.hpp` | `sim/common/modes.py`、`packages/contracts` | P | — |
| S9 | 命令微协议（0.5 s 超时、重试 3 次、IN_PROGRESS）+ MAV_RESULT 语义 + 效果状态 | MAVSDK `mavlink_command_sender.cpp`；ANetCore `effect` | `sim/core/command.py` | P | — |
| S10 | L0/L1 风：廓线、1-cos 泊松阵风、每机 Dryden（精确 OU/ZOH 离散）、冻结湍流盒 | WindNinja `windProfile.cpp`；`.cache/research/r17/r17_turb.py`、`r17_turb_box.py`；RotorPy `dryden_utils.py` | `environment/wind/{profile,turbulence,turbbox}.py` | P | 实测，σ 误差 ≤1% |
| S11 | 风阻力模型（二次机体阻力 + 线性旋翼诱导阻力）+ ISA 密度 | AirSim FastPhysics、gz WindEffects、MulticopterMotorModel；r23 | `sim/fleet/aero.py`（L1 简化版） | P | 实测 |
| S12 | 任务生成器（割草机、螺旋、环绕、5 向倾斜、扩展方形、走廊、地形跟随、编队） | x01 §3.10 | `sim/mission/generators.py` | D | — |
| S13 | 轨迹原语（修正后的圆弧中心公式、按弧长参数化带速度前馈） | px4_multi_drone_sim `g22.py` 等（需修 bug） | `sim/mission/primitives.py` | P | — |
| S14 | 虚拟结构编队 + CAPT 分配 + 队形槽位（Python 与 TS 双实现，golden JSON 对拍） | r26 `r26_formation*.py` | `swarm/formation/*`、`ui/mission/formation/slots.ts` | D | 实测 |
| S15 | 覆盖规划（DSM、扫描线、BCD-lite、均衡切分、Hungarian）+ 覆盖栅格与"扫描揭示" | r26 `r26_coverage.py`；mavsdk_drone_show `coverage_planner.py` | `swarm/coverage/*`、`sim/coverage.py` | P | 实测，每 tick 0.41 ms |

### 5.5 实时通信与时间

| # | 复用项 | 来源 | 目标文件 | 方式 | 状态 |
|---|---|---|---|---|---|
| T1 | `awr.rt.v1` 帧格式（16 B 帧头与记录头、TIME 24 B、CLIENT_DATA）与 JSON op 集 | r27 §3.3–3.5；`.cache/research/r27/gw_proto.py` | `apps/api/rt/protocol.py`、`apps/web/src/net/rt/rt.worker.ts` | D | 实测原型 |
| T2 | Full64 / Lite32 布局 + `layouts.json` 代码生成 | r27 §3.5；`bench_encode.py`、`bench_decode.mjs` | `packages/contracts/rt/layouts.json` → `layouts.py`、`layouts.ts` | D | 实测，编码 5.7 µs、解码 15 µs |
| T3 | 按 tick 对齐的最新值调度器 + 尾帧保证 + 每个 (channel, seq) 只编码一次 + 懒生产 | r27 §3.8；foxglove `FoxgloveServer.ts` | `apps/api/rt/scheduler.py`、`channels.py` | D + P | — |
| T4 | 控制面与数据面分离（有界 FIFO 满则 1013 断开；数据单槽 mailbox）+ credit 窗口 + 合并 ack | foxglove-sdk `connected_client.rs`、`send_lossy.rs`；r27 实测 | `apps/api/rt/session.py` | P | 实测，p95 138 ms，无队列膨胀 |
| T5 | Worker 持有 WebSocket + 每个渲染 tick 拉取最新帧 + transferable | ws-protocol `WorkerSocketAdapter.ts`；n05 `telemetry.worker.ts` | `apps/web/src/net/rt/*` | P | 实测 |
| T6 | 重连退避（minDelay 500 ms、grow 1.5、max 10 s、有界队列只保留控制命令） | partysocket `ws.js` | 同上 | P | — |
| T7 | TIME 与 ping/pong 最小 RTT 时钟同步 + epoch | ws-protocol Time；foxglove-sdk ping/pong | `apps/api/rt/clock.py`、`net/rt/clock.ts` | P | — |
| T8 | SimClock（倍率、钳制或循环、step(n)、暂停时保持心跳） | cesium `Clock.js`；AirSim `SteppableClock` | `sim/clock.py`、`engine/time/clock.ts` | P | — |
| T9 | 回放 Player 状态机 + seek backfill + BlockLoader 预读 | lichtblick `IterablePlayer.ts`、`BlockLoader.ts` | `engine/time/player/*`（V0.2） | P | — |
| T10 | M4 降采样（每 3 px 桶保留首、最小、最大、末值） | lichtblick `TimeBasedChart/downsample.ts` | `ui/lf/lib/downsample.ts` | P | 实测，90k 点 4.4 ms |
| T11 | 性能 HUD 指标（x realtime、fps、p95、WS Mbps、渲染点数、瓦片数） | lichtblick `PlaybackPerformance`；r15 | `ui/panels/perf/` | P | — |
| T12 | 页内 FrameSampler + LoAF + Playwright 流畅性用例（CDP tracing 摘要）+ Vitest bench | `.cache/research/n05/trial/src/perf/metrics.ts`、`perf/viewport.perf.spec.ts` | `apps/web/src/engine/perf/`、`apps/web/perf/` | P | 实测 |

### 5.6 环境视觉（MVP 部分）

| # | 复用项 | 来源 | 目标文件 | 方式 | 状态 |
|---|---|---|---|---|---|
| E1 | 无状态雨（扁平四边形、`vertexIndex/6`、world-tiling、亚像素补偿、DSM 遮挡） | natural-disasters `Precipitation.js::RAIN_VERT`；Eanpa `weather_system.js` | `engine/environment/precip/RainStreaks.ts` | P（改写为 TSL，另留 GLSL 版） | 实测，比实例化快 15–25 倍 |
| E2 | float64 积分相位 WindIntegrator | Eanpa `weather_system.js::update`、`cloud_motion.js` | `engine/environment/wind/WindIntegrator.ts` | P | — |
| E3 | 解析指数高度雾 + 地面雾（MOR→σ），在点材质里逐顶点算 | Quilez 积分（r16 数值验证） | `engine/environment/atmosphere/HeightFog.ts` | D | 实测 |
| E4 | 天气图 + 解析云阴影 + 2D 云层（Low 档） | natural-disasters `ProceduralTextures.js`、`Clouds.js::weatherAt` | `engine/environment/clouds/{WeatherMap,CloudShadow,CloudLayer2D}.ts` | P | — |
| E5 | 离线噪声烘焙（shape、detail、weather、curl） | natural-disasters `ProceduralTextures.js` | `tools/bake_env_noise.py` | P | — |
| E6 | 预设表（12 种物理单位）+ 按轴独立阻尼 + 过渡路由 | procedural-weather-threejs `weather-types.md`；Eanpa `transitionTo` | `packages/contracts/env/presets.json`、`environment/weather/*` | P | — |
| E7 | 确定性闪电节律 + 固定 buffer 的闪电几何 | Eanpa `LIGHTNING_CADENCE`、`rebuildBolt` | `engine/environment/lightning/*`、`environment/weather/lightning.py` | P | — |
| E8 | 环境质量档 + 闭环降级链 + 分项 GPU 计时 | natural-disasters `Quality.js`、`GpuProfiler.js` | `engine/environment/quality/*` | P | — |
| E9 | 风箭头（`atan2(w, ‖uv‖)` 求俯仰）；无状态流线虚线（服务端 RK4 流线加飞行时间 τ） | cesium-wind-arrows-3d；n04 `bench/src/streak.js` | `engine/environment/wind/WindViz.ts`、`environment/wind/streamlines.py` | P | 实测，1.6 万段 +32 ms（软件档） |

### 5.7 UI、设计体系与 Agent

| # | 复用项 | 来源 | 目标文件 | 方式 | 状态 |
|---|---|---|---|---|---|
| U1 | ANet Graphite 色卡 CSS 变量（shadcn 变量与 lf 变量并存）+ `@theme inline` | d01 §4.2（`.cache/research/d01/palette.mjs` 校验） | `apps/web/src/styles/globals.css`、`lib/lf/tokens.ts` | D | 校验通过 |
| U2 | `anet-base.json`（registry:base）+ 离线 registry 镜像 | d04 §3.3；`.cache/research/d04/anet-base.json`、`mirror.sh` | `design/anet-base.json`、`tools/shadcn-mirror.sh` | A | 实测 |
| U3 | 布局骨架（sidebar-16 顶栏、sidebar-07 左栏、sidebar-15 右栏、Resizable v4 视口与底部 Dock） | shadcn `bases/base/blocks/*` | `ui/layout/*` | P | — |
| U4 | motion token（7 个 duration、6 个 easing、5 个 distance、4 个 scale、3 个 blur）+ 32 个配方 CSS | transitions.dev `skills/transitions-dev/_root.css`、`01…32-*.md` | `styles/motion/tokens.css`、`styles/motion/t/*` | A | — |
| U5 | 动效 hooks：usePresence、useReplay（WAAPI）、SwapText、StatusLine、MotionNumber、useSlidingPill、useShake | transitions.dev `index.html` React 模板 p1–p34 | `ui/motion/*` | P | — |
| U6 | 图标：`<Icon>`（canonicalD）、`<StateIcon>`（createMorph + 同一 svg 内 WAAPI swap）、白名单、K=8 并发预算、空闲预热、208 个图标注册表 | morphicons `src/dom`、`src/core`；`.cache/research/d03/inventory.mjs` | `ui/icons/*` | A + D | 实测 |
| U7 | lieflat 图表组件：LfChartCard、LfStat、LfSparkline、LfLiveLine（G17）、LfTickGauge（F11）、LfRungBars（F1）、LfTickRows（F5）、LfHistogram（F14）、LfHairlineLine/Area（F2/F3）、LfTable（table.log） | lieflat `templates/*-gallery.html`、`mono-tokens.js`、`report-10.zh.html` | `ui/lf/*` | P | CPU canvas 与 SVG 已实测 |
| U8 | LfScheduler（单一 rAF，HUD 4 Hz，聚焦图 ≤10 Hz，不可见时暂停）+ Float32 环形缓冲 | d01 §3.6.2–3.6.3 | `ui/lf/scheduler.ts` | D | — |
| U9 | `lint-lf`：禁止 `Math.random`、token 以外的 hex 色值、emoji 与 U+25B2/U+25BC/U+25CF/U+25CB（三角、圆点）字形、重复 id；另加 motion-lint | lieflat `scripts/validate.mjs`；transitions.dev `refine/server/motion-tokens.mjs` | `scripts/lint-lf.mjs`、`tools/motion-lint` | P | — |
| U10 | 效果状态模型（5 种状态、Evidence、V/A 信任，钳制而不抬高） | ANetCore `effect/effect.go`；ANetLink `profile/trust.go` | `agent/runtime/effect.py`、DroneState 命令回执 | P | V0.2 起 |
| U11 | 品牌素材（徽章 SVG、头像）与使用规范 | ANet `docs/media/anet-logo.svg`；GitHub 头像 u/305781773 | `apps/web/public/brand/` | A | — |

---
## 6. 2026 年新发现项目及纳入结论

判定口径：2025–2026 年新建，或 2026 年发布关键新能力；星数少的项目只 port 算法，不做运行时依赖。

| 项目 | 新在哪里（2026） | stars | 是否纳入 | 纳入方式与位置 |
|---|---|---|---|---|
| voxelkloud（view/core/format-*） | 2026-08 新出现；两级预算选择器、视频播放器式画质阶梯、octree cut 点径、WebGL2 单 draw、WebGPU compute 三遍光栅 | 0 | **纳入（port 核心算法）** | `engine/pointcloud/core/*`；compute 光栅放 V0.3 Tier A |
| NASA-AMMOS/3DTilesRendererJS 0.5.3 | 2026-09 新增 PotreePlugin、PointCloudEffectsPlugin | 2476 | **纳入** | V0.1 对齐 SSE 口径；V0.8 作为网格与地形底座；dev 对照页 |
| Aurtechmx/openlidarviewer | 2026-06 新出现，WebGPU 优先的 LiDAR 查看器 | 23 | **纳入（port 策略函数）** | governor、自适应 DPR、Weyl 淡入、驱逐迟滞 |
| PlayCanvas gsplat-unified 与 splat-transform 3.7 | 跨实例性价比预算分配；Streamed SOG | 16944 ／ 1339 | **纳入** | V0.6 全局预算仲裁；V0.8 3DGS LOD |
| manycoretech/aholo-viewer | 2026-05 开源的 chunk LOD 3DGS | 1076 | 参考 | V0.8 |
| sparkjsdev/spark 2.x | RAD、分页 extSplats | 3660 | 纳入（算法与独立预览路由） | V0.1 算法；V0.8 预览 |
| three r186 `GaussianSplat` + `WebGLRenderer.setNodesHandler` | 官方 3DGS；经典渲染器也能跑 TSL | 116010 | **纳入** | 3DGS 默认层（V0.8）；Tier B 承载 TSL 图层 |
| ByteDance-Seed/Depth-Anything-3 | 2026 any-view 重建，有小模型 | 6402 | **纳入** | V0.1 CPU 冒烟；V0.5 DA3-Streaming |
| facebookresearch/map-anything | 以 RTK 与 LiDAR 为条件的度量重建 | 3771 | **纳入** | V0.5 度量引擎（V0.1 定接口） |
| facebookresearch/vggt-omega | VGGT 官方后继 | 4588 | 参考 | 经 MapAnything 包装做评测 |
| PRBonn/rko_lio | 2026 年 pip 可装、无需 ROS 的 LIO | 660 | **纳入** | V0.2 回归测试；V0.5 离线 LIO |
| Liansheng-Wang/Super-LIO | FAST-LIO2 的活跃后继候选 | 602 | **纳入** | V0.5 机载；OctVoxMap 移植 |
| jianboqi/CSF | 2026 仍在维护的地面滤波 | 649 | **纳入** | V0.1 语义与 DTM |
| spencerfolk/rotorpy 3.0 | 气动力矩与 HIL 桥 | 311 | **纳入（port）** | L2 气动 stage；HIL 客户端参考 |
| learnsyslab/crazyflow | 2024-11 新建，2026 活跃；SoA pipeline | 180 | **纳入（port 架构）** | FleetSim 骨架 |
| XDEI-Group/AerialClaw | 2026-03 新建的 LLM 空中智能体框架 | 132 | 纳入（语义） | V1.0 Agent Runtime |
| PeterJBurke/droneserver | 2026 年"LLM 不可信指挥官"守卫 | 7 | **纳入（port）** | V0.6 命令网关中间件；V1.0 MCP |
| takram three-geospatial | 2026 最完整的 Web 大气与云 | 1697 | 有条件纳入 | High 档（需 r184 或修复）；离线烘焙天空全景；云算法改写为 TSL |
| arch85-km/urban-wind-lab | 2026-09 v1.0，浏览器与 CPU 可跑 LBM | 1 | 纳入（port，离线） | L2.5 瞬态风（V0.4–V0.6） |
| NOC-OI/cesium-wind-layer（fork） | 2026-09 活跃的 3D 几何拖尾 | 0（上游 125） | 纳入（port） | Med 档风粒子 |
| rbischof/windinet | 2026 城市风 ML 代理（2D，行人高度） | 12 | 部分纳入 | 损失函数作为风场验收指标（V0.3） |
| moq-dev/moq | 2026 活跃的 MoQ 中继与 moq-lite | 1546 | 纳入（语义）+ 可选 | QoS 字段写进 rt 协议；V1.0 可选 WebTransport 中继 |
| rerun-io/rerun 0.38/0.39 | 研发记录仪 | 11503 | 纳入（仅 dev） | `.rrd` sidecar |
| @pmndrs/scheduler | 2026 新仓库，R3F v10 内核 | 8 | **纳入** | V0.2 统一帧循环 |
| TypeScript 7（Go 原生）、Vite 8（Rolldown）、oxlint + tsgolint | 2026 工具链换代 | — | **纳入** | V0.1 工程栈 |
| shadcn 4.x（Base UI 默认、8 种 style、Chat 原语） | 2026 年 shadcn 基本面变化 | 124743 | **纳入** | base-mira；Chat 原语用于 V1.0 ANet 协作日志 |
| guillermolg00/morphicons | 2026-08 首发 | 2705 | **纳入** | 全站图标 |
| SkyeShark/Eanpa-Sky | 2026-09 活跃的 WebGPU 天气引擎 | 56 | 纳入（port 算法） | 降水锚定、闪电、过渡 |
| ZJU-FAST-Lab/Primitive-Planner | 2026-08 仍有提交的 EGO 后继 | — | 观察 | 百机规模时再评估 |
| Visionary、LidarScout、Layered 4D-Rotor GS、splats4D、4C4D | 2026 动态高斯与外存浏览研究 | — | 参考 | V1.x Dynamic Visual World |
| amb3r-slam | 2026 千米级 SLAM（官方代码未发布） | 41 | 不纳入 | 注意：0 stars 的第三方复现不代表论文结果 |
| pywebtransport | 2026 Python WebTransport（与浏览器握手失败） | 42 | 不纳入 | — |

**对 `02-refs.md` 的增补建议**：
- Web3D 组新增：voxelkloud（P0）、3DTilesRendererJS（P1）、copc.js（P1）、openlidarviewer（P1，只取策略）、splat-transform（P2）。
- 调整：potree-core 和 three-loader 降为对照；Potree-Next 降为 P2。
- LiDAR 组新增：rko_lio、Super-LIO、CSF、MARSIM、FAST-Calib、vdbfusion。
- 更正：cvg/glomap 标注为"已并入 COLMAP 4.x"；3d-tiles-tools 更正为"处理、打包、合并工具，不含切片器"。
- 新增 3d-tiles-validator；新增 street_gaussians、drivestudio、splats4D、4C4D（V1.x）。

---

## 7. 对 01-design.md 的优化建议汇总（按重要性排序，已去重合并）

> 每条后的方括号为主要来源单元。建议在撰写各说明书时逐条落实，并在 01-design 中标注"已被 xx 说明书取代"。

**P0：不改就会导致 MVP 失败或返工**

1. **新增"跨模块契约"一章（坐标、单位、时间、命名）**：World ENU 唯一；`coordinate.json` schema 合并 r15、x01、r02、r07 的字段；GPU 只放相对坐标，禁止 float16 世界坐标；int64 `t_sim_ns`；四元数 xyzw，表示 WORLD←BODY(FLU)；PX4 NED/FRD 只在网关换算；航向显示约定；三个高程字段；所有接口带 `frame_id`。[r15, x01, r02, r09, r11, r21, r04]
2. **§15 与 §43 插入 Ingest 规范化阶段**，作为 V0.1 必经环节，覆盖单位、上方向、调平、北向、原点、法线、DTM、HAG、分类，并设 QA 门禁。写入六城实测配置；公开发布时在本地生成 World Package（数据条款禁止再分发）。[x01, r09, r06, r07, r10, n02]
3. **§14 重写点云 LOD**：删除"远 10K / 近 1M"。改为全局点预算加设备像素 SSE 两级选择（τ=1.35）、目标集与分段前缀绘制、FPS 闭环画质阶梯（含 soft 档）、节点内打散前缀密度控制、cut 点径、EDL、Weyl 淡入、首屏 BFS 前缀用一次 Range 取回；输出 `limitedBy/achievedScreenError` 遥测，并写明各档默认参数表。[r12, n01, r11, r09, r10, r13, n05]
4. **§9、§12、§34 更正渲染事实并定义 RenderBackend**：WebGPU 点恒为 1 px；`WebGPURenderer` 的 WebGL2 后端写死 `gl_PointSize=1`；GLSL ShaderMaterial 不能在 WebGPURenderer 上运行；"WGSL"改为"TSL"；WebGL2 档是一等公民并且是 CI 主路径；浏览器 compute 只做可视化；"300,000 粒子"按档位分级（WebGPU 20 万–50 万，WebGL2 ≤5 万，软件档 ≤5k）。[r11, r12, r14, n01, r13, r16]
5. **§26、§29、§35 仿真后端改为保真度阶梯**：Mock FleetSim（MVP）→ PX4 SIH 容器（V0.2）→ HIL（V0.4–V0.6）→ gz/Isaac 仅 GPU 节点；定义 `DroneAdapter/SimBackend` 接口与 capability flag；MVP 不依赖 PX4 和 Gazebo；机间交互需要统一时钟（SIH 多进程时钟会漂移）。[r20, r21, r22, r23, r19, n03]
6. **§36、§37 实时协议成章**：`awr.rt.v1` 全规范（帧格式、Full64/Lite32、订阅 rate/mode/priority、credit 背压、tick 对齐取最新值加尾帧、TIME/epoch、Playback、RPC、事件可靠通道、鉴权与 Origin 校验）；数据流改为"PX4 → MAVLink UDP → Gateway"；ROS2 只作可选适配；大块数据走 HTTP；更新频率按订阅分级；插值延迟 D=2/f。[r27, r15, r21, r22, n04, n05]
7. **§28 DroneState 给出字节级规范与语义表**：Full64/Lite32；PX4 nav_state、arming、landed；FlightFSM；authority 与 control lease；链路 fcu/gcs；定位来源与状态；效果状态；truth 与 estimate；`seq`、`t_sim_ns`；NaN 表示未知；另附 PX4 与 Prometheus 语义映射表。[r27, r20, r19, r21, r24, r08, d05]
8. **§43 MVP 重新定义为双链路并加入可测验收**：内置世界链路加重建链路（Mock 或 DA3 冒烟）；record→replay；Playwright 流畅性测试。验收门槛：TTFP ≤1 s；控制器 3 s 内收敛且变化系数 <15%；自身 JS 不产生长任务；`drawn ≤ budget`；节点失败数为 0；200 架 20 Hz 下 UI 不卡；真 GPU runner 另设绝对门槛（1080p 下 p95 ≤16.7–20 ms）。[n05, r12, r14, x01, r15]

**P1：影响架构完整性**

9. **§5、§6 重建与融合**：前馈重建天然无尺度、不对齐重力；采用 Engine Adapter v2；GNSS Sim3 前移到 V0.1，直线航带需重力增强；融合采用轨迹优先 Sim(3) → Sim(3)-GICP → 因子图；LIO 定位为度量骨架（MID-360 在 30–40 m AGL 以上会退化）；工具分工 small_gicp、Open3D、gtsam；V0.5 拆成 a/b/c 三步；新增"重建质量与验收"指标。[r01, r02, r05, r06, r07, r08, n02]
10. **§7、§8、§41 World Model 扩展**：Geometry（物理权威：DSM/DTM、占据、SDF、碰撞代理 L0–L3）、Visual（LOD 瓦片、surfel、3DGS）、Localization、Semantic（区域用矢量体表达，不用点类别）；Track 与 Dynamic 层；`world.json` manifest；目录见 §3.4；Vehicle Package 独立。[r03, r06, r08, r10, r15, r18, x01]
11. **新增 Safety & Health 模块**：四层守卫、Flight FSM、命令校验（409 加原因码）、运行时围栏回拉、能量感知 RTL、三类链路丢失处理、SafetyEvent 可靠通道、故障注入场景库进 CI。[r24, r21, n03]
12. **§17–§25 环境**：MOR→σ 作为单一真值；风分级 L0–L3 可测（L2 质量守恒法、L2.5 LBM、L3 OpenFOAM 扇区库，按流向对齐插值）；湍流改为每机滤波器加均值场；E 查询契约（含 valid/solid、source_level、湍流强度）；环境视觉三档与预算；服务端权威、客户端短阻尼；风力方程与"禁用 gz 默认 k=1"；环境溯源与验收指标；§18 注明"4D"指时变世界状态，不是 4DGS。[r16, r17, n03, n04, r23, r18]
13. **新增 Planning Service 与多机三层互避**：A* 加 B-spline 加 ESDF；统一 B-spline 轨迹契约；规划栅格金字塔；三层互避；覆盖与编队规划器；风场感知规划（V0.4）。[r25, r26, r22, x01]
14. **§31、§32、§50 ANet**：ANet 是协作平面（约 1 s）；一机一 AID；裸能力 id；合同网 find→quote→score→delegate；TSIR 验收；黑板；证据链作为 Timeline 事件源；效果状态与任务状态双轴；自建 hub；Agent Runtime 与 ANet 分层；LLM 视为不可信指挥官；§32 搜救流程落成 S3 剧本并作为 V1.0 验收用例。[d05, n03, x01]
15. **§38–§40 UI**：设计体系四件套写入 UI PRD；新增顶栏（世界 Combobox、仿真时钟、连接状态）、Ctrl+K 命令面板、性能 HUD 与质量预设、Dock 面板系统（遥测图、事件、日志）、告警模型、控制权与 SIM/REAL 徽标、Timeline 图表化（loaded range、事件标记、键盘步进）、相机模式与参考系映射、FrameTree、FPV 画中画、LabelLayer；把原文中的勾选框、播放三角等 Unicode 字符改为 shadcn 控件与 morphicons 图标，把"Fog 0.21"改为 MOR 米数。[d01–d04, r15, r14, r16]

**P2：修正与完善**

16. **§33 后端栈表修正**：P600 是 ROS1 Noetic + mavros + 私有地面站协议，不是 ROS2/DDS；Message 一行改为"进程内（MVP）→ Zenoh（V0.5）"；新增 pycolmap、small_gicp、gtsam、PDAL（Docker）、foxglove-sdk、MAVSDK v4；CFD 改为 OpenFOAM 13 Docker（`foamRun`）；诊断风用 WindNinja 算法移植（scipy/pyamg）。[r19, r27, r02, r07, r09, r17, r21]
17. **§11 表格**：Cesium 拆成"算法移植（MVP）加可选地球视图（V0.5+）"；deck.gl 为可选 2D 态势图；Potree 拆成 1.8 参考、Next 参考、potree-core 对照三行；新增 Foxglove/Lichtblick（回放架构）与 3DTilesRendererJS。[r15, r13, n01]
18. **§39 Timeline**：区分仿真时间与墙钟；BUFFERING 状态；seek 走 backfill 加 epoch；倍速 0.1–20x；可选 what-if 分叉（V0.4 checkpoint）；回放格式用 MCAP，真飞、重建和仿真共用同一个 Player。[r15, r27, r18, n04]
19. **§42 仓库结构**按 §3.0 重排，新增 `packages/contracts`、`datasets`、`scenarios`、`tools`、`world/qa`；用 oxlint 做边界约束。[n05, r27, x01, d04]
20. **文字修正**：清理 `:chatgpt-content-reference` 残留；修正"知（应为感知）""访（应为访问）""之再（应为之后再）""项（应为项目）"等缺字；修正 §41 的 `── visual/` 树形错误；产品色补上"白"。[d04, r09, r10]

---

## 8. 冲突与待决问题

### 8.1 研究单元之间的冲突与裁决

| # | 冲突 | 各方观点 | 裁决（本整合建议） | 仍需确认 |
|---|---|---|---|---|
| C1 | 应用渲染器 | r11：全应用用 `WebGPURenderer`（自动回退），WebGL2 后端靠私有 API 补丁恢复 `gl_PointSize`。r12、r14、n01：点云回退走经典 `WebGLRenderer`（r14 建议 V0.1–V0.2 默认 WebGLRenderer）。r16：环境全部用 TSL（ShaderMaterial 在 WebGPURenderer 下不可用）。n05：trial 用 WebGPURenderer 加 1 px 点跑通 | RenderBackend 双实现：Tier A 用 WebGPURenderer 加 TSL；Tier B/S 用经典 WebGLRenderer 加 GLSL 点材质，环境等 TSL 材质经 `WebGLNodesHandler` 运行，Tier B/S 不使用 compute、MRT 或 RenderPipeline，需要的后处理（EDL、体积云合成）提供 GLSL 版。补丁方案保留为备选。依据：软件档下 WebGPURenderer 的固定帧开销为 36–100 ms（取决于输出缓冲类型），经典渲染器约 9 ms；每对象 CPU 开销约 14 µs 对 22–29 µs | 见 §9 G1（实测验证） |
| C2 | LOD 细分判据与参数 | r11：`MIN_NODE_PX=150`；r12：τ=1.2–1.5 CSS px，含迟滞；r13：用 150 px 节点阈值会欠填（NY 只用到预算的 47%）；n01：τ=1.35 设备 px，两级预算；r09：τ=1.5 | 统一采用"点间距投影 SSE、设备像素、τ=1.35"，按档位调整（见 §3.5 阶梯表）；节点尺寸阈值（minNodePixelSize）只作派生量 | 实测对比确定软件档的 τ 与点径模式（G2） |
| C3 | 运行时点云格式 | r10：V0.1 原生用 3D Tiles 1.1 隐式八叉树（已通过 validator）。r09、r12、r13、n01、x01：用 Potree 2.0 加 ANET_Q16 | 运行时用 Potree 2.0 加 ANET_Q16（零解码、可分段绘制、有三个现成对照查看器）；3D Tiles 1.1 作为 GIS 与 Cesium 导出（V0.5/V1.0），沿用 r10 的 SSE 与 GE 规则；COPC 作为归档 | — |
| C4 | 场景坐标轴 | r15：RENDER = WORLD，设 `Object3D.DEFAULT_UP=(0,0,1)`，保持 Z-up。x01、r10、r14：three 保持 Y-up，WorldLayer 根节点旋转 −π/2，(E,U,−N) | 采用 Y-up 加根节点旋转：drei、camera-controls、GizmoHelper 默认 Y-up，全局改 DEFAULT_UP 会让这些辅助组件错位 | — |
| C5 | React 版本 | r14：锁 19.2（为 R3F v10 迁移留路）。n05：19.3.0（R3F 9.8.1 兼容，v10 alpha 与之冲突） | V0.1 用 19.3.0 + R3F 9.8.1；若 V0.3 决定迁到 R3F v10 且它仍要求 <19.3，再降到 19.2.x | 等 v10 rc |
| C6 | UI 基座与浮层动效 | d02：按 Radix Presence 只等 `animationend` 设计，要求把 6 类浮层改写为 keyframes，toast 主题化 Sonner。d04：采用 Base UI（base-mira），用 `data-starting-style/data-ending-style` 对接 transition，不用 Sonner | 以 d04 为准（Base UI、Base UI toast）；d02 的 token、配方、遥测显示策略、motion tier 全部沿用，radix.css 的 keyframes 改写不再需要 | Base UI 卸载时是否等待 transition 结束、各配方的前置态映射需要验证（G7） |
| C7 | lucide-react | d03：不安装，用 codemod 改写成兼容导出（依据 new-york-v4 的 16 个名字）。d04：shadcn 组件内部保留 lucide-react（与 lucide 版本一致），应用层用 AppIcon | MVP 用 d04 方案（改动小、升级 shadcn 无需重跑 codemod）；应用层一律用 `<Icon>/<StateIcon>`；V0.2 视包体再决定是否上 d03 codemod | base-mira 实际 import 的图标名清单需要重新统计 |
| C8 | 圆角 token | d01：`--radius: .75rem`；d04：radius small，控件约 .45rem | 控件用 .45rem（mira 高密度）；图卡大圆角由 LfChartCard 的 className 单独给 | — |
| C9 | 雾消光系数 | r04、r06、r11、r23：σ = 3.912/V（Koschmieder，2% 对比阈值）。r16：σ = 3.0/MOR（WMO 5% 对比阈值） | UI 与存储用 MOR（米，气象标准），σ = 3.0/MOR 作为唯一换算；传感器模型引用同一个 σ。若某个传感器文献用 2% 阈值，由该模块内部换算 | 写入环境契约（G8） |
| C10 | 协议名 | r27：`anet.rt.v1`；d05：与 ANet 事件名冲突，建议 `awr.rt.v1` | 改为 `awr.rt.v1`（awr = ANet World Runtime） | — |
| C11 | Mock 动力学方案 | r19：fake_uav 级联；r20：PX4-lite（与 SIH 对照验证）；r21：一阶速度跟踪；r23：Lee 几何控制加 FastPhysics；r24：MRS RK4 移植；n03：RotorPy 气动加 Crazyflow pipeline | L1（MVP）= PX4-lite 级联，放在 Crazyflow 式 pipeline 里，加 r23 的减速限速与抗饱和、r24 的 LineTracker；L2（V0.3–V0.4）= RotorPy 气动 stage 或 MRS 刚体，择一 | P600 参数辨识（V0.4 前） |
| C12 | DroneState 线格式 | r15：约 100 B；r19：40 B；r21：30 B swarm-lite；r27：Full64/Lite32；r20：对齐 PX4 字段 | 线格式以 r27 Full64/Lite32 为准；r20 的 PX4 语义与 r24 的 FlightFSM 放进 `flight_state`，`state_ext` 与 `safety` 放在低频 msgpack | 字段与状态映射表（G4） |
| C13 | Geometry World 查询实现 | x01、r24：numpy Height_map；r06：Open3D RaycastingScene（DSM 网格加 BVH，精确，持有 GIL）；r13：SVO 体素（前后端共用）；r04/r05：numpy z-buffer LiDAR | MVP 用 numpy Height_map 与 DSM（碰撞、AGL、安全转场）；V0.2 起 LiDAR 与精确 LOS 用 Open3D RaycastingScene，放在独立 sim-core 进程；V0.4 起前端相机防穿模用 SVO | 进程模型（G6） |
| C14 | 软件档点预算 | r09：250k；r10：300k；r11：60k；r12：40k（下限 10k）；n05：控制器收敛到下限 100k | 起步 40k，soft 档上限 150k；运行中由控制器决定 | 在本机固定负载下复测（G2） |
| C15 | UrbanScene3D 事实 | r01：六城都是 Z-up；r07：点间距差 1000 倍；r10：苏州轴向存疑；n02：旧金山疑似缩放 1:10；r05：芝加哥约 4.2×8×0.6 | 以 x01 为权威：苏州 Y-up；旧金山 10.15 m/单位且需 +90°；芝加哥 ×1000 且需调平 2.03°；规范化后点间距只差 5.4 倍 | 深圳和苏州的北向仍未验证 |
| C16 | 八叉树网格 G | r09、x01：G=64；Potree 与 n01 盒计数：spacing = cube/128 | 采用 G=64、LEAF=20000（x01 在六城规范化数据上实测，节点 658–852 个） | — |
| C17 | 浏览器 msgpack 实现 | r27：msgpackr；n05：@msgpack/msgpack（不开 records 时速度相当，且与 Python 互通） | @msgpack/msgpack | — |
| C18 | 单缓冲还是逐节点对象 | r11、r12：每节点一个 Points，平铺在一个 Group 下，可见节点上限 256；r13：统一页池加 DrawTable 单次 draw；n01：WebGL2 单缓冲单 draw（24 B/点，liveness 纹理） | 待实测。倾向：V0.1 先逐节点（ANET_Q16 零解码，实现最短），可见节点上限 256；超过 500 个节点时升级为页池加 DrawTable | G2 |
| C19 | 天空与大气 | r11、r16：SkyMesh 或解析渐变；n04：takram（npm 版与 r186 不兼容） | Low/Med 用解析渐变或 SkyMesh；Low 可用预烘焙全景；takram 只进 High 档，需修复后再用 | — |
| C20 | ENU 原点 | x01：XY 取包围盒中心、Z 取 DTM 中位数；r15：以锚点为原点，UrbanScene3D 用合成锚点并标 `synthetic` | 两者合并：真实数据以 RTK 或测量锚点为原点；合成数据的原点按 x01 规则取，并在 `coordinate.json` 标注 `anchor.kind = "synthetic"` 与示意经纬度 | — |

### 8.2 需要产品或项目负责人拍板的待决问题

| # | 问题 | 影响 | 建议默认值 |
|---|---|---|---|
| Q1 | 是否有 GPU 服务器（≥24 GB）可用于 LingBot-Map、DA3-Streaming、gsplat？何时可用？ | 决定 V0.5 真实重建与 3DGS 的时间表 | MVP 不依赖 GPU；GPU worker 以 Docker 交付，能力通过 capability 探测开启 |
| Q2 | 演示是否需要"真实飞行视频 → 重建 → 进入世界"这一步现场可跑？ | 01-design §43 原 Demo 依赖 GPU | MVP 演示用"内置世界 + Mock 重建 Job 动画 + DA3-SMALL 冒烟（可选）" |
| Q3 | 公开部署时如何处理 UrbanScene3D 数据条款（禁止再分发）？ | 能否在公网演示页内置数据 | 内网演示内置；公开版首次启动时在本地生成 World Package |
| Q4 | 真 P600 接入走哪条路线：Prometheus 地面站协议（纯 Python），还是 rosbridge（ROS1）？ | V0.5 接入工作量与安全模型 | 状态与控制走 Prometheus 协议后端（地面站 55556 心跳必须常驻）；需要话题级数据（LIO、点云）时补 rosbridge |
| Q5 | 多人协同：是否需要多个 operator 同时在线？ | 控制权、鉴权、Gateway 容量 | 单 operator 加多 viewer；Control Lease 与 operator 锁从 V0.2 起提供 |
| Q6 | 目标浏览器与硬件档位（是否保证移动端）？ | 画质阶梯与测试矩阵 | 桌面 Chrome/Edge 为主（WebGL2 必保，WebGPU 可选），不保证移动端 |
| Q7 | 暗色为默认主题、浅色只用于报告：是否认可？ | UI PRD | 认可（d01、d04 一致） |
| Q8 | 是否接受"一处红"规则，以及"状态不引入绿色和黄色"？ | 告警可辨识度 | 接受（形状、图标、文字三重编码，已做色觉异常校验） |

---
## 9. 完整性审查：进入架构设计与 MVP 实现前仍需补齐的缺口

审查口径：只列那些会直接影响架构决策或 MVP 代码、并且现有笔记给不出唯一答案（缺少细节、互相矛盾或未经验证）的问题。共 8 条，按影响排序。

| # | 缺口 | 为什么影响设计或实现 | 深挖单元 | 仓库与路径 | 期望产出 |
|---|---|---|---|---|---|
| **G1** | **渲染器路线未经端到端验证**：经典 `WebGLRenderer` + `WebGLNodesHandler` 承载 TSL 图层，对比 `WebGPURenderer` 全栈加 `gl_PointSize` 补丁 | 决定 RenderBackend、环境层是否要写两套（GLSL 与 TSL）、EDL 与云合成的实现方式、R3F `gl` 工厂写法。影响所有 Layer 的代码组织。已知 WebGLNodesHandler 不支持 MRT、RenderPipeline、storage texture 与 compute，但不知道它对 `vertexIndex` 扁平四边形、`PointsNodeMaterial`、自定义 `Fn` 雾、`onObjectUpdate` uniform 的支持度，也不知道它和 drei/R3F 9 组合后的表现 | r11、r12、r14、n01、r16、n05 | `refs/web3d/three.js/examples/jsm/tsl/WebGLNodesHandler.js`、`refs/web3d/three.js/examples/webgl_tsl_*.html`、`refs/web3d/three.js/src/renderers/webgl-fallback/nodes/GLSLNodeBuilder.js`（L1702）、`refs/web3d/three.js/src/renderers/WebGLRenderer.js`（L1077）、`refs/web3d/react-three-fiber/packages/fiber/src/core/configuration.ts`、`.cache/research/r11/`（edl.html、probe.mjs）、`.cache/research/r16/www/bench.html`、`.cache/research/n05/trial/` | 一页兼容矩阵（功能 × 后端），加上同一场景在三种组合下的固定开销、每对象开销、首帧编译耗时，给出最终 RenderBackend 决策 |
| **G2** | **PointCloudEngine 的实现形态与参数表需要本机实测对比** | 各单元给出的方案互相矛盾：选择器（n01 两级、r12 目标集加分段、r11 150 px 节点阈值）；GPU 组织（逐节点 Points 12 B/点、单缓冲 24 B/点加 liveness 纹理、页池加 DrawTable）；点径（cut 纹理每顶点最多 20 次 texelFetch，与 Lite 掩码相比在 SwiftShader 顶点瓶颈下代价不明）；控制器（n01 画质阶梯与 n05 AdaptiveBudget 叠加后是否振荡）；软件档预算（40k、60k、150k、250k）。这决定"极其流畅"在本机能承诺到什么程度（SwiftShader 25k 点约 27–37 ms/帧） | n01、r12、r11、r13、n05、x01 | `refs/discovery/voxelkloud-view/src/{lod/select.ts,quality.ts,cut.ts,sink-points.ts,stream-policy.ts}`、`refs/discovery/voxelkloud-core/src/`、`refs/web3d/potree-core/source/`、`refs/discovery/openlidarviewer/src/render/`、`.cache/research/r12-bench/`、`.cache/research/r13/r13_lod_sim.py`、`.cache/research/n05/trial/`、`.cache/research/r09_octree_fast.py` | 固定的 60 s 飞行脚本（深圳、纽约、上海），在负载可控条件下跑出各方案的 p50/p95、TTFP、收敛时间、空洞率、节点变化率；最终参数表；本机与真 GPU 分开的验收阈值 |
| **G3** | **World Package、ANET_Q16 v1、`coordinate.json` 没有统一 schema** | tiler、ingest、前端解码、环境层、重建都依赖这些字段，目前互相矛盾：`pos.w` 放 oct16 法线（r09、x01）还是强度或类别（r11）；`col.w` 放 class（r09、x01）还是 LOD 秩（r11）；类别编码用 LAS（2/64/65/1/9，x01）还是紧凑索引 0–15（n02）；`coordinate.json` 有 r15、x01、r02、r07 四套字段；`metadata.anet` 的扩展项（`levelsByteEnd`、`nnMedian`、`z_p1/p99`、tight bounds、多根森林 `roots[]`）；`world.json` manifest 尚未定义 | r09、x01、r15、r02、r07、r10、n01、n02 | `docs/research/r09-potreeconverter-pdal-lastools.md` §3.1、§3.3；`x01-urbanscene3d-data.md` §3.3、§3.6；`r15-cesium-deckgl-foxglove.md` §3.1.3；`r02-vggt-colmap.md`（Recon IR）；`refs/world/PotreeConverter/Converter/src/indexer.cpp`；`.cache/research/x01/analyze.py`；`.cache/research/r09_octree_fast.py` | `packages/contracts/schemas/{world,coordinate,pointcloud-metadata}.schema.json` 加校验器，外加一份六城实例 |
| **G4** | **DroneState、命令与状态机的语义没有统一** | Gateway、Mock、SIH、Prometheus 三类后端与 UI 共用这套语义，目前存在：PX4 `nav_state/landed_state`；Prometheus `UAVControlState`（ROS 消息用 0..3，线上结构体用 0..4）；r24 的 13 态 FlightFSM；r22 的机体生命周期；d05 的效果状态；r08 的定位状态。Full64 的 `flight_state/flags/authority` 需要映射表；各命令（Takeoff、Land、GoTo、FollowPath、Orbit、Hover、RTL、Velocity、SafetyStop、Pause）在三类后端上的实现方式与 ACK 语义也没有列表 | r27、r20、r21、r24、r19、r22、d05、r08 | `refs/sim/PX4-Autopilot/msg/versioned/VehicleStatus.msg`、`refs/sim/PX4-Autopilot/src/modules/commander/`、`refs/sim/MAVSDK/cpp/src/mavsdk/core/px4_custom_mode.hpp`、`refs/sim/Prometheus/Modules/uav_control/src/uav_controller.cpp`、`refs/sim/Prometheus/Modules/common/prometheus_msgs/msg/`、`.cache/research/r24/mrs_uav_managers/`（`control_manager.cpp`、`uav_manager.cpp`）、`refs/design/ANet`（ANetCore effect） | 状态枚举与映射表、命令能力矩阵（按后端）、`layouts.json` v1 定稿 |
| **G5** | **后端进程模型与进程间通信未定义** | r27 的 Gateway 是单核 asyncio，MVP 设想"进程内"；r06 要求 Open3D RaycastingScene 放独立进程（持有 GIL，会卡住 WS 几十 ms）；small_gicp 同样持有 GIL（r07）；规划要求单线程 BLAS 多进程（r25）；FleetSim 以 100–250 Hz 步进。尚未定义 sim-core 与 Gateway 之间怎样以 50 Hz 以上共享 SoA 状态（`multiprocessing.shared_memory` 环形缓冲、V0.1 就用 zenoh（实测 RTT 0.4 ms、15.6 万 msg/s）或其他方案），也未定义命令与事件的可靠通道、进程监管与重启 | r27、r06、r07、r25、r21、r22、n03 | `refs/backend/zenoh`、`.cache/research/r27/`（`gw_proto.py`、`z_bench.py`）、`.cache/research/r06/`、`refs/discovery/crazyflow/crazyflow/sim/`、`refs/sim/MAVSDK`（asyncio 线程池陷阱，见 r21） | 进程拓扑图、IPC 选型与基准（1000 架 SoA @50 Hz 到 Gateway 的延迟与 CPU）、失败与重启语义 |
| **G6** | **环境场 E(x,y,z,t) 没有统一的服务端与前端数据契约** | 物理、传感器、前端视觉、协议、World Package 共用同一份环境数据，目前存在：雾 σ = 3.0/MOR 与 3.912/V 两种定义；风体纹理第 4 通道放 solid fraction（r17）还是 \|u\|（n04）；r27 的 `env/wind/field` 只有 3 个 f16 分量；WindSpec 的"风从何处来"方向约定（r16、r17）；湍流是每机 Dryden（n03）还是共享冻结湍流盒（r17）；流线 `streamlines.bin`（n04）；`presets.json` 与 `derive` 的前后端对拍；天气状态推送频率（r16 为 5–20 Hz，r27 为变化时推送） | r16、r17、n04、n03、r27、r23、r04 | `.cache/research/r17/`（`r17_windfield_api.py`、`r17_masscons2.py`）、`refs/weather/windninja/src/ninja/`、`refs/weather/natural-disasters/src/weather/`、`refs/weather/Eanpa-Sky/engine/weather_system.js`、`refs/discovery/rotorpy/rotorpy/wind/dryden_utils.py`、`refs/discovery/cesium-wind-layer`、`.cache/research/n04/` | `packages/contracts/env/{env_state,wind_field,presets}.schema.json`、`environment.query` 的 API 定义、前后端 parity 测试用例 |
| **G7** | **设计体系在 Base UI 上的落地细节未验证** | d02 按 Radix（只等 `animationend`）与 Sonner 设计，d04 选的是 Base UI（transition 配合 `data-starting-style/ending-style`）与 Base UI toast，两者前提不一致。需要确认：Base UI 卸载时是否等待 transition 结束；32 个配方的前置态和关闭态如何映射到 Base UI 属性；Tailwind v4 `--ease-*` 与 transitions.dev 同名 token 的统一写法；base-mira 组件实际 import 了哪些 lucide 名字（d03 的 16 个名字来自 new-york-v4）；lieflat `LfChartCard` 在 mira 高密度下的字号与间距 | d02、d04、d03、d01 | `refs/design/transitions.dev/skills/transitions-dev/`（`_root.css`、`05/06/07/17/21/22-*.md`）、`refs/design/ui/apps/v4/registry/bases/base/ui/`、`refs/design/ui/packages/shadcn/src/tailwind.css`、`@base-ui/react` 源码（`node_modules`，n05 trial 已安装）、`refs/design/morphicons/src/`、`.cache/research/d04/`（`anet-base.json`、`mirror.sh`） | 一个可运行的 UI 样板页（Dialog、Sheet、Dropdown、Tooltip、Tabs、Toast 加上动效 token、StateIcon、LfSparkline），附 reduced 与 lite 档截图 |
| **G8** | **Mock FleetSim 的组合规格与参数标定未收敛** | MVP 的无人机运动全部依赖它，但各单元的实现并存：PX4-lite 级联（r20，已与 SIH 对照）、减速限速与抗饱和（r23）、LineTracker STOP_MOTION 与 Safety 阈值（r24）、Crazyflow 式 pipeline（n03）、风阻力方程系数 CdA 与 c_rd、Dryden 离散方式（r17 在 dt=0.02 时 σ 偏差约 10%）。需要确定 stage 顺序、控制与物理步长（250 Hz 还是 100 Hz）、1000 架 CPU 预算，以及 P600 占位参数（x500 2.064 kg 与 SDF 1.505 kg 不一致）的采用规则 | r20、r23、r24、n03、r19、r17 | `.cache/research/r20/mock_px4lite.py`、`.cache/research/r23/`（`exp2.py`、`fastphys.py`）、`.cache/research/r24/mrs_mock.py`、`.cache/research/r24/mrs_multirotor_simulator/`、`refs/discovery/crazyflow/crazyflow/sim/pipeline.py`、`refs/discovery/rotorpy/rotorpy/vehicles/multirotor.py`、`refs/sim/Prometheus/Simulator/gazebo_simulator/gazebo_models/uav_models/p600/` | `sim/fleet` 规格文档，加上 SIH 对照回归用例（阶跃、GoTo、8 m/s 侧风）和 `vehicles/p600/params.yaml` 初版（带来源与置信度列） |

**已在本次整合中关闭的疑点**：
- Starlette `StaticFiles` 是否支持 HTTP Range：已实测支持 206 与字节一致性。
- WebGLNodesHandler 的能力边界：已从源码确认。
- 旧金山单位与苏州轴向：以 x01 的地标证据为准。
- React 与 R3F 的 peer 兼容：以 n05 的 ERESOLVE 实测为准。

---

## 10. 附录

### 10.1 关键实测数字速查（设计取值的依据）

| 领域 | 数字 | 来源 |
|---|---|---|
| SwiftShader 点吞吐 | 约 0.3–1 M 点/s；每个 draw 约 40–60 µs；8-tap 720p EDL 约 107 ms | r12 |
| 实例化代价 | SwiftShader 上实例化四边形比 GL_POINTS 慢 16–80 倍；非实例化 vertex-pulling 只慢 2–4 倍；实例化雨每实例 50–100 µs | r11、r12、r16 |
| 每对象 CPU | 经典 WebGLRenderer 约 14 µs；WebGPURenderer 约 22 µs（WebGPU 后端）/ 约 29 µs（WebGL2 后端）；BundleGroup 约 6 µs | r11 |
| 固定帧开销（软件档） | WebGLRenderer 8.9 ms；WebGPURenderer 用 HalfFloat 输出时 84–100 ms，用 UnsignedByte 输出时 36–64 ms | r11 |
| n05 trial（负载约 5） | 10–15 万点 5.5–6.6 fps；200 万点 0.5–0.7 fps；TTFP 95–850 ms；控制器约 2 s 收敛 | n05 |
| 切片 | 5M 点建树 5–6 s；每城 `octree.bin` 约 60 MB；`hierarchy.bin` 15–19 KB；首屏 1.3–5 MB | r09、x01 |
| 协议 | 1000 架 raw struct 编码 5.7 µs、解码 15 µs；JSON 编码 40.7 ms；credit W=2 时 p95 138 ms（push 模式下延迟累积到 4.2 s） | r27 |
| Gateway 容量 | 200 架加 10 个客户端时 CPU 34%，p95 17 ms；50 个客户端时平滑降级 | r27 |
| Mock 动力学 | PX4-lite 1000 架 @250 Hz 单核 3.5 ms/步；MRS 移植 400 架 @100 Hz RTF 1.6 | r20、r24 |
| PX4 SIH | 每机约 0.22 核、10 MB；8 机以内 RTF ≥0.92；16 机 RTF 约 0.72 | r20、r22 |
| ANet | 一次委派往返 921–1016 ms；每个 daemon 约 14 MB | d05 |
| LiDAR mock | Open3D RaycastingScene 20k 射线约 6 ms；numpy z-buffer 9–80 ms | r06、r04、r05 |
| 风场 L2 | SF 1.34M 格每个方向 7–12 s；Shenzhen 4.64M 格约 30 s；散度 ≤3e-6 | r17 |
| UI | CPU canvas 12 图 × 1200 点 @10 Hz 保持 60 fps；叠加 blur 后 WebGL 帧率降约 65%；1000 个图标挂载 14.5 ms（morphicons 静态）对 61.5 ms（lucide-react） | d01、d02、d03 |

### 10.2 后续文档如何引用本索引

- **系统架构说明书**：§0、§3.0、§3.1、§3.6、§3.9、§3.11、§8 C1/C4/C13，以及 G1、G5 的结论。
- **技术选型说明书**：§2、§4、§6、§8。
- **业务逻辑设计说明书**：§3.9、§3.10、§3.12、§3.14，以及 G4、G8。
- **UI 交互 PRD**：§3.13、§3.5（HUD 与质量预设）、§3.6（相机与标签）、§8 C6–C8，以及 G7。
- **产品 PRD**：§0、§3.14（剧本）、§7 P0 第 8 条（验收）、§8.2（待决问题）。
- **分模块 PRD**：按 §3.2–§3.14 逐节展开，各节的"来源单元"就是深读入口。

---

## 补充深挖

- [G6 环境场 E(x,y,z,t) 统一数据契约](g06-gap.md)：MOR 为唯一能见度真值（σ=ln20/MOR，状态量改为不含降水的 `mor_bg_m`，降水 σ 加性叠加，3.912 仅用于 Kim 波长换算）；风体纹理定为 AWRV v1 RGBA16F（rgb 按流体占比预乘，A=实体占比，WebGPU 不支持 RGB16F）；方向只用气象来向 `dir_from_deg`、矢量一律去向 ENU；湍流默认共享冻结 von Kármán 盒（Dryden 改精确离散，仅作回归，不照搬 RotorPy）；流线定为 AWSL v1；`env/state` 改为关键帧变化推送加 1 Hz 心跳，两端用同一 `eval_env`/`derive` 纯函数并以 golden 对拍（JS/Python 实测 1e-12 一致）；另发现 r17 的 SF 风场库建在原始单位上，需在 x01 规范化后重建。
- [G4 DroneState、命令与状态机语义统一](g04-gap.md)：按七轴正交建模，FlightState 定稿为 14 态加 3 位子模式；Full64 的 flags 定为 ARMED/IN_AIR/LOC_OK/FAILSAFE/GCS/FCU/LOC_DEG/ALERT，byte 7 改为 ctrl（owner/locked/native/pose_src）；给出 PX4、Prometheus、Mock、MRS、生命周期、定位的映射表，以及 10 个命令在 3 类后端上的实现与 ACK 矩阵（ACK 用 accepted/running/succeeded 表示，对应 ANet 效果与 V0–V4 信任级别）；Prometheus 的 RTL 必须由 Gateway 仿真，发 UAVCommand 前必须确认 control_state==2。
- [G3 World Package / ANET_Q16 v1 / coordinate.json 统一 schema](g03-gap.md)：冻结 v1 契约，包括 6 份 JSON Schema 2020-12（world、coordinate、pointcloud-metadata、class-table、grid、common）、跨文件语义校验器和六城实例，全部通过校验，Ajv strict 也能编译，产物在 `.cache/research/g03/`。主要裁决：`pos.w` 存 oct16 法线，0 表示无法线；`col.a` 存类别紧凑索引，不存 LOD 秩，秩就是打散后的点下标；`anet-classes@1` 在运行时用 0–15 索引，与 LAS 码双向映射，屋顶取 LAS 6，x01 的 65 作废；`coordinate.json` 合并六套字段，"world" 只指 ENU@anchor，重建与 LIO 帧改名为 `map`/`engine`（`T_world_map`、`T_world_engine`），矩阵统一为行主序 `T_to_from`；多根森林放在 world.json 的 `roots[]`，每个根是独立的 Potree 容器，首屏预算按根数均分（苏州 6 根，首屏 16.7 万点、2.0 MB）；`hierarchy_ext.bin` 与 `hierarchy.bin` 逐条镜像，存子树包围盒。另外核实到 potree、potree-core、three-loader 会把 ANET_Q16 当作 DEFAULT 解码，要对照只能用 `--twin-default` 孪生容器；SF 单位定为 10.15 并已重建。
- [G5 后端进程模型与进程间通信](g05-gap.md)：V0.1 起 api 与 sim-core 分进程，Open3D、small_gicp、规划、MAVSDK 各进独立进程，由自研 supervisor 统一监管；1000 架状态走 tmpfs mmap seqlock 环 StateRing（100 Hz，Gateway CPU 1.5%，tick 年龄 p99 11 ms），命令、事件与在线状态从 V0.1 起走 zenoh（DROP + seq/replay + cid 幂等，sim-core 永不阻塞）；挂死靠主循环心跳检出（liveliness 检不出），checkpoint 加 epoch 重启，实测 kill -9 后 0.8 s、挂死 3.4 s 恢复出帧。
- [G8 Mock FleetSim 组合规格与参数标定](g08-gap.md)：所有 Mock 档位共用 PX4-lite 外环（PX4 原生 ARW，r23 条件积分退役），补上 PX4 time_stretch 后阶跃、GoTo、8 m/s 风 17 项指标全部落入 SIH 容差（阶跃 RMSE_x 0.08–0.10 m，GoTo 0.34–0.40 m，俯仰 −7.85°/−7.87°）；世界主时钟 250 Hz，L1 与 StateRing 发布为每 2 个 tick（125 Hz），env/guard 50 Hz；采用 STOP_MOTION，围栏按折线 [p, p_stop, goal] 校验；组合气动 x500 取 CdA 0.02065 m²、c_rd 8.06428e-5；Dryden 必须用标准化状态（g06 原写法在空速变化时 σ 翻倍并出现 39 m/s 尖峰，r17 所说 10% 是样本误差）；FastGuard 的 pos_err 阈值改为 3.0 m/5.0 m（r24 的 1.5 m 在中度湍流下会误报）；L1 热路径用 numba，1000 架全 pipeline 约 0.26 核（纯 numpy 约 1 核）；x500 保留为回归机体，P600 独立建 profile（3.5 kg、19.23 N/桨、Prometheus 限速 3 m/s），SDF 的 1.505 kg 未通过自洽检查。
- [G7 设计体系在 Base UI（base-mira）上的落地细节](g07-gap.md)：已验证 Base UI 卸载前会等待 Popup/Root **自身**的 CSS transition 和 animation（`getAnimations().finished`，实测 finished 后 2–13 ms 卸载；不含子树和遮罩）。d02 的 Radix keyframes 层作废，改为 `[data-starting-style]` 表示前置态、默认规则表示打开终态和打开时长、`[data-ending-style]` 表示关闭态和关闭时长（07/17/21/22 需要对调时长），放在 `@layer motion`；reduced 档直接 0s。token 用 `@theme static`（不写 static 时 `--ease-out` 等不会输出），并为 `--transition-duration-*` 建别名；`cn` 必须用 `createCn` 登记 `text-hud-*` 等名字，否则字号不生效，还会吞掉颜色类。base-mira 实际 import 的 16 个 lucide 名与 new-york 不同（多了 toast 的 4 个类型图标和 ChevronLeft 等）。LfChartCard HUD 规格定为 `size="sm"`、`rounded-xl`、13/11/22/10 字阶。可运行样板页在 `.cache/research/g07/sample`。
- [G1 渲染器路线端到端验证与 RenderBackend 定案](g01-gap.md)：定案为 Tier A 用 WebGPURenderer，Tier B/S 用经典 WebGLRenderer 加 AnetNodesHandler（WebGLNodesHandler 子类，修复 RT 双重 sRGB 编码和 scene.fogNode 被忽略两个问题），所有图层（含 G2 PointPool 点云、EDL、云合成）只写一套 TSL。28 项功能测试中，vertexIndex 四边形、PointsNodeMaterial、Fn 雾、onObjectUpdate、深度纹理与 reversed-Z、Line2NodeMaterial 在三种后端下逐像素一致。EDL 和云合成改为"显式 RT + 全屏四边形"实现，不再需要 GLSL 版；点径在经典路径用 GLPointsNodeMaterial（onBeforeCompile 加缓存键后缀，只用公开 API）。WebGPURenderer 在软件档的固定开销为 34–42 ms（经典 3 ms），可用 direct 输出模式消除。PointPool 的 TSL 版与 GLSL 版速度相当（1.47 对 1.40 µs/点）。handler 下的约束：int uniform 失效、每个 InstancedMesh 需独立材质、逐对象 uniform 会触发 UBO 重传。R3F 必须用 async gl 工厂；drei 的 GizmoHelper、Line、Text 在 WebGPURenderer 下会失效或崩溃。
- [G2 PointCloudEngine 实现形态与参数表的本机实测定案](g02-gap.md)：选择器定为 APH，即 voxelkloud 两级预算，加上“第一个被预算拒绝的 required 节点画前缀”和 ±10% 迟滞。三城 60 s 流式仿真中，APH 填充率 ≥ 0.99，r12 目标集方案在流式期间只有 0.71–0.98；τ 受限区空洞率 1.1–1.2%，目标集方案为 2.1–3.4%。GPU 组织定为 PointPool（RGBA32UI）加 DrawTable 二分，用无属性 Points 一次 draw，配对实测与逐节点、multi-draw 等速；单缓冲 liveness 因高水位多画顶点，慢 18–50%，不采用。点径全部用 Lite：在“缩小最多 1 级”的前提下与 cut 遍历数学等价，画质指标完全一致，cut 反而贵 3–8%；maxPx 按 limitedBy 在 8 与 16 之间自适应。控制器改为级联 CAS：内环在档内做对数 AIMD，外环只在饱和时换档，换档无扰。仿真与本机实时都证实 QL+AB 会出现锯齿（B 每分钟反向 23–46 次）并在 iGPU 上档位翻转；CAS 实测 p95 50 ms，没有换档。Tier S 固定 0.5 渲染比例，预算带 [10k, 40k]，起步 25k，250k 否决。另给出 flight60 六段基线脚本，以及本机与真 GPU 分开的验收阈值。
