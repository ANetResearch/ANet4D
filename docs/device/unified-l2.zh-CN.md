# 统一 L2 物理引擎（具身设备引擎，第 5 步）

## 1. 为什么要自己的 L2

第 4 步的保真档表里，`l2`（Realistic）只有 gz-sim 锁步一条路，且 gz 只有 `x500_d4d` 一个模型。L2 是本系统的核心档位（空、陆、海设备都要在"有惯量、有执行器、有气动/水动力"的物理里跑），只靠一个外部引擎、一个机型既不够用，也没有原创性。本步起，L2 由我们自己的引擎给出：`awr.phys.l2`，一个刚体核、一个档位类、每个设备族一个域模型。**定位（2026-11-04 定）：统一引擎就是 Realistic 档**，它是多个引擎的集成加我们的逻辑：gz 能用的部分（多旋翼转子模型、碰撞体、风阻力律、x500 参数集）作为组件吸收进来，gz 本身只做对拍基准；gz 没有的设备换别的引擎逻辑集成（车辆动力学、Fossen 艇模型、系数表固定翼），再没有的按 gz 等的逻辑自己为该设备补特异性模型；用不上的部分（如坠毁后的翻滚接触）不追求。等三个域都过门后，`physics = "realistic"` 这个名字切到统一引擎。

```
awr.phys.l2
├── rigid.py        六自由度刚体核（ENU / FLU，四元数，半隐式欧拉；空陆海共用）
├── multirotor.py   多旋翼域模型：转子级执行、PX4 级联、控制分配、复合气动
├── ugv.py          地面车域模型：单轨动力学 + 线性轮胎 + 载荷转移、转向 / 驱动限制、阻力、地形跟随；GroundTracker
├── ground.py       地面车机型包（vehicles/ground，awr.vehicle.ground.v1）
├── st_oracle.py    对拍基准：CommonRoad 单轨（ST）方程 + RK4（f1tenth_gym 形式）
└── tier.py         L2Tier：PhysicsTier（与 Fast、gz 同一指令、同一参考跟踪器、同一接触规则、同一 PoseHistory），空陆同一步进循环
```

## 2. 多旋翼域模型

L1（PX4-lite，`awr.sim.fleet`）把机体压成"一个推力标量 + 理想速率环"；L2 把它展开：

| 环节 | L1 | L2（本引擎） | gz（交叉验证） |
|---|---|---|---|
| 速度环 | PX4 PID（P/I/D、前馈、ARW），250 Hz | **同一 PID、同一限幅，引擎内按物理步执行**（`RefTracker(inner_pid=True)` 只给参考） | Lee 速度 P；I/D 折成速度当量按 tick 叠加 |
| 姿态环 | 四元数 P（`MC_ROLL_P`…）+ 速率限幅 | 同一律 | Lee 姿态 |
| 速率环 | 无（角速度即设定） | **PX4 速率 PID**（`MC_*RATE_P/I/D`，归一化力矩经 PX4 控制分配的同一归一化换算成 N·m）+ 陀螺前馈 | Lee 速率 |
| 控制分配 | 无 | **效率矩阵伪逆**（转子位置 `arm_xy_m`、PX4 `quad_x` / `hex_x` 布局、旋向、`km_m`），饱和时先放弃偏航再平移集体推力 | gz 多旋翼插件 |
| 执行器 | 一阶滞后，一个推力 | **每个转子一阶滞后 `tau_s`，推力 `k_f·Ω²`** | 每转子一阶 |
| 刚体 | 质点 + 姿态积分 | **惯量 `inertia_kgm2`，欧拉方程（含 ω×Jω），四元数由机体角速度积分** | ODE 刚体 |
| 气动 | 复合：½ρ·CdA·‖v_r‖·v_r + ΣΩ·c_rd 面内阻力（ΣΩ 按均匀推力估计） | **同一律，ΣΩ 取实际转子转速** | 同式插件 |
| 接触 | D2 contact | 地面：高程 + 起落架高，摩擦 μ=0.8、起落架调平、触地 > 3 m/s 判 IMPACT；世界几何：最近点距离下界门控，碰撞半径内且不在机体下方 45° 锥内即 COLLISION_WORLD | gz 碰撞体 |

参数全部来自机型包（`vehicles/<dir>/params.yaml`），引擎里没有按机型调的常数。坐标：世界 ENU、机体 FLU、四元数 xyzw（WORLD←FLU），PoseHistory 行与另外两档完全同布局。

性能：numpy 按机群向量化，单机 250 Hz 约 30 ms/tick（100 ms 仿真），即 3 倍实时；机数增加几乎不增加时间。还没有 numba 核。

## 2b. 地面车域模型（5b）

gz 没有可以搬的车模型，所以这一域集成的是车辆动力学文献里的标准模型：CommonRoad 的单轨（ST，bicycle）模型 + 线性轮胎 + 纵向载荷转移，参数口径与 f1tenth_gym 完全一致（`c_sf / c_sr` 为单位法向载荷的侧偏刚度，`mu` 摩擦系数，`h` 质心高）；再补上沙盒需要、ST 模型没有的部分：

| 环节 | ST 模型（CommonRoad / f1tenth_gym） | 统一引擎 ugv 域 |
|---|---|---|
| 状态 | (x, y, δ, v, ψ, ψ̇, β)，专用 ODE | **同一刚体核**（p, v, q, ω），力与力矩进 `RigidState.integrate`；滑移角方程里的 −ψ̇·v_x 耦合项来自世界系积分 |
| 轮胎 | 线性：F_y = μ·c_s·F_z·α，无饱和 | 同一线性律写成机体速度形式（倒车也成立），**饱和于 μ·F_z** |
| 载荷转移 | F_zf = m(g·l_r − a·h)/L，a 取输入 | 同式 |
| 低速 | \|v\| < 0.5 m/s 切换运动学单轨（否则 1/v 项爆） | **横向 / 偏航子系统每步隐式求解**（后向欧拉，线性 2×2），低速极限自然退化为运动学单轨 r = v·δ/L，不切模型 |
| 纵向 | v̇ = a；a 受 a_max、功率段 a_max·v_switch/v、v_min / v_max 约束 | 同一约束 + 牵引 μ·g + 滚阻 c_rr·g + 气动 ½ρ·CdA·\|v_r\|·v_r（相对风）+ 坡度重力分量 |
| 转向 | δ̇ 受 sv_max，δ 受 s_max | 同 |
| 竖直 / 姿态 | 无（平面） | z = 地面高 + 质心高；机体 z 轴沿地形法向（轴距与轮距四点探高），车在坡上就是坡上的车 |
| 障碍 | 无 | DSM 一步内抬升 > 0.3 m（路缘、墙脚）→ COLLISION_WORLD；网格最近点查询从质心上方 r 处发出（脚下的地面不算障碍），碰撞半径内即 COLLISION_WORLD |
| 停车 | 速度指数衰减，永不为零 | 指令停车且 \|v\| < 0.05 m/s 时静摩擦锁死，速度精确为零（坡上刹车能停住） |

参考与控制：指令集相同（velocity、goto、follow_path、orbit、hover、kill；takeoff / land 对车返回 630；kill = 紧急制动）。`GroundTracker` 把共用的 M24 跟踪器（`awr.phys.tracker.Tracker`，运动学档同一套 goto / 航点 / 环绕参考逻辑）的速度设定换成"速度 + 航向"；引擎里的航向环给出偏航角速度需求再换成转向角 δ = atan(L·r_sp / v_x)；速度环是"按可用加速度 90 % 斜坡的参考 + 前馈 + PI"（积分项在坡上把速度拉回设定，顶到限幅时冻结）。

机型包是独立的一族（`vehicles/ground/catalog.json` + `vehicles/ground/<id>/params.yaml`，schema `awr.vehicle.ground.v1` / `awr.device_catalog.ground.v1`），不扩展多旋翼的 `awr.vehicle.v1`：沙盒里的机巢、任务、UI 都是按多旋翼 / 固定翼写的，车混进同一目录会牵动一大片；只有统一引擎与保真档表读这个目录。三台车：

| id | 原型 | 参数等级 |
|---|---|---|
| `f1tenth` | F1TENTH 级 1/10 阿克曼车 | 动力学参数全部 A 级（f1tenth_gym 缺省：m 3.74 kg、I 0.04712、l_f 0.15875、l_r 0.17145、h 0.074、μ 1.0489、C_Sf 4.718、C_Sr 5.4562、s_max 0.4189、sv_max 3.2、v_max 20、a_max 9.51、v_switch 7.319） |
| `osr` | JPL 开源漫游车级六轮车（四角转向，按等效自行车处理） | 速度 ~1.6 m/s C 级（项目文档）；质量、几何、轮胎 D 级模拟参考值 |
| `podcar` | OpenPodcar 级代步车底盘载人 pod | 最高 15 km/h A 级（OpenPodcar2 摘要）；其余 D 级 |

## 3. 与 Fast 对拍

`python -m awr.phys.tier_compare` 的九个一致性用例（M24 §6.10 口径：起降、悬停、速度阶跃、限幅、偏航阶跃、航点、环绕、撞墙与失能坠地、恒风悬停）在 Fast 与 L2 上用**同一段驱动代码**运行，`report --tiers fast,l2`：

| 用例 | 接口判定 | 档间差 | 容差 | 结果 |
|---|---|---|---|---|
| takeoff_land | 一致 | 到达 0.05 s、触地 0.004 s、触地速度 0.00 m/s、位置 RMS 0.07 m | 1 / 1 / 0.3 / 1.5 | 通过 |
| hover | 一致 | 漂移 0 | 0.5 | 通过 |
| velocity_step | 一致 | 上升时间 0.17 s（Fast 1.06，L2 0.89）、超调 1.0 % | 0.3 / 10 | 通过 |
| velocity_limit | 一致 | 稳态 0.08 m/s（11.86 对 11.94，上限 12） | 1.0 | 通过 |
| yaw_step | 一致 | t90 0.12 s | 0.2 | 通过 |
| waypoints | 一致 | 位置 RMS 0.32 m、到达 0.06 s/100 m | 1.5 / 1.0 | 通过 |
| orbit | 一致 | 位置 RMS 0.17 m、半径 RMS 0.006 m | 1.5 / 0.5 | 通过 |
| contact_crash | 一致（COLLISION_WORLD、IMPACT） | 撞墙时刻 0.05 s、位置 0.38 m | 0.5 / 0.5 | 通过 |
| wind_hover | 一致 | 倾角 0.00°（两档都是 7.85°，SIH 标定值）、位置 RMS 0.02 m | 0.5 / 0.5 | 通过 |

接口判定 = 指令回执、状态序列、接触事件种类、PoseHistory 网格、指令生效步全部相同。两档的差异只来自内环与动力学（L2 多了姿态与转子滞后，所以速度阶跃起步稍慢、上升段稍快）。曲线见在线页 `/anet4d/device/l2-multirotor/`。

## 3a. 与 gz 对拍（x500，dmax d3-gz 容器，gz-sim 8.15）

同一批运行里加上 `run --tier realistic`（launcher = local），`report --tiers l2,realistic`：

| 用例 | 接口 | L2 ↔ gz 档间差 | 参照：Fast ↔ gz | 结果 |
|---|---|---|---|---|
| takeoff_land | 一致 | 到达 0.15 s、触地 0.004 s、触地速度 0.00、位置 RMS 0.08 m | 0.10 / 0.00 / 0.00 / 0.01 | 通过 |
| hover | 一致 | 漂移 0.04 m（gz 侧） | 0.04 | 通过 |
| velocity_step | 一致 | 上升时间 0.09 s（L2 0.89、gz 0.98）、超调 0.4 % | 0.08 / 0.6 % | 通过 |
| velocity_limit | 一致 | 稳态 0.93 m/s（L2 11.94、gz 11.00；上限 12） | 0.86（Fast 11.86） | 通过 |
| yaw_step | 一致 | t90 0.02 s | 0.10 | 通过 |
| waypoints | 一致 | 位置 RMS 0.75 m、到达 0.17 s/100 m | 0.44 / 0.12 | 通过 |
| orbit | 一致 | 位置 RMS 1.0 m、半径 RMS 0.014 m、到达 0.36 s/100 m | 1.1 / 0.02 / 0.37 | 通过 |
| contact_crash | 一致 | 撞墙时刻 0.03 s、位置 0.45 m | 0.02 / 0.24 | 通过 |
| wind_hover | 一致 | 倾角 0.00°（都是 7.85°）、位置 RMS 0.014 m | 0.00 / 0.02 | 通过 |

L2 ↔ gz 的差与原有的 Fast ↔ gz 同量级或更小。限幅用例里 gz 只到 11.0 m/s：它的速度积分是 RefTracker 按 tick 折算的速度当量，顶到限幅就被反算掉，补不了阻力；L2 在引擎内按物理步跑 PX4 的 PID，到 11.94，更接近 PX4 本来的行为。运行用的镜像是 `d3-gz:dev` 加 pyzmq、jsonschema、numba、protobuf 4.25（原镜像的 protobuf 4.21 缺 `GetMessageClass`，`awr.phys.gz.protocol` 导入即失败）。

## 3b. 地面域：与 CommonRoad ST 基准对拍（5b）

`python -m awr.phys.tier_compare run --set ground --tier l2|st`，`report --set ground --tiers l2,st`。基准 `awr.phys.l2.st_oracle` 跑 f1tenth_gym 形式的 ST 方程（RK4，|v| < 0.5 m/s 切运动学单轨），加上引擎同样的阻力与坡度项；**控制器与引擎共用**（同一速度环、航向环、转向限制），所以档间差只来自动力学模型与积分方式，而不是来自"两个不同的司机"。8 个可对拍用例全部通过（接口判定 + 容差），撞墙用例只有引擎（基准无世界）：

| 用例 | 车 | 接口 | L2 ↔ ST 档间差 | 容差 | 结果 |
|---|---|---|---|---|---|
| g_speed_step 0 → 3 m/s → 停 | f1tenth | 一致（GROUNDED → MOVING → GROUNDED） | 上升时间 0.00 s（都是 0.28 s）、超调 0.0 %（0.8 %）、位置 RMS 0.004 m | 0.3 / 10 / 1.0 | 通过 |
| g_speed_limit 30 m/s 指令按 20 限幅 | f1tenth | 一致 | 稳态 0.00 m/s（都到 20.0，最大 20.006） | 1.0 | 通过 |
| g_goto_stop 到 (20, 10) 停车 | f1tenth | 一致 | 到达 0.004 s（7.73 s）、停车误差 0.45 m、位置 RMS 0.012 m | 1.0 / 1.0 / 1.0 | 通过 |
| g_waypoints 20 m 方形，3 m/s | f1tenth | 一致 | 位置 RMS 0.27 m、到达 0.27 s/100 m | 1.5 / 2.0 | 通过 |
| g_orbit R 6 m，2 m/s 一圈 | f1tenth | 一致 | 位置 RMS 0.003 m、半径 RMS 差 0.001 m（都是 0.225）、绕圈时刻 0.00 | 1.5 / 0.5 / 2.0 | 通过 |
| g_slope 15 % 坡 2 m/s | f1tenth | 一致 | 定速 0.00 m/s（都是 1.998）、地形跟随误差 0.000 m、位置 RMS 0.005 m；引擎倾角 8.53°（= atan 0.15），基准平面 | 0.3 / 0.05 / 1.0 | 通过 |
| g_speed_step_podcar 0 → 3 → 停 | podcar | 一致 | 上升时间 0.00 s（都是 2.32 s，a_max 1.2）、超调 0.0 % | 0.5 / 10 | 通过 |
| g_waypoints_osr 20 m 方形，1.2 m/s | osr | 一致 | 位置 RMS 0.04 m、到达 0.03 s/100 m | 1.5 / 2.0 | 通过 |
| g_wall 3 m/s 撞墙 | f1tenth | 引擎独有 | COLLISION_WORLD（mesh）于 x = 15.68（墙面 16，碰撞半径 0.33），状态 GROUNDED → MOVING → CRASHED | — | 通过 |

直线用例几乎逐位一致（纵向两边都是 v̇ = a），差异集中在转弯（航点 0.27 m）：这是刚体 + 隐式轮胎与 ST 专用 ODE + RK4 之间的差。性能：单车约 25–35 ms / 100 ms tick（引擎与基准相当）。

## 4. 怎么用

- Env：`spec.physics = "l2"`（与 `kinematic / fast / realistic` 并列），机型由 `fleet[].vehicle_ref` 给出：多旋翼目录里的机型走多旋翼域，`vehicles/ground` 里的 id（`f1tenth`、`osr`、`podcar`）走地面域，空陆可以混编在同一机队；固定翼 / VTOL / 艇仍是 630。
- 直接用：`L2Tier(uavs, spawn, world, grounded=…, vehicles=…, ground=…)`，接口同 `FastTier`；`set_wind(n×3)` 每 tick 写风；车永远落在地面上（`grounded` 只对空中设备有意义）。
- 状态：车新增 `VehicleStatus.MOVING`（行驶中），静止为 GROUNDED，撞上为 CRASHED；Env 的 `status.mode` 对应 `driving`。
- 对拍：`python -m awr.phys.tier_compare run --tier l2 --out DIR`，`report --out DIR --tiers fast,l2`；地面域 `run --set ground --tier l2|st`，`report --set ground --tiers l2,st`。
- 保真档表：`python -m awr.sim.device ladder` 的 `l2` 列按"统一引擎有没有该设备族的域模型"判定，x500 另标 `+ gz`；陆行来自 `vehicles/ground`。

## 5. 接下来（同一刚体、同一档位）

按用户定的顺序：

1. ~~地面车 `ugv`~~：5b 已做（§2b、§3b）。还没做的：差速 / 滑移转向的专用运动学（现在 `drive.type` 只是标注，六轮漫游车按等效自行车处理）、Pacejka 非线性轮胎、悬挂。
2. **无人艇 `usv`**：Fossen 三自由度（附加质量、线性 + 二次阻尼、双推进器分配）+ 风、流；参考 BlueBoat、NAC-USV。
3. **固定翼 `fixed_wing` / `vtol`**：升阻力矩系数表 + 推进 + 协调转弯控制，与 `awr.sim.fixedwing` 快速档对拍。
4. numba 核（机群大时）；三个域过门后把 `physics = "realistic"` 切到统一引擎。
5. `l3`（SITL）不在本步。

## 6. 回归

- 新增 `tests/d3/phys/test_l2.py` 12 项：刚体与四元数助手、`quad_x` 布局 / 分配 / PX4 归一化权威、悬停与样本自洽、起降 / goto / 落地 / 失能坠地 / 确定性（两次 `pose_sha256` 相同）、速度与偏航阶跃、恒风倾角等于 SIH 标定值、撞墙与 IMPACT、非多旋翼拒绝、与 Fast 的接口一致（4 个用例）。
- `tests/sensors/test_servo_ladder.py` 的保真档断言改为统一引擎口径。
- `RefTracker` 的 `inner_pid=False` 缺省分支不变，gz 档行为不变；`Setpoint` 只加了带缺省值的字段。
- 5b：新增 `tests/d3/phys/test_l2_ugv.py` 17 项：机型包校验与装载、控制限幅、低速隐式解 = 运动学单轨、速度阶跃精确停车与确定性、坡面跟随（倾角 = atan 坡度、定速）、撞墙（mesh）与路缘（dsm）、空陆混编（x500 + f1tenth 同一档位，车拒绝 takeoff / land）、环绕、基准本身、与 ST 基准接口一致且容差内（8 用例）。`L2Tier` 重构成域分组（多旋翼代码路径不变，`test_l2.py` 12 项照旧通过）；保真档断言改为含陆行；`VehicleStatus.MOVING` 只有车会报，空中设备路径不变。
