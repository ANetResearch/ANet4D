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

## 2c. 无人艇域模型（5c）

gz 的水面支持是一组按模型逐个调参的插件（浮力、升阻力船体、推进器，VRX 的做法），不是一个艇模型，所以这一域集成的是操纵性文献的标准模型：Fossen 三自由度（纵荡、横荡、偏航）水平面模型，形式与 MSS `python_vehicle_simulator` 的 Otter 一致，再补上沙盒要的部分：

| 环节 | Fossen 三自由度（MSS otter.py） | 统一引擎 usv 域 |
|---|---|---|
| 质量 | M = M_RB(m, x_g, I_z) + diag(X_u̇, Y_v̇, N_ṙ)，机体系 ODE | **同一刚体核**：机体系算 ν̇，a_w = R(ν̇ + r k × ν) 进 `RigidState.integrate`；z、横滚、俯仰约束到水面（三自由度） |
| 科氏 / 向心 | C_RB(ν)ν + C_A(ν_r)ν_r，偏航 Munk 力矩置零 | 同 |
| 阻尼 | 线性 diag(X_u, Y_v, N_r) + 偏航二次 10·N_r·\|r\|r + 横流阻力条带积分（Hoerner C_d） | 同一式，另加通用二次项 diag(X_\|u\|u, Y_\|v\|v, N_\|r\|r)（WAM-V 的 VRX 参数就是这种形式，横流项为零） |
| 推进 | 双固定推进器 T = k_pos·n\|n\|（倒车 k_neg），轴速一阶滞后 T_n，限转速；X = T_l + T_r，N = y_t(T_r − T_l) | 同 |
| 环境 | 流（ν_r）、风（无人艇例子里未接） | 流 `set_current`（进每一个水动力项，推进器关掉就随流漂）；风 ½ρ_air·C_dA·\|w_rel\|·w_rel |
| 水面 / 岸 | 无 | z = water_z + 质心出水高；DSM 高于 water_z − 吃水即搁浅（COLLISION_WORLD，艇停在触底处）；网格最近点查询从质心上方 r 处发出；无世界时为深水 |
| 停船 | 速度指数衰减，永不为零 | 静水中指令停船且 \|u\|、\|v\| < 0.05 m/s、\|r\| < 0.02 rad/s 即静止，速度精确为零（有流时不静止：随流漂或顶流保位） |

参考与控制：指令集相同（velocity、goto、follow_path、orbit、hover、kill；takeoff / land 对艇返回 630；kill = 推进器关）。`WaterTracker` 把共用 M24 跟踪器的**对地**速度设定减去流速，得到艇要**对水**做出的速度与航向——于是"对地速度为零"在流里就是顶流保位；hover 是保位（偏离 0.3 m 后以 0.3·偏距 的对地速度回到保位点，保位点取"能停下的位置"= 指令位置 + 刹车距离，不掉头）；到 goto 目标 / 末航点 / 环绕终点 0.5 m 内停船，冲过了（目标在艉后一个艇长内）也停——艇不能原地掉头。控制器：速度参考按 a_max 斜坡 + 前馈 + PI（阻尼补偿、抗卷绕），航向误差 → 偏航角速度需求（K 1.0，限 r_max）→ P 偏航角速度环（阻尼补偿），分配到两轴并按轴饱和（饱和时偏航优先）；纵荡参考乘 cos(航向误差)（截 0），朝向不对先转再走。

机型包同地面车一族的做法（`vehicles/usv/catalog.json` + `vehicles/usv/<id>/params.yaml`，schema `awr.vehicle.usv.v1` / `awr.device_catalog.usv.v1`；装载器 `awr.phys.l2.ground` 泛化为按族装载）。三艘：

| id | 原型 | 参数等级 |
|---|---|---|
| `otter` | Maritime Robotics Otter 级 2 m 双体艇（55 kg + 25 kg 载荷） | 水动力与推进器参数逐项取 MSS otter.py（A 级：m、L、B、B_pont、y_pont、X_u̇ = −0.1 m、Y_v̇ = −1.5 m、k_pos / k_neg、T_n、U_max 6 kn；B 级为按其公式算出的 X_u、Y_v、N_r、N_ṙ、n_max / n_min、吃水、x_g、Hoerner C_d）；对拍基准也是它 |
| `blueboat` | Blue Robotics BlueBoat 级 1.2 m 小艇 | 尺寸、干重、最高速 3 m/s、电池 532 Wh / 1 m/s 18 h 为规格页 A 级；水动力与推进器系数 D 级（按 Otter 比例与 3 m/s 推阻平衡定） |
| `wamv` | WAM-V 16 级 4.9 m 双体艇（RobotX / VRX） | 质量 180 kg、I_z 446、推进器位置、线性 + 二次阻尼系数取 VRX 模型（A 级）；推力上限 250 / −100 N 为 VRX 插件缺省（B 级），其余 D 级 |

## 2d. 固定翼域模型（5d）

gz 没有一个固定翼模型：PX4 的 advanced_plane 等是按机型逐个调参的 LiftDrag 插件组合，所以这一域集成的是小型无人机教科书的标准模型：Beard & McLain《Small Unmanned Aircraft》第 4 章的系数表六自由度（附录 E 的 Aerosonde 形式），再补沙盒要的部分：

| 环节 | Beard & McLain 系数表六自由度 | 统一引擎 fixed_wing 域 |
|---|---|---|
| 坐标 | NED / FRD，欧拉角 | **同一刚体核**（ENU / FLU 四元数，半隐式欧拉，对角惯量）；B&M 的 FRD 方程按 y → −y、z → −z 映到 FLU，气动力矩同 |
| 升力 | C_L0 + C_Lα·α 与平板 2·sgn α·sin²α·cos α 以 sigmoid 混合（锐度 M，失速角 α0） | 同一式；α0 = (C_Lmax − C_L0)/C_Lα + 3/M（机型包只给 cl_max，由它反推），M = 100 |
| 阻力 | C_D0 + C_L²/(π e AR) | 同 |
| 俯仰 / 侧力 / 滚转 / 偏航 | C_m0、C_mα、C_mq、C_mδe；C_Y、C_l、C_n 的 β、p、r、δa、δr 导数 | 同一表，机型包 `fw_dynamics` 块给出（每个系数 value / conf / src） |
| 推进 | 螺旋桨模型（S_prop、C_prop、k_motor） | 功率限制的螺旋桨律 T = δt·min(T_static, η·P_max / V_a)，取机型包已有的 `propulsion`（静推力、最大功率、效率） |
| 执行器 | 舵面直接给定 | 舵面与电机一阶滞后（τ_servo、τ_motor），舵面限幅（δe、δa、δr 各自上限） |
| 发射 / 回收 | 无（空中起算） | **弹射**：倾角 12° 的导轨以恒加速度在 `launch.t_s` 内把机体带到 v_min + 1.5 m/s，姿态沿导轨、无角速度，脱轨即 LIFTOFF；**伞降**：land 指令开伞，阻力 k = m g / v_desc² 的平方阻力 + 角速度阻尼，机体以 τ 1 s 改平，以额定下降率触地 → TOUCHDOWN、GROUND |
| 地面 | 无 | 飞行中触地即 IMPACT（固定翼没有起落架可落地，回收只有伞降）；kill = 无动力、舵面回中，自由滑翔 / 螺旋至触地；撞网格为 COLLISION_WORLD |

参考与控制：指令集 velocity、goto、follow_path、orbit、hover、takeoff、land、kill。`WingTracker` 把 M24 的指令变成固定翼的参考（空速、航向 χ、航向前馈、爬升率）：直线与圆的向量场用 `awr.sim.fixedwing.kinematics` 快速档同一套 `line_field` / `circle_field`（罗盘航向换成 ENU），goto 到点后、follow_path 末航点后、hover 都是顺时针盘旋，半径取 max(1.2 V²/(g tan φ_max(V))，机型缺省盘旋半径，指令半径)，航点到达圈 max(30 m, 0.5 盘旋半径)，orbit 按圈数计满后继续盘旋。自动驾驶仪是 B&M 第 6 章的逐环结构，**与基准共用**：横向——航向误差 → 航向角速度需求 → 协调转弯倾角 φ_c = −atan(V_a·ψ̇_c / g)（ENU 逆时针为正），倾角上限 min(φ_max, acos((1.15 V_s / V_a)²))（载荷因子不逼近失速），滚转角速率限幅，副翼 PD，方向舵按侧滑 β 配平；纵向——爬升率需求先过"速度优先"的余量 clip((V_a − 1.1 V_s)/(v_min − 1.1 V_s), 0, 1)（慢了先低头），γ_c = asin(ḣ_c / V_a)，转弯载荷 n = 1/cos φ_c 下的需求迎角作前馈，俯仰角需求 = γ_c + α_req + 高度环 PI，并限在当前迎角 + 0.85 α_stall 以内（迎角保护），升降舵 PD + 积分，并扣去协调转弯的俯仰角速度 g/V_a·tan φ·sin φ·cos θ；空速——设定 clip(max(V_c, 1.15 V_s·√n), v_min, v_max) 按 a_max 斜坡，油门 = 配平推力前馈 + PI + (1 − 余量)（慢时满油门加低头）。滚转 / 俯仰内环增益按 B&M 的饱和设计在巡航点算出（最大误差 15° / 10°，ζ = 0.707），机型不同自动不同。

机型包：固定翼没有另开一族，仍是 `vehicles/` 的多旋翼目录（schema `awr.vehicle_profile` 1.2）：`fw_dynamics` 块（chord_m、lift、drag、pitch、side、roll、yaw、controls、propulsion）与 `inertia_kgm2`，多旋翼禁止带 `fw_dynamics`，`fw_dynamics` 要求同时有 `inertia_kgm2` 与 `airframe`。`awr_f1`、`awr_v1` 升到 1.1.0（旧版归档在 `history/`）：系数为 B&M 附录 E 的 Aerosonde 形状按翼展 / 翼面积缩放，D 级；`awr_f1` 惯量 0.29 / 0.40 / 0.63 kg·m²，静推力 26 N。VTOL（`awr_v1`）带了系数但还没有域（过渡段的旋翼 + 固定翼混合没有做），`L2Tier` 仍返回 630。

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

## 3c. 海面域：与 Fossen 三自由度基准对拍（5c）

`python -m awr.phys.tier_compare run --set water --tier l2|fossen`，`report --set water --tiers l2,fossen`。基准 `awr.phys.l2.fossen_oracle` 把 Fossen 方程写成矩阵形式（M、C_RB、C_A、D 显式构造，横流条带积分，轴速滞后进 ODE），状态 (x, y, ψ, u, v, r, n_l, n_r)，机体系 RK4；**控制器与引擎共用**，档间差只来自动力学写法与积分方式。8 个可对拍用例全部通过，搁浅用例只有引擎（基准无世界）：

| 用例 | 艇 | 接口 | L2 ↔ Fossen 档间差 | 容差 | 结果 |
|---|---|---|---|---|---|
| w_speed_step 0 → 2 m/s → 停 | otter | 一致（AFLOAT → MOVING → AFLOAT） | 上升时间 0.004 s（1.52 s）、超调 0.6 %（2.9 / 3.5 %）、位置 RMS 0.024 m | 0.5 / 10 / 1.0 | 通过 |
| w_speed_limit 10 m/s 指令按 6 kn 限幅 | otter | 一致 | 稳态 0.034 m/s（3.05 / 3.08，上限 3.086） | 0.3 | 通过 |
| w_goto_stop 到 (30, 15) 停船 | otter | 一致 | 到达 0.016 s（17.4 s）、停船误差 0.23 m、位置 RMS 0.02 m | 2.0 / 1.5 / 2.0 | 通过 |
| w_waypoints 40 m 方形，2 m/s | otter | 一致 | 位置 RMS 0.015 m、到达 0.03 s/100 m | 3.0 / 3.0 | 通过 |
| w_orbit R 10 m，1.5 m/s 一圈 | otter | 一致 | 位置 RMS 0.004 m、半径 RMS 差 0.001 m（都是 0.46）、绕圈时刻 0.00 | 3.0 / 1.0 / 3.0 | 通过 |
| w_current_hold 0.5 m/s 流中保位 | otter | 一致（MOVING） | 两档各自：最大漂移 5.7 m（出发时艇头顺流，掉头 180° 期间被流带走）、稳态偏差 0.20 m、顶流保位对地速度 0；位置 RMS 0.008 m | 8.0 / 1.0 / 2.0 | 通过 |
| w_speed_step_wamv 0 → 1.2 m/s → 停 | wamv | 一致 | 上升时间 0.004 s（1.89 s）、超调 0.5 %、位置 RMS 0.014 m | 1.0 / 10 / 1.0 | 通过 |
| w_waypoints_blueboat 40 m 方形，2 m/s | blueboat | 一致 | 位置 RMS 0.14 m、到达 0.14 s/100 m | 3.0 / 3.0 | 通过 |
| w_grounding 2 m/s 驶向 5 % 坡岸 | otter | 引擎独有 | COLLISION_WORLD（dsm）于 x = −3.90（0.05·x > −吃水 0.195 处），状态 AFLOAT → MOVING → CRASHED | — | 通过 |

档间差比地面域还小（位置 RMS 厘米级）：两边都在机体系算 ν̇，差异只有积分（半隐式欧拉 vs RK4）与世界系 / 机体系的换算。性能：单艇约 25 ms / 100 ms tick，基准 35 ms。

## 3d. 固定翼域：与 Beard–McLain 六自由度基准和固定翼快速档对拍（5d）

`python -m awr.phys.tier_compare run --set wing --tier l2|bm|fw`，`report --set wing --tiers l2,bm` 与 `l2,fw`。基准 `awr.phys.l2.bm_oracle`（`bm`）把 B&M 的 3.14–3.17 与 4.18–4.20 式按书写法实现：状态 (p_n, p_e, p_d, u, v, w, φ, θ, ψ, p, q, r, δe, δa, δr, δt)，NED / FRD，RK4，**控制器与引擎共用**，档间差只来自坐标写法、欧拉角 vs 四元数与积分方式。第二个对照 `awr.phys.fwkin`（`fw`）是现有的固定翼快速档（`awr.sim.fixedwing.kinematics` 的运动学飞行）套上同一跟踪器，衡量 L2 相对快速档多出的动力学差。7 个可对拍用例两对全部通过，发射 / 伞降 / 撞墙 / kill 只有引擎：

| 用例 | 接口 | L2 ↔ B&M 档间差 | 容差 | L2 ↔ fw 档间差 | 容差 |
|---|---|---|---|---|---|
| f_speed_step 18 → 25 m/s | 一致（AIRBORNE） | 上升时间 0.12 s（4.91 / 5.02）、超调 0.3 %（7.3 / 7.0）、位置 RMS 0.52 m | 1.0 / 10 / 5.0 | 上升时间 0.20 s、超调 7.3 %（fw 无超调）、位置 RMS 5.2 m | 4.0 / 10 / 40 |
| f_speed_limit 40 m/s 指令按 v_max 28 限幅 | 一致（峰值 28.54 / 28.52，上限 + 1.0） | 稳态 0.002 m/s（27.99 / 27.99） | 1.0 | 稳态 0.012 m/s（fw 正好 28） | 2.0 |
| f_climb 3 m/s 爬升 20 s 后改平 | 一致 | 爬升率 0.061 m/s（3.05 / 3.11）、高度增量 0.18 m（59.9 / 59.8）、位置 RMS 0.64 m | 0.3 / 5.0 / 5.0 | 0.045 m/s、0.05 m、1.10 m | 0.5 / 15 / 40 |
| f_goto 到 (900, 400, 130) 后盘旋 | 一致 | 到达 0.02 s（52.5 s）、盘旋半径 0.016 m（80.75 / 80.77）、位置 RMS 0.77 m | 3.0 / 8.0 / 10 | 0.028 s、0.33 m（80.43）、3.3 m | 6.0 / 20 / 60 |
| f_waypoints 三航点 600 m 腿，20 m/s | 一致 | 位置 RMS 0.16 m、到达 0.001 s/100 m（27.9 / 58.4 / 89.2 s） | 10 / 1.0 | 3.7 m、0.005 s/100 m | 60 / 2.0 |
| f_orbit R 100 m，18 m/s 一圈 | 一致 | 位置 RMS 0.28 m、半径 RMS 差 0.026 m（2.76 / 2.78）、绕圈时刻 0.003 s/100 m（35.7 s） | 10 / 5.0 / 1.0 | 4.6 m、0.84 m（fw 1.92）、0.044 | 60 / 15 / 3.0 |
| f_wind_track 8 m/s 侧风中沿线 1200 m | 一致 | 位置 RMS 0.15 m、偏航距 RMS 两边都是 3.5 m（起始切入时最大 45 m）、对地速度 16.37 | 5.0 / 10 | 4.3 m、偏航距 4.7 m | 10 / 60 |
| f_launch 弹射到 60 m 后盘旋 | 引擎独有 | GROUNDED → LIFTOFF（1.5 s，15.0 m/s = v_min + 1.5）→ 4 m/s 爬升 → 60 m 盘旋，AIRBORNE | — | — | — |
| f_chute 80 m 巡航中伞降 | 引擎独有 | AIRBORNE → LANDING（5 m/s 额定下降、机体改平）→ TOUCHDOWN → GROUNDED | — | — | — |
| f_wall 18 m/s 平飞撞墙 | 引擎独有 | COLLISION_WORLD（mesh），CRASHED | — | — | — |
| f_kill 100 m 巡航中 kill | 引擎独有 | 无动力无操纵，螺旋下降，IMPACT（约 20 s，27 m/s），CRASHED | — | — | — |

与 B&M 基准的差是厘米到分米级（同一方程、同一控制器，只差积分与姿态表示）；与快速档的差是米级（快速档没有舵面滞后、迎角、载荷因子与速度环超调），这就是 L2 相对 fw 多出的那部分动力学。性能：单机 250 Hz 约 35 ms / 100 ms tick，基准约 65 ms，快速档约 3 ms。

第一版的四处偏差及修法：（1）协调转弯倾角符号——B&M 的 φ_c = atan(V_g χ̇ / g) 是 NED 顺时针为正，ENU 里要取负，否则越偏越转；（2）失速——迎角反馈 + 过宽的失速混合让稍大的爬升指令直接失速，改为需求迎角前馈 + 迎角上限 + 把混合锐度从 50 提到 100 并把 α0 挪到 cl_max 之外；（3）转弯中掉高——只有 P 的俯仰环顶不住倾斜配平，加升降舵积分并扣除协调转弯自带的俯仰角速度；（4）弹射脱轨后趴在地上——平导轨脱轨时机体仍在地面高度，被伞降触地那条"改平 + 摩擦"分支拖停，改为 12° 倾角的运动学导轨，脱轨即带正爬升率离地。

## 4. 怎么用

- Env：`spec.physics = "realistic"`（正式档就是统一引擎，5e / ADR-226，见 §7；`"l2"` 为旧名仍可用；gz 锁步改为 `"gz"`，只作交叉验证），机型由 `fleet[].vehicle_ref` 给出：多旋翼目录里的机型走多旋翼域，`vehicles/ground` 里的 id（`f1tenth`、`osr`、`podcar`）走地面域，`vehicles/usv` 里的 id（`otter`、`blueboat`、`wamv`）走海面域，多旋翼目录里带 `fw_dynamics` 的固定翼（`awr_f1`）走固定翼域，空陆海可以混编在同一机队；VTOL（`awr_v1`）仍是 630。
- 直接用：`L2Tier(uavs, spawn, world, grounded=…, vehicles=…, ground=…, water_z=…)`，接口同 `FastTier`；`set_wind(n×3)` 每 tick 写风（多旋翼与固定翼感到），`set_current(n×2)` 写流（只有艇感到）；车永远落在地面上，艇永远浮在 `water_z` 上（`grounded` 只对空中设备有意义；固定翼 `grounded` 即在发射架上，只有 takeoff 能让它动）。
- 状态：车与艇用 `VehicleStatus.MOVING`（行驶 / 航行中），静止分别为 GROUNDED / `AFLOAT`，撞上或搁浅为 CRASHED；Env 的 `status.mode` 对应 `driving` / `afloat`。固定翼用空中设备的词：GROUNDED（发射架）→ AIRBORNE → LANDING（伞降中）→ GROUNDED，飞行中触地 / 撞网格为 CRASHED。
- 对拍：`python -m awr.phys.tier_compare run --tier l2 --out DIR`，`report --out DIR --tiers fast,l2`；地面域 `run --set ground --tier l2|st`，`report --set ground --tiers l2,st`；海面域 `run --set water --tier l2|fossen`，`report --set water --tiers l2,fossen`；固定翼域 `run --set wing --tier l2|bm|fw`，`report --set wing --tiers l2,bm` / `l2,fw`。
- 保真档表：`python -m awr.sim.device ladder` 的 `l2` 列按"统一引擎有没有该设备族的域模型"判定，x500 另标 `+ gz`；陆行来自 `vehicles/ground`，海来自 `vehicles/usv`；固定翼按多旋翼目录的 `category: fixed_wing` 判，VTOL 行仍为"尚无 vtol 模型"。

## 5. 接下来（同一刚体、同一档位）

按用户定的顺序：

1. ~~地面车 `ugv`~~：5b 已做（§2b、§3b）。还没做的：差速 / 滑移转向的专用运动学（现在 `drive.type` 只是标注，六轮漫游车按等效自行车处理）、Pacejka 非线性轮胎、悬挂。
2. ~~无人艇 `usv`~~：5c 已做（§2c、§3c）。还没做的：横摇 / 纵摇 / 升沉（波浪）、单舵单桨与全回转推进的分配（`propulsion.type` 现在只接受 `twin_fixed` 的动力学）、风的 A 级系数。
3. ~~固定翼 `fixed_wing`~~：5d 已做（§2d、§3d）。还没做的：VTOL `vtol` 域（旋翼 + 固定翼过渡），非对角惯量 J_xz，螺旋桨的扭矩 / 滑流效应，固定翼的滑跑起降（现在只有弹射 + 伞降），系数的 A / B 级来源（现在是 Aerosonde 形状缩放的 D 级）。
4. numba 核（机群大时）。~~三个域过门后把 `physics = "realistic"` 切到统一引擎~~：5e 已做（§7）。
5. `l3`（SITL）不在本步。

## 6. 回归

- 新增 `tests/d3/phys/test_l2.py` 12 项：刚体与四元数助手、`quad_x` 布局 / 分配 / PX4 归一化权威、悬停与样本自洽、起降 / goto / 落地 / 失能坠地 / 确定性（两次 `pose_sha256` 相同）、速度与偏航阶跃、恒风倾角等于 SIH 标定值、撞墙与 IMPACT、非多旋翼拒绝、与 Fast 的接口一致（4 个用例）。
- `tests/sensors/test_servo_ladder.py` 的保真档断言改为统一引擎口径。
- `RefTracker` 的 `inner_pid=False` 缺省分支不变，gz 档行为不变；`Setpoint` 只加了带缺省值的字段。
- 5d：新增 `tests/d3/phys/test_l2_fixedwing.py` 12 项：机型包参数与配平（巡航 / 最低速配平在包线内，多旋翼机型被拒 630）、速度阶跃与爬升保持及确定性（两次 `pose_sha256` 相同）、弹射起飞（地面指令不动、GROUND → LAUNCH → FLY、LIFTOFF 时刻与速度、导轨姿态、爬到 60 m 后盘旋）、伞降触地与 kill 触地、8 m/s 侧风沿线（偏航距、偏流角、对地速度）、空 / 固定翼 / 车混编（空中 takeoff 为空操作、车拒绝 630）、两个基准的内核名与只收空中指令、与 B&M 基准（爬升、环绕）和快速档（goto、侧风）接口一致且容差内（全部 7 × 2 用例由 `tier_compare --set wing` 跑）、引擎独有的撞墙。保真档断言改为含固定翼（`awr_f1` l2 可用，`awr_v1` 仍无）；`test_l2.py` 的"无域模型被拒"改用 VTOL。
- 5c：新增 `tests/d3/phys/test_l2_usv.py` 18 项：机型包校验与装载、分配与轴限幅、基准 ODE 静止 / 定速自洽、速度阶跃精确停船与确定性（3 自由度：无横滚 / 俯仰 / 升沉）、流中保位（顶流、对地速度 0）与 kill 随流漂、搁浅与水位、空陆海混编（x500 + f1tenth + blueboat 同一档位）、环绕一圈后停、航行中 hover 不掉头、基准本身、与 Fossen 基准接口一致且容差内（8 用例）。`VehicleStatus.AFLOAT` 只有艇会报。
- 5b：新增 `tests/d3/phys/test_l2_ugv.py` 17 项：机型包校验与装载、控制限幅、低速隐式解 = 运动学单轨、速度阶跃精确停车与确定性、坡面跟随（倾角 = atan 坡度、定速）、撞墙（mesh）与路缘（dsm）、空陆混编（x500 + f1tenth 同一档位，车拒绝 takeoff / land）、环绕、基准本身、与 ST 基准接口一致且容差内（8 用例）。`L2Tier` 重构成域分组（多旋翼代码路径不变，`test_l2.py` 12 项照旧通过）；保真档断言改为含陆行；`VehicleStatus.MOVING` 只有车会报，空中设备路径不变。

## 7. 5e：`physics = "realistic"` 切到统一引擎（ADR-226）

四个域都过门后，正式档换成统一引擎，gz 退为交叉验证：

- Env：`spec.physics = "realistic"` 创建 `L2Tier`（正式时钟规则不变：d1_250、M24 输出集合校验），运行清单 `tier = l2`；`"l2"` 是旧名，等价但不套正式时钟规则；`"gz"` 是 gz-sim 锁步（原 Realistic 档），只作交叉验证，`ranked_physics` 规则只允许 `realistic` 排名，所以 gz 不出正式成绩。
- 调度器只为 `gz`、`sitl` 预留 gz 容器资源；`realistic` 运行在 Env 进程内的 CPU 上。
- `run_spec.v1` 的 `physics` 枚举增加 `gz`（契约版本不变）。
- `tier_compare`：gz 档的标签改为 `gz`（`run --tier gz`，制品 `<case>.gz.npz`，`report --tiers fast,gz` 写 `tier_compare.json`）；报告 schema `drone4d.phys.tier_compare.v1` 不变，`/lab/compare` 页按报告的 `tiers` 取第二档的名字，封板前写出的制品（标签 `realistic`）照常可读。
- 没做、登记为后续事项：排行榜在本 ADR 前后的正式成绩分别在 gz 与统一引擎上取得，下一次正式评测前须登记新的榜单版本与平台摘要（F-150）；正式配置的确定性复测（`determinism_official`）在统一引擎上于 dmax 独占卡重做（F-151）；`physics_opts.vehicle_collision`（机间碰撞）只有 gz 实现，统一引擎暂无机间碰撞。
