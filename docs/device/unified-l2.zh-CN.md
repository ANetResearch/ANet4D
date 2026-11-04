# 统一 L2 物理引擎（具身设备引擎，第 5 步）

## 1. 为什么要自己的 L2

第 4 步的保真档表里，`l2`（Realistic）只有 gz-sim 锁步一条路，且 gz 只有 `x500_d4d` 一个模型。L2 是本系统的核心档位（空、陆、海设备都要在"有惯量、有执行器、有气动/水动力"的物理里跑），只靠一个外部引擎、一个机型既不够用，也没有原创性。本步起，L2 由我们自己的引擎给出：`awr.phys.l2`，一个刚体核、一个档位类、每个设备族一个域模型；gz 保留为交叉验证（它有模型的机型上两边对拍）。

```
awr.phys.l2
├── rigid.py        六自由度刚体核（ENU / FLU，四元数，半隐式欧拉；空陆海共用）
├── multirotor.py   多旋翼域模型：转子级执行、PX4 级联、控制分配、复合气动
└── tier.py         L2Tier：PhysicsTier（与 Fast、gz 同一指令、同一参考跟踪器、同一接触规则、同一 PoseHistory）
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

gz 对拍（x500）要在 dmax 的 d3-gz 容器里跑 `run --tier realistic`，然后 `report --tiers l2,realistic`；本步未跑。

## 4. 怎么用

- Env：`spec.physics = "l2"`（与 `kinematic / fast / realistic` 并列），机型由 `fleet[].vehicle_ref` 给出，任一机型不是多旋翼即 630。
- 直接用：`L2Tier(uavs, spawn, world, grounded=…, vehicles=…)`，接口同 `FastTier`；`set_wind(n×3)` 每 tick 写风。
- 对拍：`python -m awr.phys.tier_compare run --tier l2 --out DIR`，`report --out DIR --tiers fast,l2`。
- 保真档表：`python -m awr.sim.device ladder` 的 `l2` 列现在按"统一引擎有没有该设备族的域模型"判定，x500 另标 `+ gz`。

## 5. 接下来（同一刚体、同一档位）

按用户定的顺序：

1. **地面车 `ugv`**：机型包（`kind: ugv`，schema 要加车辆字段）+ 域模型（自行车运动学 → 单轨动力学 + 线性轮胎 + 滚阻 / 坡度），参考 JPL Open Source Rover、F1TENTH、OpenPodcar；参考跟踪器要换成路径跟踪（Pure Pursuit / Stanley）。
2. **无人艇 `usv`**：Fossen 三自由度（附加质量、线性 + 二次阻尼、双推进器分配）+ 风、流；参考 BlueBoat、NAC-USV。
3. **固定翼 `fixed_wing` / `vtol`**：升阻力矩系数表 + 推进 + 协调转弯控制，与 `awr.sim.fixedwing` 快速档对拍。
4. gz 交叉验证跑一次（x500），numba 核（机群大时）。
5. `l3`（SITL）不在本步。

## 6. 回归

- 新增 `tests/d3/phys/test_l2.py` 12 项：刚体与四元数助手、`quad_x` 布局 / 分配 / PX4 归一化权威、悬停与样本自洽、起降 / goto / 落地 / 失能坠地 / 确定性（两次 `pose_sha256` 相同）、速度与偏航阶跃、恒风倾角等于 SIH 标定值、撞墙与 IMPACT、非多旋翼拒绝、与 Fast 的接口一致（4 个用例）。
- `tests/sensors/test_servo_ladder.py` 的保真档断言改为统一引擎口径。
- `RefTracker` 的 `inner_pid=False` 缺省分支不变，gz 档行为不变；`Setpoint` 只加了带缺省值的字段。
