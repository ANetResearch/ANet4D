# 设备描述与保真档（具身设备引擎，第 4 步）

## 1. 设备描述：沿用机型包，不另起格式

一台设备 = 机型包 `vehicles/<dir>/params.yaml`（`awr.vehicle.v1`，schema 1.1：质量、惯量、电机、桨、气动、电池、限速配置、风等级、声学/视觉/雷达/红外特征、链路、传感器挂载）+ 它挂的传感器型号 `vehicles/sensor_models/*.yaml`（相机、热像、GNSS、IMU、激光雷达、毫米波、声阵列；云台限位与速率在型号文件的 `gimbal` 块）+ `vehicles/catalog.json` 的目录行。这套描述已经经过 AWR-20 的自洽检查（VH/VT/FW/MNT 规则、derived 值比对、E 级值拒绝），本步**不另起一套设备描述格式**，只补两件事：

1. 机型 `fidelity` 字段（`l0 / l1 / sih / fw_kin`）对应到统一的保真档名，并由代码说明每一档现在到底有没有实现（`awr.sim.device.ladder`）。
2. 传感器 `gimbal` 块新增 `servo` 子块（伺服档），见 §3。

## 2. 保真档（空、陆、海共用的词表）

| 档 | 含义 | 代码 | 现状 |
|---|---|---|---|
| `l0` | 回放 / 运动学：位姿来自记录或运动提供者，无动力学 | `Fidelity.L0`，stage `kinematic` | 可用 |
| `l1` | PX4-lite 级联，125 Hz：位置环 → 姿态环 → 电机 → 气动 → 积分 | `Fidelity.L1`，stage `l1`（numba 融合核与 numpy oracle 对拍） | 可用 |
| `l2` | Realistic：统一 L2 物理引擎（转子 / 执行器级驱动、速率环、控制分配、刚体惯量、气动 / 水动力），每个设备族一个域模型；gz-sim 锁步为交叉验证 | `awr.phys.l2`（[unified-l2.zh-CN.md](unified-l2.zh-CN.md)）；`awr.phys.gz.GzSimBackend`（d3-gz 容器，只有 `x500_d4d`） | **可用：多旋翼**（第 5 步 5a）；固定翼 / VTOL、陆、海的域模型随第 5 步后续补 |
| `l3` | SITL：真实 PX4 / ArduPilot 飞控栈在环 | 无 | **无**，第 5 步 |
| `fw` | 固定翼 / VTOL 快速档：运动学飞行 + 能量模型 | `Fidelity.FW`，stage `fw_kin`（`awr.sim.fixedwing`） | 可用 |

机型声明（`fidelity.supported`）与代码实现取交集才算"可用"（`l2` 由 run_spec 的档位选择而不是机型字段，按统一引擎有没有该设备族的域模型判，x500 另标 + gz）；`awr.sim.device.ladder()` 对目录里每台设备、每一档给出 可用 / 已声明未实现 / 代码有、机型未声明 / 无。陆、海设备现在没有机型包，表里以"计划（第 5 步）"行列出（JPL 开源漫游车级、F1TENTH 级、OpenPodcar 级；BlueBoat 级、NAC-USV 级），第 5 步补机型包时同一张表自动变成实测状态。

`python -m awr.sim.device ladder [--json] [--responses]` 打印这张表（`awr.device_ladder.v1`）；在线页 `/anet4d/device/` 就是它。

## 3. 云台：一个引擎，两档伺服

此前有三处云台实现：sim-core 的 `GimbalBank`（5 种模式、4 层指向仲裁、标量 / numpy / numba 三条路径）、agent-env 的 `camera.Gimbal`（钳制 + 限速）、CHIMERA 的瞬时云台。本步把"轴怎么动"收成一个模型 `awr.sim.sensors.servo`，前两者都调用它（CHIMERA 的瞬时云台随第 6 步接口合并后退役）：

| 伺服档 | 轴的运动 | 用途 |
|---|---|---|
| `l1` | 限速逼近：每步向钳制后的目标走至多 `rate_max·dt`，到位即停。与以前逐位相同（缺省） | 指向逻辑、大机群 |
| `l2` | 二阶伺服：轴有角速度状态；速度指令 `err/τ`，受 `rate_max` 与离散制动曲线 `√(h²+2·a·|err|)−h`（`h = a·dt/2`）限制；角速度每步变化至多 `acc_max·dt`；角度按速度积分，跨过目标时落在目标并归零速度。有起步、匀速、减速、指数收敛，无过冲 | 相机在转动中看到什么；评测云台延迟 |

配置在传感器型号或挂载文件的 `gimbal` 块：

```yaml
gimbal: {pitch_min_deg: -90, pitch_max_deg: 30, yaw_min_deg: -160, yaw_max_deg: 160, rate_max_deg_s: 90,
         servo: {tier: l2, acc_max_deg_s2: 400, tau_s: 0.08}}
```

缺省 `tier: l1`。`l2` 的角速度状态放在 `GimbalBank.omega`（不进 checkpoint，重启后从静止续算）与 `camera.Gimbal.omega_*`。机群里只要有一列 `l2`，大机群路径走 numpy（融合核只实现 `l1`）；标量与向量路径逐位相同（测试）。

90°/s 云台对 60° 阶跃：`l1` 0.667 s 到位；`l2`（400°/s²、τ = 0.08 s）约 0.9 s 到位，前 0.1 s 只转 2°（起步），曲线见在线页。

另修一处单位 bug：`sensorsvc.camera.effects.gimbal_reading` 把度为单位的编码器噪声直接加在弧度角上（CHIMERA 补丁 `gimbal-noise-units`），现在先换算成弧度。

## 4. 回归

- `tests/sensors`、`tests/sim`、`tests/d3/sensorsvc`：改动前后同一组失败（见引擎进展页第 4 步），云台与多旋翼 L0/L1/FW 的既有测试全部不变。
- 新增 `tests/sensors/test_servo_ladder.py`（9 项）：`l1` 与旧表达式逐位相同；`l2` 守限速与加速度上限、单调无过冲、正负对称；标量与向量路径逐位相同；`l1` 是 `l2` 的极限；agent-env 云台两档；规格解析与拒绝；台架里 `l2` 列起步慢于 `l1`、不过冲、向量路径 == 标量路径；编码器噪声单位；保真表。

## 5. 没做的

- `l2`：第 5 步 5a 已由统一引擎给出多旋翼（见 unified-l2.zh-CN.md）；固定翼、陆、海的域模型与 `l3` SITL 在后面。
- 陆、海机型包与它们的 L1 模型：第 5 步。
- 云台 `l2` 的融合核版本与 checkpoint 中的角速度状态。
