// 快速仿真（Fast）与第二档（报告 `tiers[1]`：ADR-226 后为 gz 锁步 `gz`，封板前的制品为 `realistic`）档间对照的数据层（D3-AC-35"档间差异报告 `tier_compare.json`（位置 RMS、到达时刻差、
// 指标差）带容差"；AWR-05 §5.3.6 档间差异报告行；M24 §6.10.6、§6.10.7、FR-081、AC-016、AC-028；AWR-14 §21.10；UX-AC-097）。
// 制品由 `python -m awr.phys.tier_compare run|m25|report --out DIR` 写在同一目录，内部经 BN-54 `GET /runs/{rid}/artifacts/tier_compare/{file}` 读取：
// - `tier_compare.json`（`drone4d.phys.tier_compare.v1`，`build_report`）：机型、时钟档、风模式、力律、碰撞源、容差状态；`cases[]` 的
//   `interface`（指令回执、VehicleStatus 序列、接触事件、PoseHistory 网格、指令生效步、限幅）、`metrics.{fast,realistic}`、`pos_rms_m`、`diff`、
//   `tol`、`within`、`tol_pass`、`pass`；`sampling_consistency`（注入模式两档采样对拍与 M25 LiDAR、IMU）；`m25_bytewise`（同一 PoseHistory 两次送入 M25）；
// - `<case>.<tier>.npz`（`CaseRun.save`）：`t_ns` i64 (n,) 与每机 `s_<uav>` f64 (n, 16)（`px py pz qx qy qz qw vx vy vz wx wy wz fx fy fz`，ENU）。
// 页面不重新判定动力学：差值、容差与判定取自报告；轨迹只用于显示与核对——两档时刻须逐位相同（规格一致），由轨迹按报告同式重算的
// 位置 RMS（`sqrt(mean(|p_fast − p_real|²))`）须与 `pos_rms_m` 一致（≤ 1e-9 m），报告的 `within` 须与 `diff ≤ tol` 一致。
import { readNpz, type NpyArray } from './npz'

export const TIER_COMPARE_SCHEMA = 'drone4d.phys.tier_compare.v1'
/** 缺省两档名（报告没有 `tiers` 时）：封板前制品的第二档叫 `realistic`，ADR-226 之后写出的叫 `gz` */
export const TIERS = ['fast', 'realistic'] as const
export type Tier = string
/** 报告的两档名：`tiers[0]` 为快速仿真一侧，`tiers[1]` 为对照一侧（页面的 `real*` 字段都指它） */
export type TierPair = readonly [Tier, Tier]
export const REPORT_FILE = 'tier_compare.json'
/** 运行 id（采集运行为命名，基准运行为 run-<10 位 base32>；与 /lab/runs/:rid 相同） */
export const RUN_ID_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/
/** 用例 id（`awr.phys.tier_compare.CASES`），同时用作制品文件名，不含路径分隔符 */
export const CASE_ID_RE = /^[a-z][a-z0-9_]{0,63}$/
const UAV_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/
/** PoseHistory 每样本 16 个 f64（awr.phys.pose_history.STATE_N） */
export const STATE_N = 16
/** 由轨迹重算的位置 RMS 与报告值的核对容差（两边同为 f64，差异只来自求和次序） */
export const POS_RMS_CHECK_M = 1e-9
/** 每条曲线最多绘制的点数（确定性等步长抽稀；核对用全部样本） */
export const PLOT_MAX_POINTS = 1500

/** 报告不合规（`code`：`schema`、`cases`、`case`） */
export class TierCompareDocError extends Error {
  readonly code: string
  constructor(code: string, message?: string) {
    super(message ?? code)
    this.name = 'TierCompareDocError'
    this.code = code
  }
}
/** 轨迹制品不合规（`code`：`t_ns`、`state`、`grid`） */
export class TrackError extends Error {
  readonly code: string
  constructor(code: string, message?: string) {
    super(message ?? code)
    this.name = 'TrackError'
    this.code = code
  }
}

// ---------------------------------------------------------------------------------------------------- 报告
export type MetricValue = number | string | boolean | readonly (number | null)[] | null
export interface SeqCheck { fast: string[]; realistic: string[]; expected: string[]; pass: boolean }
export interface CaseInterface {
  dispatchEqual: boolean | null
  status: Readonly<Record<string, SeqCheck>>
  contacts: Readonly<Record<string, SeqCheck>>
  grid: { samples: number | null; tBitwise: boolean; quatUnit: boolean; pass: boolean } | null
  onset: { fastS: number | null; realisticS: number | null; maxS: number | null; pass: boolean } | null
  limitCap: { capMps: number | null; fastMaxMps: number | null; realisticMaxMps: number | null; pass: boolean } | null
  pass: boolean
}
/** 一项带容差的档间差：`recomputed` 为页面按 `diff ≤ tol` 的判定，`consistent` 为其与报告 `within` 是否一致 */
export interface DiffRow { key: string; diff: number | null; tol: number; within: boolean | null; ratio: number | null; recomputed: boolean; consistent: boolean }
export interface TierCase {
  id: string
  title: string
  iface: CaseInterface
  failures: string[]
  metrics: Readonly<Record<Tier, Readonly<Record<string, MetricValue>>>>
  posRms: Readonly<Record<string, number>>
  rows: DiffRow[]
  tolPass: boolean
  pass: boolean
  /** 报告自身判定一致（每行 within 与 diff ≤ tol 一致，tol_pass 与各行一致） */
  consistent: boolean
  /** 各行 diff / tol 的最大值（容差裕度最小处） */
  maxRatio: number | null
  wallS: Readonly<Record<Tier, number | null>> | null
  uavs: string[]
}
export interface PoseCheck { posMaxM: number | null; attMaxRad: number | null; velBitwise: boolean; omegaBitwise: boolean; fBBitwise: boolean; pass: boolean }
export interface Sampling {
  pose: { tBitwise: boolean; n: number | null; uavs: Readonly<Record<string, PoseCheck>>; tolPosM: number | null; tolAttRad: number | null; pass: boolean } | null
  lidar: { pointWorldMaxM: number | null; originMaxM: number | null; frames: number | null; points: number | null; pointsEqualFrames: number | null; tolM: number | null; pass: boolean } | null
  imu: { accMaxMps2: number | null; gyroMaxRps: number | null; samples: number | null; bitwiseBlocks: number | null; blocks: number | null; tolAccMps2: number | null; tolGyroRps: number | null; pass: boolean } | null
  pass: boolean
}
export interface M25Bytewise { streams: { key: string; sha256: string[]; frames: number | null }[]; world: string | null; renderer: string | null; pass: boolean }
export interface TierCompareDoc {
  schema: string
  /** 两档名：`[0]` 快速仿真一侧，`[1]` 对照一侧（gz，封板前制品为 realistic） */
  tiers: TierPair
  generatedUtc: string | null
  vehicle: string | null
  clockProfile: { name: string | null; mainHz: number | null; physicsHz: number | null } | null
  tickMs: number | null
  poseHistoryHz: number | null
  windMode: string | null
  turbulenceSigma: number | null
  dragLaw: string | null
  collisionSource: string | null
  m09Autonomy: string | null
  realisticTracker: string | null
  kernel: string | null
  gz: { simVersion: string | null; pluginBuild: string | null } | null
  toleranceStatus: string | null
  /** 容差已由 ADR 冻结（`tolerance_status` 以 frozen 开头；ADR-208 确认前为 initial） */
  toleranceFrozen: boolean
  cases: TierCase[]
  summary: { cases: number; interfacePass: number; tolPass: number; pass: number }
  sampling: Sampling | null
  m25: M25Bytewise | null
  pass: boolean
}

type Obj = Record<string, unknown>
const isObj = (x: unknown): x is Obj => typeof x === 'object' && x !== null && !Array.isArray(x)
const num = (x: unknown): number | null => (typeof x === 'number' && Number.isFinite(x) ? x : null)
const str = (x: unknown): string | null => (typeof x === 'string' ? x : null)
const bool = (x: unknown): boolean => x === true
const strs = (x: unknown): string[] => (Array.isArray(x) ? x.filter((s): s is string => typeof s === 'string') : [])
const obj = (x: unknown): Obj => (isObj(x) ? x : {})

function metricValue(x: unknown): MetricValue | undefined {
  if (x === null || typeof x === 'string' || typeof x === 'boolean') return x
  if (typeof x === 'number') return Number.isFinite(x) ? x : null
  if (Array.isArray(x) && x.every((v) => v === null || typeof v === 'number')) return x as (number | null)[]
  return undefined
}
function metricMap(x: unknown): Record<string, MetricValue> {
  const out: Record<string, MetricValue> = {}
  for (const [k, v] of Object.entries(obj(x))) {
    const m = metricValue(v)
    if (m !== undefined) out[k] = m
  }
  return out
}
function seqChecks(x: unknown, [ta, tb]: TierPair): Record<string, SeqCheck> {
  const out: Record<string, SeqCheck> = {}
  for (const [u, v] of Object.entries(obj(x))) {
    if (!isObj(v)) continue
    out[u] = { fast: strs(v[ta]), realistic: strs(v[tb]), expected: strs(v.expected), pass: bool(v.pass) }
  }
  return out
}

function parseInterface(x: unknown, tiers: TierPair): CaseInterface {
  const [ta, tb] = tiers
  const i = obj(x)
  const g = isObj(i.pose_history_grid) ? i.pose_history_grid : null
  const o = isObj(i.command_onset) ? i.command_onset : null
  const l = isObj(i.limit_cap) ? i.limit_cap : null
  return {
    dispatchEqual: typeof i.dispatch_equal === 'boolean' ? i.dispatch_equal : null,
    status: seqChecks(i.status, tiers), contacts: seqChecks(i.contacts, tiers),
    grid: g ? { samples: num(g.samples), tBitwise: bool(g.t_bitwise), quatUnit: bool(g.quat_unit), pass: bool(g.pass) } : null,
    onset: o ? { fastS: num(o[`${ta}_s`]), realisticS: num(o[`${tb}_s`]), maxS: num(o.max_s), pass: bool(o.pass) } : null,
    limitCap: l ? { capMps: num(l.cap_mps), fastMaxMps: num(l[`${ta}_max_mps`]), realisticMaxMps: num(l[`${tb}_max_mps`]), pass: bool(l.pass) } : null,
    pass: bool(i.pass),
  }
}

/** 带容差的行：先按 `diff` 的键序，再补只在 `tol` 中出现的键（差值缺失，判为不在容差内） */
export function diffRows(diff: Obj, tol: Obj, within: Obj): DiffRow[] {
  const keys = [...new Set([...Object.keys(diff), ...Object.keys(tol)])].filter((k) => num(tol[k]) !== null)
  return keys.map((key) => {
    const d = num(diff[key])
    const tl = num(tol[key])!
    const w = typeof within[key] === 'boolean' ? (within[key] as boolean) : null
    const recomputed = d !== null && d <= tl
    return { key, diff: d, tol: tl, within: w, ratio: d !== null && tl > 0 ? d / tl : null, recomputed, consistent: w === recomputed }
  })
}

function parseCase(x: unknown, tiers: TierPair): TierCase {
  const [ta, tb] = tiers
  if (!isObj(x)) throw new TierCompareDocError('case', 'case is not an object')
  const id = str(x.id)
  if (!id || !CASE_ID_RE.test(id)) throw new TierCompareDocError('case', `case id ${String(x.id)}`)
  const m = obj(x.metrics)
  const rows = diffRows(obj(x.diff), obj(x.tol), obj(x.within))
  const tolPass = bool(x.tol_pass)
  const posRms: Record<string, number> = {}
  for (const [u, v] of Object.entries(obj(x.pos_rms_m))) if (UAV_RE.test(u) && num(v) !== null) posRms[u] = v as number
  const iface = parseInterface(x.interface, tiers)
  const uavs = [...new Set([...Object.keys(posRms), ...Object.keys(iface.status)])].filter((u) => UAV_RE.test(u))
  const ratios = rows.map((r) => r.ratio).filter((r): r is number => r !== null)
  const w = isObj(x.wall_s) ? x.wall_s : null
  return {
    id, title: str(x.title) ?? id, iface, failures: strs(x.failures),
    metrics: { fast: metricMap(m[ta]), realistic: metricMap(m[tb]) }, posRms, rows, tolPass, pass: bool(x.pass),
    consistent: rows.every((r) => r.consistent) && tolPass === rows.every((r) => r.recomputed),
    maxRatio: ratios.length ? Math.max(...ratios) : null,
    wallS: w ? { fast: num(w[ta]), realistic: num(w[tb]) } : null,
    uavs,
  }
}

function parseSampling(x: unknown): Sampling | null {
  if (!isObj(x)) return null
  const p = isObj(x.pose_history) ? x.pose_history : null
  const m = obj(x.m25)
  const li = isObj(m.lidar) ? m.lidar : null
  const im = isObj(m.imu) ? m.imu : null
  const uavs: Record<string, PoseCheck> = {}
  if (p) {
    for (const [u, v] of Object.entries(obj(p.uavs))) {
      if (!isObj(v) || !UAV_RE.test(u)) continue
      uavs[u] = { posMaxM: num(v.pos_max_m), attMaxRad: num(v.att_max_rad), velBitwise: bool(v.vel_bitwise), omegaBitwise: bool(v.omega_bitwise), fBBitwise: bool(v.f_b_bitwise), pass: bool(v.pass) }
    }
  }
  const pt = obj(p?.tol)
  const it = obj(im?.tol)
  return {
    pose: p ? { tBitwise: bool(p.t_bitwise), n: num(p.n), uavs, tolPosM: num(pt.pos_m), tolAttRad: num(pt.att_rad), pass: bool(p.pass) } : null,
    lidar: li ? {
      pointWorldMaxM: num(li.point_world_max_m), originMaxM: num(li.origin_max_m), frames: num(li.frames), points: num(li.points),
      pointsEqualFrames: num(li.points_equal_frames), tolM: num(li.tol_m), pass: bool(li.pass),
    } : null,
    imu: im ? {
      accMaxMps2: num(im.acc_max_mps2), gyroMaxRps: num(im.gyro_max_rps), samples: num(im.samples), bitwiseBlocks: num(im.bitwise_blocks), blocks: num(im.blocks),
      tolAccMps2: num(it.acc_mps2), tolGyroRps: num(it.gyro_rps), pass: bool(im.pass),
    } : null,
    pass: bool(x.pass),
  }
}

function parseM25(x: unknown): M25Bytewise | null {
  if (!isObj(x)) return null
  const runs = Array.isArray(x.runs) ? x.runs.filter(isObj) : []
  const keys = [...new Set(runs.flatMap((r) => Object.keys(obj(r.sha256))))].sort()
  return {
    streams: keys.map((key) => ({
      key, sha256: runs.map((r) => str(obj(r.sha256)[key]) ?? ''), frames: num(obj(runs[0]?.frames)[key]),
    })),
    world: str(x.world), renderer: str(runs[0]?.renderer), pass: bool(x.pass),
  }
}

/** 解析并校验 `tier_compare.json`；schema 不符或无用例时整体拒绝 */
/** 报告的 `tiers`（两个档名，`build_report` 的 meta）；缺失或不合规时按封板前制品的 `[fast, realistic]` */
function parseTiers(x: unknown): TierPair {
  const ok = Array.isArray(x) && x.length === 2 && x.every((v) => typeof v === 'string' && /^[a-z][a-z0-9_]{0,15}$/.test(v)) && x[0] !== x[1]
  return ok ? [x[0] as string, x[1] as string] : [TIERS[0], TIERS[1]]
}

export function parseTierCompare(raw: unknown): TierCompareDoc {
  if (!isObj(raw) || raw.schema !== TIER_COMPARE_SCHEMA) throw new TierCompareDocError('schema', `schema ${isObj(raw) ? String(raw.schema) : typeof raw}`)
  if (!Array.isArray(raw.cases) || !raw.cases.length) throw new TierCompareDocError('cases', 'no cases')
  const tiers = parseTiers(raw.tiers)
  const cases = raw.cases.map((c) => parseCase(c, tiers))
  if (new Set(cases.map((c) => c.id)).size !== cases.length) throw new TierCompareDocError('cases', 'duplicate case id')
  const cp = isObj(raw.clock_profile) ? raw.clock_profile : null
  const gz = isObj(raw.gz) ? raw.gz : null
  const ts = str(raw.tolerance_status)
  const s = obj(raw.summary)
  return {
    schema: TIER_COMPARE_SCHEMA, tiers, generatedUtc: str(raw.generated_utc), vehicle: str(raw.vehicle),
    clockProfile: cp ? { name: str(cp.name), mainHz: num(cp.main_hz), physicsHz: num(cp.physics_hz) } : str(raw.clock_profile) ? { name: str(raw.clock_profile), mainHz: null, physicsHz: null } : null,
    tickMs: num(raw.tick_ms), poseHistoryHz: num(raw.pose_history_hz), windMode: str(raw.wind_mode), turbulenceSigma: num(raw.turbulence_sigma),
    dragLaw: str(raw.drag_law), collisionSource: str(raw.collision_source), m09Autonomy: str(raw.m09_autonomy), realisticTracker: str(raw[`${tiers[1]}_tracker`]),
    kernel: str(raw.kernel), gz: gz ? { simVersion: str(gz.gz_sim_version), pluginBuild: str(gz.plugin_build) } : null,
    toleranceStatus: ts, toleranceFrozen: ts !== null && /^frozen\b/i.test(ts.trim()),
    cases,
    summary: {
      cases: num(s.cases) ?? cases.length, interfacePass: num(s.interface_pass) ?? cases.filter((c) => c.iface.pass).length,
      tolPass: num(s.tol_pass) ?? cases.filter((c) => c.tolPass).length, pass: num(s.pass) ?? cases.filter((c) => c.pass).length,
    },
    sampling: parseSampling(raw.sampling_consistency), m25: parseM25(raw.m25_bytewise), pass: bool(raw.pass),
  }
}

/** 容差裕度最小的用例（diff / tol 最大） */
export function tightestCase(doc: TierCompareDoc): TierCase | null {
  let best: TierCase | null = null
  for (const c of doc.cases) if (c.maxRatio !== null && (best === null || c.maxRatio > (best.maxRatio ?? -1))) best = c
  return best
}

// ---------------------------------------------------------------------------------------------------- 采样一致性与 M25 逐字节
/** 一行核对：`value` 实测，`limit` 阈值（同单位），`ok` 为页面判定（数值行为 value ≤ limit 且报告 pass；无阈值行取报告判定） */
export interface CheckRow { key: string; uav?: string; value: number | string | null; limit: number | null; unit: string; ok: boolean | null }
const within = (v: number | null, lim: number | null, pass: boolean): boolean | null => (v === null || lim === null ? null : v <= lim && pass)

export function samplingRows(s: Sampling | null): CheckRow[] {
  if (!s) return []
  const out: CheckRow[] = []
  if (s.pose) {
    out.push({ key: 'pose.grid', value: s.pose.n, limit: null, unit: '', ok: s.pose.tBitwise })
    for (const [u, p] of Object.entries(s.pose.uavs)) {
      out.push({ key: 'pose.pos', uav: u, value: p.posMaxM, limit: s.pose.tolPosM, unit: 'm', ok: within(p.posMaxM, s.pose.tolPosM, p.pass) })
      out.push({ key: 'pose.att', uav: u, value: p.attMaxRad, limit: s.pose.tolAttRad, unit: 'rad', ok: within(p.attMaxRad, s.pose.tolAttRad, p.pass) })
      out.push({ key: 'pose.bitwise', uav: u, value: null, limit: null, unit: '', ok: p.velBitwise && p.omegaBitwise && p.fBBitwise })
    }
  }
  if (s.lidar) {
    const l = s.lidar
    out.push({ key: 'lidar.point', value: l.pointWorldMaxM, limit: l.tolM, unit: 'm', ok: within(l.pointWorldMaxM, l.tolM, l.pass) })
    out.push({ key: 'lidar.frames', value: l.frames === null ? null : `${l.pointsEqualFrames ?? '—'} / ${l.frames}`, limit: null, unit: '', ok: l.frames !== null && l.pointsEqualFrames === l.frames && l.pass })
    out.push({ key: 'lidar.points', value: l.points, limit: null, unit: '', ok: null })
  }
  if (s.imu) {
    const m = s.imu
    out.push({ key: 'imu.acc', value: m.accMaxMps2, limit: m.tolAccMps2, unit: 'm/s²', ok: within(m.accMaxMps2, m.tolAccMps2, m.pass) })
    out.push({ key: 'imu.gyro', value: m.gyroMaxRps, limit: m.tolGyroRps, unit: 'rad/s', ok: within(m.gyroMaxRps, m.tolGyroRps, m.pass) })
    out.push({ key: 'imu.blocks', value: m.blocks === null ? null : `${m.bitwiseBlocks ?? '—'} / ${m.blocks}`, limit: null, unit: '', ok: m.blocks !== null && m.bitwiseBlocks === m.blocks && m.pass })
  }
  return out
}

/** M25 逐字节：同一 PoseHistory 两次送入 M25，每路输出摘要须全部相同 */
export function m25Rows(m: M25Bytewise | null): CheckRow[] {
  if (!m) return []
  return m.streams.map((s) => ({
    key: 'm25.stream', uav: s.key, value: s.sha256[0] ? s.sha256[0].slice(0, 12) : null, limit: null, unit: '',
    ok: s.sha256.length >= 2 && s.sha256.every((h) => h !== '' && h === s.sha256[0]),
  }))
}

// ---------------------------------------------------------------------------------------------------- 轨迹
export interface TierTrack { tier: Tier; n: number; tNs: BigInt64Array; tS: Float64Array; state: ReadonlyMap<string, Float64Array> }

/** 从 `CaseRun.save` 的 `.npz` 构建一档轨迹：`t_ns` i64 (n,) 严格递增，`s_<uav>` f64 (n, 16) */
export function trackFromArrays(arrays: ReadonlyMap<string, NpyArray>, tier: Tier): TierTrack {
  const t = arrays.get('t_ns')
  if (!t || t.dtype !== 'i8' || t.shape.length !== 1) throw new TrackError('t_ns', 't_ns must be int64 (n,)')
  const tNs = t.data as BigInt64Array
  const n = tNs.length
  for (let i = 1; i < n; i++) if (tNs[i]! <= tNs[i - 1]!) throw new TrackError('t_ns', `t_ns not increasing at ${i}`)
  const tS = new Float64Array(n)                                    // 仿真秒（PoseHistory 时刻以回合起点为 0）
  for (let i = 0; i < n; i++) tS[i] = Number(tNs[i]!) / 1e9
  const state = new Map<string, Float64Array>()
  for (const [k, a] of arrays) {
    if (!k.startsWith('s_')) continue
    const uav = k.slice(2)
    if (!UAV_RE.test(uav)) continue
    if (a.dtype !== 'f8' || a.shape.length !== 2 || a.shape[0] !== n || a.shape[1] !== STATE_N) throw new TrackError('state', `${k} must be float64 (${n}, ${STATE_N})`)
    state.set(uav, a.data as Float64Array)
  }
  if (!state.size) throw new TrackError('state', 'no s_<uav> arrays')
  return { tier, n, tNs, tS, state }
}

export async function decodeTrack(buf: ArrayBuffer, tier: Tier): Promise<TierTrack> {
  return trackFromArrays(await readNpz(buf), tier)
}

export const trackFile = (caseId: string, tier: Tier): string => `${caseId}.${tier}.npz`

/** 两档规格核对：机体集合相同、样本数相同、时刻逐位相同（同一 Run 规格在两档下的 PoseHistory 网格） */
export interface Alignment { ok: boolean; reason: 'uavs' | 'samples' | 'time' | null; n: number; uavs: string[] }
export function alignTracks(a: TierTrack, b: TierTrack): Alignment {
  const ua = [...a.state.keys()]
  const ub = new Set(b.state.keys())
  const uavs = ua.filter((u) => ub.has(u))
  if (uavs.length !== ua.length || uavs.length !== ub.size) return { ok: false, reason: 'uavs', n: Math.min(a.n, b.n), uavs }
  if (a.n !== b.n) return { ok: false, reason: 'samples', n: Math.min(a.n, b.n), uavs }
  for (let i = 0; i < a.n; i++) if (a.tNs[i] !== b.tNs[i]) return { ok: false, reason: 'time', n: a.n, uavs }
  return { ok: true, reason: null, n: a.n, uavs }
}

/** 逐样本位置差 |p_fast − p_real|（m），只取两档共有的前 n 个样本 */
export function posDiff(a: TierTrack, b: TierTrack, uav: string): Float64Array {
  const sa = a.state.get(uav)
  const sb = b.state.get(uav)
  const n = Math.min(a.n, b.n)
  const out = new Float64Array(sa && sb ? n : 0)
  if (!sa || !sb) return out
  for (let i = 0; i < n; i++) {
    const o = i * STATE_N
    const dx = sa[o]! - sb[o]!
    const dy = sa[o + 1]! - sb[o + 1]!
    const dz = sa[o + 2]! - sb[o + 2]!
    out[i] = Math.sqrt(dx * dx + dy * dy + dz * dz)
  }
  return out
}

/** 位置 RMS，与 `compare_case` 同式：sqrt(mean(sum((p_fast − p_real)², axis = 1))) */
export function posRmsOf(a: TierTrack, b: TierTrack, uav: string): number | null {
  const sa = a.state.get(uav)
  const sb = b.state.get(uav)
  if (!sa || !sb || a.n !== b.n || !a.n) return null
  let s = 0
  for (let i = 0; i < a.n; i++) {
    const o = i * STATE_N
    const dx = sa[o]! - sb[o]!
    const dy = sa[o + 1]! - sb[o + 1]!
    const dz = sa[o + 2]! - sb[o + 2]!
    s += dx * dx + dy * dy + dz * dz
  }
  return Math.sqrt(s / a.n)
}

export interface RmsCheck { uav: string; page: number | null; report: number | null; absDiff: number | null; ok: boolean }
/** 图表与报告一致：每机由轨迹重算的位置 RMS 与 `pos_rms_m` 之差 ≤ POS_RMS_CHECK_M，`diff.pos_rms_m` 为各机最大值 */
export function checkPosRms(c: TierCase, a: TierTrack, b: TierTrack): { rows: RmsCheck[]; diffOk: boolean | null; ok: boolean } {
  const rows = c.uavs.map((uav) => {
    const page = posRmsOf(a, b, uav)
    const report = c.posRms[uav] ?? null
    const absDiff = page !== null && report !== null ? Math.abs(page - report) : null
    return { uav, page, report, absDiff, ok: absDiff !== null && absDiff <= POS_RMS_CHECK_M }
  })
  const row = c.rows.find((r) => r.key === 'pos_rms_m')
  const reportMax = Object.values(c.posRms).length ? Math.max(...Object.values(c.posRms)) : null
  const diffOk = row ? row.diff !== null && reportMax !== null && Math.abs(row.diff - reportMax) <= POS_RMS_CHECK_M : null
  return { rows, diffOk, ok: rows.length > 0 && rows.every((r) => r.ok) && diffOk !== false }
}

/** 抽稀步长：⌈n / max⌉，至少 1（确定性，首尾样本保留） */
export const plotStride = (n: number, max = PLOT_MAX_POINTS): number => Math.max(1, Math.ceil(n / Math.max(1, max)))

/** 抽稀后的样本下标（含最后一个样本） */
export function plotIndices(n: number, max = PLOT_MAX_POINTS): number[] {
  const s = plotStride(n, max)
  const out: number[] = []
  for (let i = 0; i < n; i += s) out.push(i)
  if (n && out[out.length - 1] !== n - 1) out.push(n - 1)
  return out
}

/** 不晚于 t 的最近样本下标（二分；t 早于首样本时为 0） */
export function sampleAt(tS: Float64Array, t: number): number {
  let lo = 0
  let hi = tS.length - 1
  if (hi < 0) return -1
  if (t <= tS[0]!) return 0
  if (t >= tS[hi]!) return hi
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (tS[mid]! <= t) lo = mid
    else hi = mid
  }
  return lo
}

export type Vec3 = readonly [number, number, number]
export const posAt = (s: Float64Array, i: number): Vec3 => [s[i * STATE_N]!, s[i * STATE_N + 1]!, s[i * STATE_N + 2]!]

/** 平面图范围（E、N，m）：两档全部样本的包围盒，外扩 5%，最小跨度 `minSpan`，两轴等比例由绘制方处理 */
export function planExtent(states: readonly Float64Array[], minSpan = 2): { e: readonly [number, number]; n: readonly [number, number] } {
  let e0 = Infinity
  let e1 = -Infinity
  let n0 = Infinity
  let n1 = -Infinity
  for (const s of states) {
    for (let o = 0; o + 2 < s.length; o += STATE_N) {
      const e = s[o]!
      const n = s[o + 1]!
      if (e < e0) e0 = e
      if (e > e1) e1 = e
      if (n < n0) n0 = n
      if (n > n1) n1 = n
    }
  }
  if (!Number.isFinite(e0)) return { e: [-1, 1], n: [-1, 1] }
  const pad = (a: number, b: number): readonly [number, number] => {
    const span = Math.max(minSpan, (b - a) * 1.1)
    const c = (a + b) / 2
    return [c - span / 2, c + span / 2]
  }
  return { e: pad(e0, e1), n: pad(n0, n1) }
}

// ---------------------------------------------------------------------------------------------------- 文本与单位
/** 指标键的单位（按 awr 单位后缀；`<uav>.` 前缀不计） */
export function metricUnit(key: string): string {
  const k = key.includes('.') ? key.slice(key.lastIndexOf('.') + 1) : key
  if (k.endsWith('_s_per_100m')) return 's/100 m'
  if (k.endsWith('_mps2')) return 'm/s²'
  if (k.endsWith('_mps')) return 'm/s'
  if (k.endsWith('_rps')) return 'rad/s'
  if (k.endsWith('_pct')) return '%'
  if (k.endsWith('_deg')) return '°'
  if (k.endsWith('_rad')) return 'rad'
  if (k.endsWith('_m')) return 'm'
  if (k.endsWith('_s')) return 's'
  return ''
}
/** `wall.t_crash_s` → { prefix: 'wall', base: 't_crash_s' } */
export function metricParts(key: string): { prefix: string | null; base: string } {
  const i = key.indexOf('.')
  return i > 0 ? { prefix: key.slice(0, i), base: key.slice(i + 1) } : { prefix: null, base: key }
}
/** 数值显示：4 位有效数字，绝对值 < 1e-3 用科学计数，0 显示 0 */
export function fmtNum(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—'
  if (v === 0) return '0'
  if (Math.abs(v) < 1e-3) return v.toExponential(2)
  return String(Number(v.toPrecision(4)))
}
export function fmtMetric(v: MetricValue | undefined): string {
  if (v === undefined || v === null) return '—'
  if (typeof v === 'number') return fmtNum(v)
  if (typeof v === 'string') return v
  if (typeof v === 'boolean') return v ? 'true' : 'false'
  return v.map((x) => fmtNum(x)).join(', ')
}

// ---------------------------------------------------------------------------------------------------- URL 与制品路径
export interface TierCompareSearch { run: string | null; caseId: string | null; uav: string | null; t: number | null }
/** 查询参数：`run`（承载对照制品的运行；缺省时取 `b`，即高保真仿真一侧的运行）、`case`、`uav`、`t`（仿真秒） */
export function parseTierCompareSearch(s: URLSearchParams): TierCompareSearch {
  const run = s.get('run') ?? s.get('b')
  const c = s.get('case')
  const u = s.get('uav')
  const t = Number(s.get('t'))
  return {
    run: run !== null && RUN_ID_RE.test(run) ? run : null, caseId: c !== null && CASE_ID_RE.test(c) ? c : null,
    uav: u !== null && UAV_RE.test(u) ? u : null, t: s.get('t') !== null && Number.isFinite(t) && t >= 0 ? t : null,
  }
}
export function resolveCase(doc: TierCompareDoc, caseId: string | null): TierCase {
  return doc.cases.find((c) => c.id === caseId) ?? doc.cases[0]!
}
export const resolveUav = (c: TierCase, uav: string | null): string | null => (uav !== null && c.uavs.includes(uav) ? uav : (c.uavs[0] ?? null))

/** 制品下载地址（BN-54）：`/runs/{rid}/artifacts/tier_compare/{file}`；文件名由已校验的用例 id 与档名构成 */
export const tierCompareArtifactPath = (rid: string, file: string): string => `/runs/${encodeURIComponent(rid)}/artifacts/tier_compare/${file}`
