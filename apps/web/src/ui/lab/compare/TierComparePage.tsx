// /lab/compare，快速仿真与高保真仿真对照（D3-AC-35 验证方法列"Playwright lab/compare"；AWR-05 §5.3.6 档间差异报告；AWR-14 §21.10；UX-AC-097
// "两运行共用播放头；规格不一致时提示；差异图表与评测文件一致"；M24-AC-016、AC-028）。内部 D3 构建，最低角色研究者，只读。
// - 数据：BN-54 `GET /runs/{run}/artifacts/tier_compare/`下的 `tier_compare.json` 与所选用例的 `<case>.<tier>.npz`（两档名取报告的 `tiers`：
//   `fast` 与 `gz`，封板前的制品为 `fast` 与 `realistic`，ADR-226）
//   （tierCompare.ts）；`run` 缺省时取 `b`（高保真仿真一侧的运行）；
// - 左栏：用例表（接口判定、容差判定、容差裕度）与报告汇总（通过数、容差裕度最小的用例、容差状态）；
// - 主区：两档轨迹平面图（快速仿真实线、高保真仿真虚线）与高度曲线、逐样本位置差曲线（报告位置 RMS 为参考线），三者共用 lab 播放头；
//   指标差带（差值相对容差带）与两档指标表；接口层判定（指令回执、VehicleStatus 序列、接触事件、PoseHistory 网格、指令生效步、限幅）；
// - 规格核对：两档机体集合、样本数与时刻须逐位相同，否则提示"对照结果不可比"；由轨迹重算的位置 RMS 须与报告一致（≤ 1e-9 m），报告的
//   within 须与 diff ≤ tol 一致，不一致时提示（data-rms-check-*、data-band-consistent）；
// - 底部：两档采样一致性（注入模式 PoseHistory、M25 LiDAR 逐点位姿差 ≤ 1 mm、IMU 差）与 M25 同一 PoseHistory 逐字节结果；
// - URL：`run`、`case`、`uav`、`t`（播放头以 1 Hz 静默写回）。
import { useQueries, useQuery, type UseQueryResult } from '@tanstack/react-query'
import * as React from 'react'
import { hasKey, useT } from '@/app/i18n'
import { navigate, updateSearch, type RouteProps } from '@/app/router/router'
import { sanitizeText } from '@/lib/sanitize'
import { BenchError, benchBytes, benchGet, ROLE_RANK } from '@/net/bench'
import { usePlayhead } from '@/stores/lab/playhead'
import { PanelEmpty } from '@/ui/brand'
import { Alert, AlertDescription, AlertTitle } from '@/ui/components/ui/alert'
import { Badge } from '@/ui/components/ui/badge'
import { Button } from '@/ui/components/ui/button'
import { NativeSelect, NativeSelectOption } from '@/ui/components/ui/native-select'
import { Slider } from '@/ui/components/ui/slider'
import { Icon } from '@/ui/icons/Icon'
import { useBenchSession } from '@/ui/lab/useBenchSession'
import { useDocTitle } from '@/ui/lab/worlds/LabHeader'
import { useElementWidth } from '@/ui/layout/useElementWidth'
import { LfChartCard } from '@/ui/lf/LfChartCard'
import { LfStat } from '@/ui/lf/LfStat'
import { LfTable, type LfColumn } from '@/ui/lf/LfTable'
import { DiffBands, TimeChart, TrackPlan, type TimeSeries } from './charts'
import {
  alignTracks, checkPosRms, decodeTrack, fmtMetric, fmtNum, m25Rows, metricParts, metricUnit, parseTierCompare, parseTierCompareSearch, planExtent, posAt,
  posDiff, POS_RMS_CHECK_M, REPORT_FILE, resolveCase, resolveUav, sampleAt, samplingRows, STATE_N, tierCompareArtifactPath, TierCompareDocError, TIERS,
  tightestCase, trackFile, type CheckRow, type DiffRow, type MetricValue, type Tier, type TierCase, type TierCompareDoc, type TierTrack,
} from './tierCompare'

export const tierCompareDocKey = (rid: string) => ['lab', 'tier-compare', rid, 'doc'] as const
export const tierCompareTrackKey = (rid: string, caseId: string, tier: Tier) => ['lab', 'tier-compare', rid, 'track', caseId, tier] as const
const URL_WRITE_MS = 1000
const STEP_S = 0.1
const WIDE_MIN_W = 1100
const SIDE_W = 360

/** 两档轨迹查询的合并结果（模块级函数，各查询结果不变时 TanStack Query 复用上一次的返回值） */
function combineTracks(rs: readonly UseQueryResult<TierTrack>[]) {
  return {
    fast: rs[0]?.data ?? null, real: rs[1]?.data ?? null, pending: rs.some((r) => r.isPending && r.fetchStatus !== 'idle'),
    failed: TIERS.filter((_, i) => rs[i]?.isError), missing: rs.some((r) => r.error instanceof BenchError && r.error.status === 404),
  }
}

const clock = (s: number) => `${s.toFixed(2)} s`
const xyz = (p: readonly number[]) => p.map((v) => v.toFixed(2)).join(', ')
/** 表格高度：表头与每行留足两行文字的高度，不出现内部滚动（行数均为个位或十余行） */
const tableH = (rows: number) => 48 + rows * 40

function Verdict({ ok, label }: { ok: boolean | null; label?: string }) {
  const t = useT()
  if (ok === null) return <span className="text-muted-foreground">—</span>
  return (
    <span className={`inline-flex items-center gap-1 ${ok ? '' : 'text-brand-text'}`} data-verdict={String(ok)}>
      <Icon icon={ok ? 'check' : 'alert.warning'} />
      {label ?? t(ok ? 'lab.compare.ok' : 'lab.compare.notOk')}
    </span>
  )
}

function useLabels() {
  const t = useT()
  return React.useMemo(() => ({
    caseTitle: (c: TierCase) => (hasKey(`lab.compare.case.${c.id}`) ? t(`lab.compare.case.${c.id}`) : c.title),
    metric: (key: string) => {
      const { prefix, base } = metricParts(key)
      const l = hasKey(`lab.compare.metric.${base}`) ? t(`lab.compare.metric.${base}`) : base
      return prefix ? `${prefix} · ${l}` : l
    },
    tier: (tier: Tier) => t(`lab.compare.tier.${tier}`),
  }), [t])
}

function withUnit(v: string, unit: string): string {
  return v === '—' || !unit ? v : unit === '%' || unit === '°' ? `${v}${unit}` : `${v} ${unit}`
}

// ---------------------------------------------------------------------------------------------------- 左栏
function CaseTable({ doc, sel }: { doc: TierCompareDoc; sel: TierCase }) {
  const t = useT()
  const L = useLabels()
  const cols: LfColumn<TierCase>[] = [
    { key: 'case', label: t('lab.compare.cases.case'), format: (c) => <span data-case-id={c.id}>{L.caseTitle(c)}</span> },
    { key: 'iface', label: t('lab.compare.cases.iface'), format: (c) => <Verdict ok={c.iface.pass} label={c.iface.pass ? t('lab.compare.ok') : t('lab.compare.notOk')} /> },
    { key: 'tol', label: t('lab.compare.cases.tol'), format: (c) => <Verdict ok={c.tolPass} /> },
    { key: 'margin', label: t('lab.compare.cases.ratio'), unit: '%', align: 'right', format: (c) => (c.maxRatio === null ? '—' : (c.maxRatio * 100).toFixed(0)) },
  ]
  const failing = doc.cases.find((c) => !c.pass)
  return (
    <LfTable columns={cols} rows={doc.cases} rowKey={(c) => c.id} selected={sel.id} hot={failing ? { row: failing.id, col: 'tol' } : null}
      onRowClick={(c) => updateSearch({ case: c.id, uav: null })} height={tableH(doc.cases.length)} rowHeight={30}
      ariaLabel={t('lab.compare.cases.title')} figureId="lab-compare-cases" />
  )
}

function Summary({ doc }: { doc: TierCompareDoc }) {
  const t = useT()
  const L = useLabels()
  const tight = tightestCase(doc)
  const n = doc.summary.cases
  return (
    <div className="flex flex-col gap-2" data-compare-summary={`${doc.summary.pass}/${n}`}>
      <div className="grid grid-cols-3 gap-x-4">
        <LfStat label={t('lab.compare.summary.pass')} value={doc.summary.pass} format={(v) => `${v} / ${n}`} />
        <LfStat label={t('lab.compare.summary.iface')} value={doc.summary.interfacePass} format={(v) => `${v} / ${n}`} />
        <LfStat label={t('lab.compare.summary.tol')} value={doc.summary.tolPass} format={(v) => `${v} / ${n}`} />
      </div>
      {tight && tight.maxRatio !== null ? (
        <span className="text-hud-sub text-muted-foreground" data-compare-tightest={tight.id}>
          {t('lab.compare.summary.tightest', { name: L.caseTitle(tight), pct: (tight.maxRatio * 100).toFixed(0) })}
        </span>
      ) : null}
      <span className="text-hud-sub text-muted-foreground" data-tolerance-status={doc.toleranceFrozen ? 'frozen' : 'initial'}>
        {t(doc.toleranceFrozen ? 'lab.compare.summary.frozen' : 'lab.compare.summary.initial')}
      </span>
    </div>
  )
}

// ---------------------------------------------------------------------------------------------------- 主区表格
interface MetricRow { key: string; fast: MetricValue | undefined; real: MetricValue | undefined; diff: DiffRow | null }
function MetricTable({ c, tb }: { c: TierCase; tb: Tier }) {
  const t = useT()
  const L = useLabels()
  const keys = [...new Set([...Object.keys(c.metrics.fast), ...Object.keys(c.metrics.realistic), ...c.rows.map((r) => r.key)])]
  const rows: MetricRow[] = keys.map((key) => ({ key, fast: c.metrics.fast[key], real: c.metrics.realistic[key], diff: c.rows.find((r) => r.key === key) ?? null }))
  const u = (r: MetricRow) => metricUnit(r.key)
  const cols: LfColumn<MetricRow>[] = [
    { key: 'metric', label: t('lab.compare.metrics.metric'), format: (r) => <span title={r.key}>{L.metric(r.key)}</span> },
    { key: 'fast', label: L.tier('fast'), align: 'right', format: (r) => withUnit(fmtMetric(r.fast), typeof r.fast === 'string' ? '' : u(r)) },
    { key: 'real', label: L.tier(tb), align: 'right', format: (r) => withUnit(fmtMetric(r.real), typeof r.real === 'string' ? '' : u(r)) },
    { key: 'diff', label: t('lab.compare.metrics.diff'), align: 'right', format: (r) => (r.diff ? withUnit(fmtNum(r.diff.diff), u(r)) : '') },
    { key: 'tol', label: t('lab.compare.metrics.tol'), align: 'right', format: (r) => (r.diff ? withUnit(fmtNum(r.diff.tol), u(r)) : '') },
    { key: 'within', label: t('lab.compare.metrics.within'), format: (r) => (r.diff ? <Verdict ok={r.diff.recomputed} /> : '') },
  ]
  const out = rows.find((r) => r.diff && !r.diff.recomputed)
  return <LfTable columns={cols} rows={rows} rowKey={(r) => r.key} hot={out ? { row: out.key, col: 'diff' } : null} height={tableH(rows.length)}
    ariaLabel={t('lab.compare.metrics.title')} figureId="lab-compare-metrics" />
}

interface IfaceRow { key: string; item: string; fast: string; real: string; expect: string; ok: boolean | null }
function interfaceRows(c: TierCase, t: (k: string, p?: Record<string, string | number>) => string): IfaceRow[] {
  const seq = (s: readonly string[]) => (s.length ? s.join(' → ') : '—')
  const i = c.iface
  const out: IfaceRow[] = []
  if (i.dispatchEqual !== null) out.push({ key: 'dispatch', item: t('lab.compare.iface.dispatch'), fast: '', real: '', expect: '', ok: i.dispatchEqual })
  for (const [u, s] of Object.entries(i.status)) out.push({ key: `status/${u}`, item: t('lab.compare.iface.status', { uav: u }), fast: seq(s.fast), real: seq(s.realistic), expect: seq(s.expected), ok: s.pass })
  for (const [u, s] of Object.entries(i.contacts)) out.push({ key: `contacts/${u}`, item: t('lab.compare.iface.contacts', { uav: u }), fast: seq(s.fast), real: seq(s.realistic), expect: seq(s.expected), ok: s.pass })
  if (i.grid) {
    out.push({
      key: 'grid', item: t('lab.compare.iface.grid'), fast: '', real: '',
      expect: t('lab.compare.iface.gridExpect', { n: i.grid.samples ?? '—' }), ok: i.grid.pass,
    })
  }
  if (i.onset) out.push({ key: 'onset', item: t('lab.compare.iface.onset'), fast: withUnit(fmtNum(i.onset.fastS), 's'), real: withUnit(fmtNum(i.onset.realisticS), 's'), expect: `≤ ${withUnit(fmtNum(i.onset.maxS), 's')}`, ok: i.onset.pass })
  if (i.limitCap) out.push({ key: 'limit', item: t('lab.compare.iface.limit'), fast: withUnit(fmtNum(i.limitCap.fastMaxMps), 'm/s'), real: withUnit(fmtNum(i.limitCap.realisticMaxMps), 'm/s'), expect: `≤ ${withUnit(fmtNum(i.limitCap.capMps), 'm/s')}`, ok: i.limitCap.pass })
  return out
}
function InterfaceTable({ c, tb }: { c: TierCase; tb: Tier }) {
  const t = useT()
  const L = useLabels()
  const rows = interfaceRows(c, t)
  const cols: LfColumn<IfaceRow>[] = [
    { key: 'item', label: t('lab.compare.iface.item'), format: (r) => r.item },
    { key: 'fast', label: L.tier('fast'), format: (r) => <span className="font-mono">{r.fast}</span> },
    { key: 'real', label: L.tier(tb), format: (r) => <span className="font-mono">{r.real}</span> },
    { key: 'expect', label: t('lab.compare.iface.expect'), format: (r) => <span className="font-mono">{r.expect}</span> },
    { key: 'ok', label: t('lab.compare.iface.result'), format: (r) => <Verdict ok={r.ok} /> },
  ]
  const bad = rows.find((r) => r.ok === false)
  return <LfTable columns={cols} rows={rows} rowKey={(r) => r.key} hot={bad ? { row: bad.key, col: 'ok' } : null} height={tableH(rows.length)}
    ariaLabel={t('lab.compare.iface.title')} figureId="lab-compare-interface" />
}

function CheckTable({ rows, title, figure }: { rows: CheckRow[]; title: string; figure: string }) {
  const t = useT()
  const val = (r: CheckRow) => (typeof r.value === 'number' ? withUnit(fmtNum(r.value), r.unit) : (r.value ?? ''))
  const cols: LfColumn<CheckRow>[] = [
    { key: 'item', label: t('lab.compare.check.item'), format: (r) => <span data-check-row={r.key} data-check-ok={String(r.ok)}>{t(`lab.compare.check.${r.key}`, { uav: r.uav ?? '' })}</span> },
    { key: 'value', label: t('lab.compare.check.value'), align: 'right', format: (r) => <span className="font-mono">{val(r)}</span> },
    { key: 'limit', label: t('lab.compare.check.limit'), align: 'right', format: (r) => (r.limit === null ? '' : `≤ ${withUnit(fmtNum(r.limit), r.unit)}`) },
    { key: 'ok', label: t('lab.compare.check.result'), format: (r) => <Verdict ok={r.ok} /> },
  ]
  if (!rows.length) return <span className="text-hud-sub text-muted-foreground">{t('lab.compare.check.none')}</span>
  return <LfTable columns={cols} rows={rows} rowKey={(r, i) => `${r.key}/${r.uav ?? ''}/${i}`} height={tableH(rows.length)} ariaLabel={title} figureId={figure} />
}

/** 所选用例与机体的两档派生视图：规格核对、高度与位置差序列、平面图范围、位置 RMS 核对（轨迹或用例变化时重算） */
function deriveView(fast: TierTrack | null, real: TierTrack | null, sel: TierCase | null, uav: string | null) {
  if (!fast || !real || !sel || !uav) return null
  const align = alignTracks(fast, real)
  const n = Math.min(fast.n, real.n)
  const alt = (tr: TierTrack): Float64Array => {
    const s = tr.state.get(uav)
    const out = new Float64Array(n).fill(Number.NaN)
    if (s) for (let i = 0; i < n; i++) out[i] = s[i * STATE_N + 2]!
    return out
  }
  const diff = posDiff(fast, real, uav)
  const altSeries: TimeSeries[] = [{ key: 'fast', values: alt(fast), kind: 'fast' }, { key: 'realistic', values: alt(real), kind: 'realistic' }]
  const diffSeries: TimeSeries[] = [{ key: 'diff', values: diff, kind: 'diff' }]
  const extent = planExtent([fast.state.get(uav), real.state.get(uav)].filter((s): s is Float64Array => !!s))
  return {
    align, n, diff, altSeries, diffSeries, extent, sf: fast.state.get(uav) ?? null, sr: real.state.get(uav) ?? null,
    tS: fast.tS.length === n ? fast.tS : fast.tS.slice(0, n), rms: checkPosRms(sel, fast, real),
  }
}

// ---------------------------------------------------------------------------------------------------- 页面
export function TierComparePage({ search }: RouteProps) {
  const t = useT()
  const L = useLabels()
  const session = useBenchSession()
  const q = React.useMemo(() => parseTierCompareSearch(search), [search])
  const rid = q.run ?? ''
  useDocTitle(t('lab.compare.docTitle'))
  const canRead = session !== null && ROLE_RANK[session.role] >= ROLE_RANK.researcher
  const docQ = useQuery({
    queryKey: tierCompareDocKey(rid),
    queryFn: async ({ signal }) => parseTierCompare(await benchGet<unknown>(tierCompareArtifactPath(rid, REPORT_FILE), signal)),
    enabled: canRead && rid !== '', retry: false, retryOnMount: false, staleTime: Infinity,
  })
  const doc = docQ.data
  const sel = React.useMemo(() => (doc ? resolveCase(doc, q.caseId) : null), [doc, q.caseId])
  const uav = sel ? resolveUav(sel, q.uav) : null
  const caseId = sel?.id ?? ''
  const tracks = useQueries({
    queries: (doc?.tiers ?? TIERS).map((tier) => ({
      queryKey: tierCompareTrackKey(rid, caseId, tier),
      queryFn: async ({ signal }: { signal: AbortSignal }) => decodeTrack((await benchBytes(tierCompareArtifactPath(rid, trackFile(caseId, tier)), signal)).buf, tier),
      enabled: canRead && rid !== '' && caseId !== '', staleTime: Infinity, retry: 1, retryOnMount: false,
    })),
    combine: combineTracks,
  })
  const fast = tracks.fast
  const real = tracks.real

  const derived = React.useMemo(() => deriveView(fast, real, sel, uav), [fast, real, sel, uav])

  // lab 播放头：两档共用；进入页面时取 URL 的 t，轨迹就绪后以末样本时刻为时长，播放中以墙钟推进，1 Hz 静默写回 URL
  const ph = usePlayhead()
  React.useEffect(() => {
    usePlayhead.setState({ tS: q.t ?? 0, durationS: q.t ?? 0, playing: false, rate: 1 })
  }, []) // eslint-disable-line react-hooks/exhaustive-deps
  const endS = derived && derived.n ? derived.tS[derived.n - 1]! : null
  React.useEffect(() => {
    if (endS !== null) usePlayhead.getState().setDuration(endS)
  }, [endS])
  React.useEffect(() => {
    if (!ph.playing) return
    let last = performance.now()
    let raf = requestAnimationFrame(function step(now) {
      usePlayhead.getState().advance((now - last) / 1000)
      last = now
      raf = requestAnimationFrame(step)
    })
    return () => cancelAnimationFrame(raf)
  }, [ph.playing])
  React.useEffect(() => {
    const h = setInterval(() => {
      const v = usePlayhead.getState().tS.toFixed(2)
      if (new URLSearchParams(location.search).get('t') !== v) updateSearch({ t: v }, { silent: true })
    }, URL_WRITE_MS)
    return () => clearInterval(h)
  }, [])
  const [box, bw] = useElementWidth<HTMLDivElement>(1280)

  if (session !== null && !canRead) {
    return (
      <section data-view="lab-compare" className="app-layer-overlay-page bg-background p-4">
        <PanelEmpty title={t('lab.compare.roleDenied')} description={t('lab.compare.roleDeniedHint')}
          action={<Button size="sm" variant="outline" onClick={() => navigate('/lab/board')}>{t('lab.compare.toBoard')}</Button>} />
      </section>
    )
  }
  const header = (
    <header className="flex flex-wrap items-center gap-2 border-b px-4 py-2">
      <Button size="sm" variant="ghost" onClick={() => navigate(rid ? `/lab/runs/${encodeURIComponent(rid)}` : '/lab/runs')}>
        <Icon icon="chev.left" data-icon="inline-start" />
        {t('lab.compare.back')}
      </Button>
      <Icon icon="layer.trajectory" />
      <h1 className="text-ed-title font-bold">{t('lab.compare.title')}</h1>
      {rid ? <span className="font-mono text-hud-sub">{sanitizeText(rid, 128)}</span> : null}
      {doc ? (
        <>
          <span className="ml-auto font-mono text-hud-sub text-muted-foreground" data-compare-meta="">
            {[doc.vehicle, doc.clockProfile?.name, doc.poseHistoryHz !== null ? `PoseHistory ${doc.poseHistoryHz} Hz` : null,
              doc.windMode !== null ? t('lab.compare.meta.wind', { mode: doc.windMode, sigma: doc.turbulenceSigma ?? '—' }) : null, doc.dragLaw,
              doc.gz?.simVersion ? `Gazebo Sim ${doc.gz.simVersion}` : null].filter(Boolean).join(' · ')}
          </span>
          <Badge variant="outline" className="font-mono" data-report-pass={String(doc.pass)}>
            {t(doc.pass ? 'lab.compare.meta.pass' : 'lab.compare.meta.fail', { n: doc.summary.pass, total: doc.summary.cases })}
          </Badge>
        </>
      ) : null}
    </header>
  )
  if (!doc || !sel) {
    const missing = docQ.error instanceof BenchError && docQ.error.status === 404
    return (
      <section data-view="lab-compare" className="app-layer-overlay-page flex flex-col bg-background">
        {header}
        <div className="p-4">
          {rid === '' ? <PanelEmpty title={t('lab.compare.noRun')} description={t('lab.compare.noRunHint')} />
            : docQ.isPending && canRead ? <PanelEmpty title={t('lab.compare.loading')} brand={false} />
              : missing ? <PanelEmpty title={t('lab.compare.missing')} description={t('lab.compare.missingHint')} />
                : docQ.error instanceof TierCompareDocError ? <PanelEmpty title={t('lab.compare.invalid')} description={t('lab.compare.invalidHint')} />
                  : docQ.isError ? <PanelEmpty title={t('lab.compare.unavailable')} description={t('lab.unreachableHint')} /> : null}
        </div>
      </section>
    )
  }

  const wide = bw >= WIDE_MIN_W
  const mainW = Math.max(280, wide ? bw - SIDE_W - 16 : bw)
  const twoUp = mainW >= 760
  const chartW = Math.max(260, twoUp ? Math.floor((mainW - 16) / 2) - 24 : mainW - 24)
  const idx = derived && derived.n ? Math.min(sampleAt(derived.tS, ph.tS), derived.n - 1) : -1
  const sf = derived?.sf ?? null
  const sr = derived?.sr ?? null
  const pF = sf && idx >= 0 ? posAt(sf, idx) : null
  const pR = sr && idx >= 0 ? posAt(sr, idx) : null
  const dNow = derived && idx >= 0 ? derived.diff[idx] : undefined
  const rmsReport = uav !== null ? (sel.posRms[uav] ?? null) : null
  const rmsRow = derived?.rms.rows.find((r) => r.uav === uav) ?? null
  const rmsMax = derived ? Math.max(0, ...derived.rms.rows.map((r) => r.absDiff ?? Number.POSITIVE_INFINITY)) : null
  const bandsBad = sel.rows.filter((r) => !r.consistent)
  const alignOk = derived ? derived.align.ok : null
  const tracksNote = tracks.missing ? t('lab.compare.tracks.missing') : tracks.failed.length ? t('lab.compare.tracks.failed', { tiers: tracks.failed.map(L.tier).join(', ') })
    : tracks.pending ? t('lab.compare.tracks.loading') : null

  return (
    <section data-view="lab-compare" data-figure="lab-compare" data-compare-case={sel.id} data-compare-uav={uav ?? ''}
      data-compare-align={derived ? (derived.align.reason ?? 'ok') : ''} className="app-layer-overlay-page flex flex-col bg-background">
      {header}
      <div ref={box} className="min-h-0 flex-1 overflow-auto p-4">
        <div className={wide ? 'flex gap-4' : 'flex flex-col gap-4'}>
          <aside className="flex shrink-0 flex-col gap-3" style={{ width: wide ? SIDE_W : bw }}>
            <LfChartCard title={t('lab.compare.cases.title')} sub={t('lab.compare.cases.sub')} figureId="lab-compare-cases-card">
              <CaseTable doc={doc} sel={sel} />
            </LfChartCard>
            <LfChartCard title={t('lab.compare.summary.title')} src={t('lab.compare.summary.src', { utc: doc.generatedUtc ?? '—' })}>
              <Summary doc={doc} />
            </LfChartCard>
          </aside>
          <div className="flex min-w-0 flex-1 flex-col gap-3">
            <div className="flex flex-wrap items-end gap-3">
              <h2 className="text-ed-title font-semibold" data-compare-case-title={sel.id}>{L.caseTitle(sel)}</h2>
              <Verdict ok={sel.pass} label={t(sel.pass ? 'lab.compare.casePass' : 'lab.compare.caseFail')} />
              {sel.uavs.length > 1 ? (
                <label className="ml-auto flex items-center gap-2 text-hud-sub">
                  <span className="text-muted-foreground">{t('lab.compare.uav')}</span>
                  <NativeSelect size="sm" value={uav ?? ''} data-compare-uav-select="" onChange={(e) => updateSearch({ uav: e.target.value })}>
                    {sel.uavs.map((u) => <NativeSelectOption key={u} value={u}>{u}</NativeSelectOption>)}
                  </NativeSelect>
                </label>
              ) : null}
            </div>
            {sel.failures.length ? (
              <ul className="list-disc pl-5 font-mono text-hud-sub" data-compare-failures={sel.failures.length}>
                {sel.failures.map((f, i) => <li key={i}>{sanitizeText(f, 240)}</li>)}
              </ul>
            ) : null}
            {alignOk === false ? (
              <Alert data-compare-mismatch={derived?.align.reason ?? ''}>
                <Icon icon="alert.warning" />
                <AlertTitle>{t('lab.compare.mismatch')}</AlertTitle>
                <AlertDescription>{t(`lab.compare.mismatchReason.${derived?.align.reason ?? 'time'}`)}</AlertDescription>
              </Alert>
            ) : null}
            {!sel.consistent ? (
              <Alert data-compare-inconsistent={bandsBad.map((r) => r.key).join(',')}>
                <Icon icon="alert.warning" />
                <AlertTitle>{t('lab.compare.inconsistent')}</AlertTitle>
                <AlertDescription>{t('lab.compare.inconsistentHint', { keys: bandsBad.map((r) => L.metric(r.key)).join(', ') || t('lab.compare.metrics.tolPass') })}</AlertDescription>
              </Alert>
            ) : null}
            {derived && alignOk && !derived.rms.ok ? (
              <Alert data-compare-rms-mismatch="">
                <Icon icon="alert.warning" />
                <AlertTitle>{t('lab.compare.rmsMismatch')}</AlertTitle>
                <AlertDescription>{t('lab.compare.rmsMismatchHint', { tol: POS_RMS_CHECK_M })}</AlertDescription>
              </Alert>
            ) : null}
            {tracksNote ? <span className="text-hud-sub text-muted-foreground" data-compare-tracks={tracks.missing ? 'missing' : tracks.failed.length ? 'failed' : 'loading'}>{tracksNote}</span> : null}
            <div className={twoUp ? 'grid grid-cols-2 gap-3' : 'flex flex-col gap-3'}>
              <LfChartCard title={t('lab.compare.plan.title')} sub={t('lab.compare.legend')} figureId="lab-compare-plan">
                {derived ? (
                  <TrackPlan fast={sf} real={sr} n={derived.n} idx={idx} extent={derived.extent} width={chartW} height={Math.round(chartW * 0.62)}
                    ariaLabel={t('lab.compare.plan.title')} axisLabel={t('lab.compare.plan.axis', {
                      e: `${fmtNum(derived.extent.e[0])} – ${fmtNum(derived.extent.e[1])}`, n: `${fmtNum(derived.extent.n[0])} – ${fmtNum(derived.extent.n[1])}`,
                    })} />
                ) : <PanelEmpty brand={false} title={tracksNote ?? t('lab.compare.tracks.loading')} />}
              </LfChartCard>
              <LfChartCard title={t('lab.compare.alt.title')} sub={t('lab.compare.legend')} figureId="lab-compare-alt">
                {derived ? <TimeChart series={derived.altSeries} tS={derived.tS} idx={idx} format={(v) => `${v.toFixed(1)} m`} width={chartW}
                  ariaLabel={t('lab.compare.alt.title')} figure="alt" /> : null}
              </LfChartCard>
            </div>
            <LfChartCard title={t('lab.compare.diff.title')}
              sub={rmsRow ? t('lab.compare.diff.sub', { page: fmtNum(rmsRow.page), report: fmtNum(rmsRow.report), d: fmtNum(rmsRow.absDiff) }) : t('lab.compare.diff.subNone')}
              src={t('lab.compare.diff.src')} figureId="lab-compare-diff">
              <div data-rms-check-ok={derived ? String(derived.rms.ok) : ''} data-rms-check-max={rmsMax ?? ''} data-rms-check-n={derived?.rms.rows.length ?? 0}>
                {derived ? <TimeChart series={derived.diffSeries} tS={derived.tS} idx={idx} width={Math.max(260, mainW - 24)} format={(v) => `${fmtNum(v)} m`}
                  refLine={rmsReport !== null ? { v: rmsReport, label: t('lab.compare.diff.ref', { v: fmtNum(rmsReport) }) } : null}
                  ariaLabel={t('lab.compare.diff.title')} figure="diff" /> : null}
              </div>
            </LfChartCard>
            <LfChartCard title={t('lab.compare.bands.title')} sub={t('lab.compare.bands.sub')} figureId="lab-compare-bands">
              <DiffBands rows={sel.rows} label={L.metric} ariaLabel={t('lab.compare.bands.title')} width={Math.max(260, Math.min(mainW - 24, 900))}
                value={(r, v) => `${withUnit(fmtNum(v), metricUnit(r.key))} / ${fmtNum(r.tol)}`} />
            </LfChartCard>
            <LfChartCard title={t('lab.compare.iface.title')} sub={t('lab.compare.iface.sub')} figureId="lab-compare-iface-card">
              <InterfaceTable tb={doc.tiers[1]} c={sel} />
            </LfChartCard>
            <LfChartCard title={t('lab.compare.metrics.title')} sub={t('lab.compare.metrics.sub')} figureId="lab-compare-metrics-card">
              <MetricTable tb={doc.tiers[1]} c={sel} />
            </LfChartCard>
            <div className={twoUp ? 'grid grid-cols-2 gap-3' : 'flex flex-col gap-3'}>
              <LfChartCard title={t('lab.compare.sampling.title')} sub={t('lab.compare.sampling.sub')} figureId="lab-compare-sampling">
                <CheckTable rows={samplingRows(doc.sampling)} title={t('lab.compare.sampling.title')} figure="lab-compare-sampling-table" />
              </LfChartCard>
              <LfChartCard title={t('lab.compare.m25.title')} sub={t('lab.compare.m25.sub', { world: doc.m25?.world ?? '—', renderer: doc.m25?.renderer ?? '—' })} figureId="lab-compare-m25">
                <CheckTable rows={m25Rows(doc.m25)} title={t('lab.compare.m25.title')} figure="lab-compare-m25-table" />
              </LfChartCard>
            </div>
          </div>
        </div>
      </div>
      <footer className="flex flex-wrap items-center gap-3 border-t px-4 py-2" data-figure="lab-compare-timeline" data-compare-playhead={idx}>
        <Button size="sm" variant="outline" disabled={!derived} aria-label={ph.playing ? t('lab.compare.pause') : t('lab.compare.play')} onClick={() => (ph.playing ? ph.pause() : ph.play())}>
          <Icon icon={ph.playing ? 'tl.pause' : 'tl.play'} />
        </Button>
        <Button size="sm" variant="ghost" disabled={!derived} aria-label={t('lab.compare.stepBack')} onClick={() => ph.seek(ph.tS - STEP_S)}><Icon icon="tl.stepback" /></Button>
        <Button size="sm" variant="ghost" disabled={!derived} aria-label={t('lab.compare.stepFwd')} onClick={() => ph.seek(ph.tS + STEP_S)}><Icon icon="tl.stepfwd" /></Button>
        <Slider className="min-w-48 flex-1" min={0} max={Math.max(ph.durationS, 0.1)} step={0.004} value={[ph.tS]} aria-label={t('lab.compare.timeline')}
          disabled={!derived} onValueChange={(v) => ph.seek(Array.isArray(v) ? Number(v[0]) : Number(v))} />
        <span className="font-mono text-hud-sub tabular-nums" data-compare-clock={ph.tS.toFixed(3)}>{`${clock(ph.tS)} / ${clock(ph.durationS)}`}</span>
        <span className="font-mono text-hud-sub tabular-nums text-muted-foreground" data-compare-readout={dNow ?? ''}>
          {pF && pR ? t('lab.compare.readout', { fast: xyz(pF), real: xyz(pR), d: fmtNum(dNow) }) : ''}
        </span>
      </footer>
    </section>
  )
}
