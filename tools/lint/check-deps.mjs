#!/usr/bin/env node
// Dependency governance (AWR-11 §7.4, TECH-FR-003/004; AWR-18 §13.1).
//   deps/unlisted    BOM mode: a direct dependency in a manifest or lock is not registered in the BOM.
//                    Fallback mode: a bare import in apps/web/{src,tests,perf,dev} is not declared in apps/web or root package.json.
//   deps/version     declared npm versions must be exact; lock versions must equal the declaration (and the BOM when present).
//   deps/denied      TECH-FR-003 rejected list (AWR-11 §5 X-numbers), npm and PyPI.
//   deps/stale       BOM runtime entries whose last commit is older than 12 months and that have no fallback.
//   imports/boundary TS import boundaries (AWR-11 §3.4, AWR-03 §4.2): lucide-react, @base-ui/react outside ui/components/ui,
//                    @msgpack/msgpack outside net/**, camera-controls outside engine/camera, dev oracles only in dev/oracles,
//                    drei AdaptiveDpr/PerformanceMonitor/CameraControls, engine/** and net/** free of react, zustand, R3F and UI paths,
//                    rt.worker free of react, three and zustand.
// BOM (tools/ci/bom.json, AWR-11 §7.3: {schemaVersion, generatedFrom, collectedAt, items[]}; FX-GW): every field is validated
// (missing required field, enum out of range or non-exact version -> deps/version); npm manifest dependencies and the PyPI direct
// dependencies (pyproject.toml dependencies and locked extras, requirements.in) must be registered (deps/unlisted). Without the
// file the tool runs the fallback checks and says so; the legacy {packages: [...]} shape is still read.
// Usage: node tools/lint/check-deps.mjs [--bom-only]
import { existsSync, readFileSync } from 'node:fs'
import { builtinModules } from 'node:module'
import { join } from 'node:path'
import { ROOT, Reporter, WEB_SRC, isCode, isMain, lineCol, readText, run, stripComments, walk } from './_common.mjs'

const BOM = 'tools/ci/bom.json'
const MANIFESTS = ['package.json', 'apps/web/package.json', 'packages/contracts/package.json']
const EXACT = /^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/
const DENIED_NPM = {
  'lucide-react': 'X09', recharts: 'X05', echarts: 'X06', 'chart.js': 'X06', sonner: 'X07', motion: 'X08', 'framer-motion': 'X08',
  '@radix-ui/*': 'X10', potree: 'X15', 'three-loader': 'X15', jotai: 'X39', valtio: 'X39', comlink: 'X39', '@tanstack/pacer': 'X39', 'r3f-perf': 'X39',
  next: 'X38', eslint: 'X37', 'typescript-eslint': 'X37', '@typescript-eslint/*': 'X37', msgpackr: 'X26', cesium: 'X01', '@zenoh/zenoh-ts': 'X23',
  '@foxglove/ws-protocol': 'X22', pnpm: 'X46', 'takram-three-atmosphere': 'X43', '@takram/three-atmosphere': 'X43',
}
const DENIED_PY = { redis: 'X30', pyzmq: 'X30', supervisor: 'X29', aioquic: 'X24', pywebtransport: 'X24', airsim: 'X34', rotorpy: 'X33', roslibpy: 'X21' }
const deniedNpm = (name) => DENIED_NPM[name] ?? Object.entries(DENIED_NPM).find(([k]) => k.endsWith('/*') && name.startsWith(k.slice(0, -1)))?.[1]

const pkgName = (spec) => (spec.startsWith('@') ? spec.split('/').slice(0, 2).join('/') : spec.split('/')[0])
const BUILTINS = new Set([...builtinModules, ...builtinModules.map((m) => `node:${m}`)])

function readJson(rel) {
  try {
    return JSON.parse(readFileSync(join(ROOT, rel), 'utf8'))
  } catch {
    return null
  }
}

function lineOf(rel, needle) {
  const t = readText(rel) ?? ''
  const i = t.indexOf(needle)
  return i < 0 ? [1, 1] : lineCol(t, i)
}

export function checkManifests(R, bom) {
  const lock = readJson('package-lock.json')
  const declared = new Map()
  for (const m of MANIFESTS) {
    const pj = readJson(m)
    if (!pj) continue
    for (const sect of ['dependencies', 'devDependencies', 'optionalDependencies', 'peerDependencies']) {
      for (const [name, ver] of Object.entries(pj[sect] ?? {})) {
        declared.set(name, { ver, manifest: m })
        const [l, c] = lineOf(m, `"${name}"`)
        const x = deniedNpm(name)
        if (x) R.add(m, l, c, 'deps/denied', `${name} is rejected (AWR-11 §5 ${x})`)
        if (sect !== 'peerDependencies' && !EXACT.test(String(ver)) && !String(ver).startsWith('workspace:') && !(name.startsWith('@awr/'))) {
          R.add(m, l, c, 'deps/version', `${name}@${ver}: versions must be exact (no ranges)`)
        }
        const locked = lock?.packages?.[`node_modules/${name}`]
        if (locked && !locked.link && EXACT.test(String(ver)) && locked.version !== ver) {
          R.add(m, l, c, 'deps/version', `${name}: declared ${ver}, package-lock.json has ${locked.version}`)
        }
        if (bom) {
          const e = bom.get(`npm:${name}`)
          if (!e && !name.startsWith('@awr/')) R.add(m, l, c, 'deps/unlisted', `${name} (${m}) is not registered in ${BOM}`)
          else if (e && e.version && e.version !== ver) R.add(m, l, c, 'deps/version', `${name}: expected ${e.version} (BOM), got ${ver}`)
        }
      }
    }
  }
  // transitive packages are reported, not failed: X-numbers reject direct use (cmdk pulls @radix-ui/react-dialog,
  // react-scan pulls eslint); direct imports are caught by imports/boundary and deps/unlisted below
  const transitive = new Set()
  for (const k of Object.keys(lock?.packages ?? {})) {
    const name = k.replace(/^.*node_modules\//, '')
    if (k.includes('node_modules/') && deniedNpm(name) && !declared.has(name)) transitive.add(name)
  }
  if (transitive.size) R.note(`transitive packages on the rejected list (not imported directly): ${[...transitive].sort().join(', ')}`)
  return declared
}

const pyName = (n) => n.toLowerCase().replace(/[_.]/g, '-')
// extras that are not part of the D1 lock (pyproject.toml comments; AWR-11 §3.6)
const PY_UNLOCKED_EXTRAS = new Set(['geo-worker', 'px4', 'life', 'life-ext'])

/** PyPI direct dependencies: [project] dependencies, the locked optional-dependency groups and requirements.in pins. */
export function pythonDirectDeps(pyproject, requirementsIn) {
  const out = new Map()
  const addSpec = (spec, where, line) => {
    const m = /^\s*([A-Za-z0-9_.-]+)\s*==\s*([^\s;]+)/.exec(spec)
    if (m) out.set(pyName(m[1]), { ver: m[2], where, line })
  }
  if (pyproject) {
    const lines = pyproject.split('\n')
    let sect = '', inDeps = false, group = null
    lines.forEach((raw, i) => {
      const line = raw.replace(/#.*$/, '')
      const h = /^\s*\[([^\]]+)\]\s*$/.exec(line)
      if (h) {
        sect = h[1]
        inDeps = false
        return
      }
      if (sect === 'project' && /^\s*dependencies\s*=\s*\[/.test(line)) inDeps = true
      if (sect === 'project.optional-dependencies') {
        const g = /^\s*([A-Za-z0-9_-]+)\s*=\s*\[/.exec(line)
        if (g) {
          group = g[1]
          inDeps = !PY_UNLOCKED_EXTRAS.has(group)
        }
      }
      if (inDeps) for (const q of line.matchAll(/"([^"]+)"/g)) addSpec(q[1], 'pyproject.toml', i + 1)
      if (inDeps && line.includes(']')) inDeps = false
    })
  }
  if (requirementsIn) {
    requirementsIn.split('\n').forEach((raw, i) => {
      const line = raw.replace(/#.*$/, '')
      if (line.trim()) addSpec(line, 'requirements.in', i + 1)
    })
  }
  return out
}

export function checkPython(R, bom) {
  const t = readText('requirements.lock')
  if (t === null) {
    R.note('requirements.lock not found; PyPI checks skipped')
    return
  }
  const locked = new Map()
  for (const m of t.matchAll(/^([A-Za-z0-9_.-]+)==([^\s;\\]+)/gm)) {
    const name = pyName(m[1])
    const [l, c] = lineCol(t, m.index)
    locked.set(name, m[2])
    if (DENIED_PY[name]) R.add('requirements.lock', l, c, 'deps/denied', `${name} is rejected (AWR-11 §5 ${DENIED_PY[name]})`)
    if (bom) {
      const e = bom.get(`pypi:${name}`)
      if (e && e.version && e.version !== m[2]) R.add('requirements.lock', l, c, 'deps/version', `${name}: expected ${e.version} (BOM), locked ${m[2]}`)
    }
  }
  if (!bom) return
  for (const [name, d] of pythonDirectDeps(readText('pyproject.toml'), readText('requirements.in'))) {
    if (!bom.get(`pypi:${name}`)) R.add(d.where, d.line, 1, 'deps/unlisted', `${name} (${d.where}) is not registered in ${BOM}`)
    if (locked.size && !locked.has(name)) R.add(d.where, d.line, 1, 'deps/version', `${name}==${d.ver} is declared but not in requirements.lock`)
  }
}

// ---------------------------------------------------------------- BOM schema (AWR-11 §7.3)
const BOM_ENUMS = {
  ecosystem: ['npm', 'pypi', 'docker', 'vendor', 'self'],
  scope: ['runtime', 'dev', 'tool'],
  d1: ['core', 'ext', 'stub', 'no'],
  new2026: ['yes', 'capability', 'no'],
  status: ['PROPOSED', 'REVIEW', 'SPIKE', 'LOCKED', 'WATCH', 'UPGRADING', 'REJECTED', 'RETIRED'],
}
const DATE = /^\d{4}-\d{2}-\d{2}$/
const PEP440_EXACT = /^\d+(?:\.\d+)*(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?$/
const BOM_KEYS = new Set(['id', 'upstream', 'ecosystem', 'name', 'version', 'scope', 'd1', 'targetVersion', 'adapterBoundary', 'fallback',
  'stars', 'lastCommit', 'new2026', 'deviation', 'status', 'collectedAt'])

const exactVersion = (eco, v) =>
  typeof v === 'string' && (eco === 'npm' ? EXACT.test(v) : eco === 'pypi' ? PEP440_EXACT.test(v) : eco === 'vendor' ? /^[0-9a-f]{40}$/.test(v)
    : eco === 'self' ? v === '-' : /^[\w][\w.-]*$/.test(v) && v !== 'latest')

/** Validate the whole document; every finding is `deps/version` (AWR-11 §7.3: missing field, enum or non-exact version). */
export function validateBom(R, doc) {
  const bad = (msg) => R.add(BOM, 1, 1, 'deps/version', msg)
  if (doc.schemaVersion !== 'awr.bom.v1') bad(`schemaVersion must be "awr.bom.v1", got ${JSON.stringify(doc.schemaVersion)}`)
  if (typeof doc.generatedFrom !== 'string' || !doc.generatedFrom) bad('generatedFrom is required')
  if (typeof doc.collectedAt !== 'string' || !DATE.test(doc.collectedAt)) bad('collectedAt must be YYYY-MM-DD')
  const items = doc.items ?? []
  const seen = new Set()
  let prev = null
  for (const [i, p] of items.entries()) {
    const who = `${p?.name ?? '?'} (items[${i}])`
    if (!p || typeof p !== 'object') {
      bad(`items[${i}] is not an object`)
      continue
    }
    for (const k of Object.keys(p)) if (!BOM_KEYS.has(k)) bad(`${who}: unknown field ${k}`)
    for (const k of ['id', 'upstream', 'ecosystem', 'name', 'version', 'scope', 'd1', 'new2026', 'status']) {
      if (typeof p[k] !== 'string' || !p[k]) bad(`${who}: missing field ${k}`)
    }
    if (typeof p.id === 'string' && !/^T[0-9]{2}$/.test(p.id)) bad(`${who}: id ${p.id} does not match ^T[0-9]{2}$`)
    for (const [k, vals] of Object.entries(BOM_ENUMS)) if (typeof p[k] === 'string' && !vals.includes(p[k])) bad(`${who}: ${k} ${p[k]} not in ${vals.join('|')}`)
    if (typeof p.upstream === 'string' && p.upstream !== '-' && !/^[\w.-]+\/[\w.-]+$/.test(p.upstream)) bad(`${who}: upstream must be owner/repo or "-"`)
    if (p.ecosystem && !exactVersion(p.ecosystem, p.version)) bad(`${who}: version ${JSON.stringify(p.version)} is not exact for ${p.ecosystem}`)
    if (p.d1 === 'no' && typeof p.targetVersion !== 'string') bad(`${who}: d1 = no needs targetVersion`)
    if (p.scope === 'runtime') {
      if (!Array.isArray(p.adapterBoundary) || !p.adapterBoundary.length || p.adapterBoundary.some((g) => typeof g !== 'string' || !g)) {
        bad(`${who}: runtime entries need a non-empty adapterBoundary`)
      }
      if (typeof p.fallback !== 'string' || !p.fallback) bad(`${who}: runtime entries need a fallback`)
    }
    // stars and lastCommit are required; a self entry with no port source (upstream "-") records null for both
    const noSource = p.ecosystem === 'self' && p.upstream === '-'
    if (!(noSource && p.stars === null) && !(Number.isInteger(p.stars) && p.stars >= 0)) bad(`${who}: stars must be an integer >= 0`)
    if (!(noSource && p.lastCommit === null) && !(typeof p.lastCommit === 'string' && DATE.test(p.lastCommit))) bad(`${who}: lastCommit must be YYYY-MM-DD`)
    if (p.collectedAt !== undefined && !(typeof p.collectedAt === 'string' && DATE.test(p.collectedAt))) bad(`${who}: collectedAt must be YYYY-MM-DD`)
    const at = p.collectedAt ?? doc.collectedAt
    if (typeof p.lastCommit === 'string' && typeof at === 'string' && p.lastCommit > at) bad(`${who}: lastCommit ${p.lastCommit} is after its collection date ${at}`)
    for (const k of ['deviation', 'targetVersion', 'fallback']) if (p[k] !== undefined && (typeof p[k] !== 'string' || !p[k])) bad(`${who}: ${k} must be a non-empty string`)
    const key = `${p.ecosystem}:${p.ecosystem === 'pypi' ? pyName(String(p.name)) : p.name}`
    if (seen.has(key)) bad(`${who}: duplicate entry ${key}`)
    seen.add(key)
    const order = [String(p.id), String(p.name)]
    if (prev && (order[0] < prev[0] || (order[0] === prev[0] && order[1] < prev[1]))) bad(`${who}: items must be sorted by id, then name`)
    prev = order
  }
}

const RULES = [
  { test: (s) => s === 'lucide-react' || s.startsWith('lucide-react/'), where: () => true, msg: 'lucide-react is forbidden (ADR-030)' },
  { test: (s) => s === '@base-ui/react' || s.startsWith('@base-ui/react/'), where: (f) => !f.startsWith(`${WEB_SRC}/ui/components/ui/`), msg: '@base-ui/react only in ui/components/ui (AWR-03 §4.2)' },
  { test: (s) => s === '@msgpack/msgpack', where: (f) => f.startsWith(`${WEB_SRC}/`) && !f.startsWith(`${WEB_SRC}/net/`), msg: '@msgpack/msgpack only in net/** (AWR-11 §3.4)' },
  { test: (s) => s === 'camera-controls', where: (f) => f.startsWith(`${WEB_SRC}/`) && !f.startsWith(`${WEB_SRC}/engine/camera/`), msg: 'camera-controls only in engine/camera (AWR-11 §3.4)' },
  { test: (s) => s === 'potree-core' || s.startsWith('@voxelkloud/'), where: (f) => !f.startsWith('apps/web/dev/oracles/'), msg: 'dev oracles only in apps/web/dev/oracles (AWR-11 §3.4)' },
  { test: (s) => /^(react|react-dom|zustand|@react-three\/[\w-]+)(\/|$)/.test(s) || /(^|\/)(ui|viewport|stores)\//.test(s) || /^@\/(ui|viewport|stores)(\/|$)/.test(s),
    where: (f) => f.startsWith(`${WEB_SRC}/engine/`) || f.startsWith(`${WEB_SRC}/net/`), msg: 'engine/** and net/** must not import React, zustand, R3F, ui, viewport or stores (AWR-03 §4.2)' },
  { test: (s) => /^(react|react-dom|three|zustand)(\/|$)/.test(s), where: (f) => /\/net\/rt\/[^/]*worker[^/]*\.ts$/.test(f), msg: 'rt.worker must not import react, three or zustand (AWR-11 §3.4)' },
]

export function checkImports(f, text, R, declared) {
  const src = stripComments(text)
  const re = /(?:^|[;\n])\s*(?:import|export)\s+(?:type\s+)?(?:[\s\S]*?\sfrom\s+)?['"]([^'"]+)['"]|\bimport\(\s*['"]([^'"]+)['"]\s*\)|\brequire\(\s*['"]([^'"]+)['"]\s*\)/g
  for (const m of src.matchAll(re)) {
    const spec = m[1] ?? m[2] ?? m[3]
    const idx = m.index + m[0].indexOf(spec)
    const [l, c] = lineCol(src, idx)
    for (const r of RULES) if (r.test(spec) && r.where(f)) R.add(f, l, c, 'imports/boundary', `${spec}: ${r.msg}`)
    const denied = !spec.startsWith('.') && deniedNpm(pkgName(spec))
    if (denied && pkgName(spec) !== 'lucide-react') R.add(f, l, c, 'deps/denied', `${pkgName(spec)} is rejected (AWR-11 §5 ${denied})`)
    const stmt = m[0]
    if (spec === '@react-three/drei') {
      for (const bad of ['AdaptiveDpr', 'PerformanceMonitor', 'CameraControls']) {
        if (new RegExp(`\\b${bad}\\b`).test(stmt)) R.add(f, l, c, 'imports/boundary', `drei ${bad} is excluded from the whitelist (AWR-11 X47)`)
      }
    }
    if (declared && !spec.startsWith('.') && !spec.startsWith('/') && !spec.startsWith('@/') && !BUILTINS.has(spec) && !spec.startsWith('virtual:') &&
        !spec.startsWith('~') && !spec.includes('?')) {
      const name = pkgName(spec)
      if (!declared.has(name) && !BUILTINS.has(name)) R.add(f, l, c, 'deps/unlisted', `${name} is imported but not declared in apps/web or root package.json`)
    }
  }
}

function loadBom(R) {
  if (!existsSync(join(ROOT, BOM))) return null
  const d = readJson(BOM)
  const list = Array.isArray(d?.items) ? d.items : Array.isArray(d?.packages) ? d.packages : null
  if (!list) throw new Error(`${BOM} is not a valid BOM (expected {schemaVersion, items: [...]}, AWR-11 §7.3)`)
  const map = new Map()
  for (const p of list) {
    if (!p?.name || !p?.ecosystem) {
      R.add(BOM, 1, 1, 'deps/unlisted', `BOM entry without name or ecosystem: ${JSON.stringify(p).slice(0, 80)}`)
      continue
    }
    map.set(`${p.ecosystem}:${p.ecosystem === 'pypi' ? pyName(p.name) : p.name}`, p)
  }
  return { map, doc: d, list, legacy: !Array.isArray(d.items) }
}

if (isMain(import.meta.url)) {
  await run('check-deps', async () => {
    const R = new Reporter('check-deps')
    const bomOnly = process.argv.includes('--bom-only')
    const bom = loadBom(R)
    if (bomOnly) {
      if (!bom) throw new Error(`${BOM} does not exist; --bom-only needs the BOM`)
      if (bom.legacy) R.add(BOM, 1, 1, 'deps/version', 'legacy {packages: [...]} shape; AWR-11 §7.3 requires {schemaVersion, generatedFrom, collectedAt, items}')
      else validateBom(R, bom.doc)
      R.finish({ bom: true, items: bom.list.length })
      return
    }
    if (!bom) R.note(`${BOM} not present: BOM checks (deps/unlisted against the BOM, deps/stale) skipped; manifest, lock, denied and boundary checks ran`)
    else if (!bom.legacy) validateBom(R, bom.doc)
    const declared = checkManifests(R, bom?.map ?? null)
    checkPython(R, bom?.map ?? null)
    if (bom) {
      const year = 365 * 24 * 3600 * 1000
      for (const p of bom.list) {
        if (p.scope === 'runtime' && p.lastCommit && !p.fallback && Date.now() - Date.parse(p.lastCommit) > year) {
          R.add(BOM, 1, 1, 'deps/stale', `${p.name}: last commit ${p.lastCommit} is older than 12 months and has no fallback`)
        }
      }
    }
    const files = ['apps/web/src', 'apps/web/tests', 'apps/web/perf', 'apps/web/dev'].flatMap((d) => walk(d, isCode))
    for (const f of files) {
      const t = readText(f)
      if (t !== null) checkImports(f, t, R, declared)
    }
    R.finish({ files: files.length, bom: Boolean(bom) })
  })
}
