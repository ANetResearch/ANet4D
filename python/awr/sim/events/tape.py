"""Canonical Event Tape（AWR-05 §5.2.3，ADR-134；M23 §6.7、§6.8）：`tape.bin` 写出、校验与读取插值。

记录：每帧每演员 22 B 小端 `x_mm i32 | y_mm i32 | z_mm i32 | yaw_cdeg i16 | vx_cmps i16 | vy_cmps i16 | vz_cmps i16 | state u8 |
container u8`；规范帧数据 F = 对 tick k、演员 idx 依次拼接（tick 主序）。

容器：魔数 "D4DTAPE\\x01" | u32 header_len | 规范头 JCS 字节 | u32 n_blocks | n_blocks × 40 B 块索引（u64 off_S、u32 comp_S、
u32 raw_S、u64 off_B、u32 comp_B、u32 raw_B、u32 tick0、u16 n_ticks、u16 0）| 块数据。块 b 覆盖 tick [100b, 100b + 100)，按演员拆为
脚本演员子块 S（idx < n_s）与背景子块 B；子块原文按字段 → 演员 → tick 排列，每个演员在块内首值为绝对值、其余为与前一 tick 的差
（按字段位宽补码回绕；yaw 先折叠到 [−18000, 18000)），zstd 级别 19、带内容校验、单帧。压缩级别与拆分方式不影响 F 与内容哈希；
各子块独立成帧，按子块并行压缩（线程数不影响压缩字节）。

哈希：content = SHA-256(header_bytes ‖ F)、body = SHA-256(F)、header = SHA-256(header_bytes)、file = SHA-256(tape.bin)。校验方对存储的
规范头原始字节求哈希，不重新序列化。脚本演员列哈希 `scripted_sha256` = SHA-256(JCS(脚本演员表) ‖ F_S)，F_S 为 idx < n_s 的 tick 主序记录。
"""

from __future__ import annotations

import hashlib
import os
import struct
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from awr.sim.events.errors import EventError
from awr.sim.events.jcs import jcs

__all__ = ["BLOCK_TICKS", "MAGIC", "MAGIC_V2", "NO_CONTAINER", "NO_CONTAINER_V2", "RECORD", "RECORD_V2", "STATE_VISIBLE", "TICK_NS", "ActorStates", "EnvSample", "TapeReader", "fold_yaw", "manifest", "open_tape",
           "record_spec", "write_tape"]

MAGIC = b"D4DTAPE\x01"
MAGIC_V2 = b"D4DTAPE\x02"
TICK_NS = 100_000_000
BLOCK_TICKS = 100
ZSTD_LEVEL = 19
PARALLEL_MIN_BYTES = 1 << 20                  # 子块原文合计 ≥ 1 MiB 时按子块并行压缩
FIELDS = (("x_mm", "<i4"), ("y_mm", "<i4"), ("z_mm", "<i4"), ("yaw_cdeg", "<i2"), ("vx_cmps", "<i2"), ("vy_cmps", "<i2"),
          ("vz_cmps", "<i2"), ("state", "u1"), ("container", "u1"))
RECORD = np.dtype([(n, t) for n, t in FIELDS])
assert RECORD.itemsize == 22
FIELDS_V2 = (*FIELDS[:-1], ("container", "<u4"), ("pitch_cdeg", "<i2"), ("roll_cdeg", "<i2"))
RECORD_V2 = np.dtype(list(FIELDS_V2))
NO_CONTAINER_V2 = 0xFFFFFFFF
STATES = {0: "absent", 1: "moving", 2: "waiting", 3: "in_vehicle", 4: "inside", 5: "gone"}
STATE_VISIBLE = frozenset({1, 2})
NO_CONTAINER = 255
_IDX = struct.Struct("<QIIQIIIHH")
assert _IDX.size == 40


def fold_yaw(cdeg: np.ndarray) -> np.ndarray:
    """centi-degree 折叠到 [−18000, 18000)。"""
    return ((np.asarray(cdeg, np.int64) + 18000) % 36000 - 18000).astype(np.int16)


def record_spec(version: int = 1) -> dict:
    if version not in (1, 2):
        raise ValueError("unsupported tape version")
    fields = FIELDS if version == 1 else FIELDS_V2
    types = {"<i4":"i32", "<i2":"i16", "u1":"u8", "<u4":"u32"}
    return {"bytes": 22 if version == 1 else 29, "endian":"le", "fields":[[n,types[t]] for n,t in fields],
            "states":{str(k):v for k,v in STATES.items()}, "frame":"world-enu"}



def _encode_sub(fr: np.ndarray) -> bytes:
    """fr (n_t, n_a) RECORD → 子块原文：字段 → 演员 → tick；块内首值绝对，其余差分（位宽内回绕）。"""
    parts = []
    for name in fr.dtype.names:
        t = fr.dtype.fields[name][0]
        a = np.ascontiguousarray(fr[name].T)                  # (n_a, n_t)
        d = a.copy()
        if a.shape[1] > 1:
            d[:, 1:] = a[:, 1:] - a[:, :-1]                    # numpy 整型减法即按位宽补码回绕
        parts.append(d.astype(t).tobytes())
    return b"".join(parts)


def _decode_sub(raw: bytes, n_t: int, n_a: int, dtype=RECORD) -> np.ndarray:
    out = np.zeros((n_t, n_a), dtype)
    p = 0
    for name in dtype.names:
        dt = dtype.fields[name][0]
        ln = n_t * n_a * dt.itemsize
        d = np.frombuffer(raw, dt, count=n_t * n_a, offset=p).reshape(n_a, n_t)
        p += ln
        with np.errstate(over="ignore"):
            a = np.cumsum(d, axis=1, dtype=dt)                 # 同位宽累加，与写出时的回绕差分互逆
        out[name] = a.T
    if p != len(raw):
        raise EventError(620, "tape sub-block length mismatch")
    return out


def _fold(frames: np.ndarray, dtype=RECORD) -> np.ndarray:
    f = np.array(frames, dtype, copy=True)
    f["yaw_cdeg"] = fold_yaw(f["yaw_cdeg"])
    return f


def _workers() -> int:
    """压缩线程数：`AWR_TAPE_THREADS`（≥ 1）或可用 CPU 数，上限 8。"""
    env = os.environ.get("AWR_TAPE_THREADS", "").strip()
    if env.isdigit() and int(env) >= 1:
        return int(env)
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):  # pragma: no cover - 非 Linux
        n = os.cpu_count() or 1
    return max(1, min(8, n))


def write_tape(path: Path, header: dict, frames: np.ndarray, *, n_scripted: int, level: int = ZSTD_LEVEL) -> dict:
    """frames (n_ticks, n_actors) RECORD。返回清单的哈希部分 {content_sha256, body_sha256, header_sha256, file_sha256, bytes}。"""
    import zstandard

    version = {"drone4d.tape.v1": 1, "drone4d.tape.v2": 2}.get(header.get("schema"))
    if version is None:
        raise EventError(611, "unsupported tape schema")
    dtype, magic = (RECORD, MAGIC) if version == 1 else (RECORD_V2, MAGIC_V2)
    if header.get("record") != record_spec(version):
        raise EventError(611, "record layout does not match tape schema")
    sentinel = NO_CONTAINER if version == 1 else NO_CONTAINER_V2
    n_t, n_a = frames.shape
    if n_a > sentinel or not 0 <= n_scripted <= n_a:
        raise EventError(611, "actor count exceeds tape format capacity; use v2")
    containers = frames["container"].astype(np.int64)
    if ((containers != sentinel) & ((containers < 0) | (containers >= n_a))).any():
        raise EventError(611, "container index out of range")
    if version == 2:
        riding = frames["state"] == 3
        if (riding & (containers == sentinel)).any():
            raise EventError(611, "in_vehicle requires a container")
        if ((~riding) & (containers != sentinel)).any():
            raise EventError(611, "container requires in_vehicle state")
        vehicle = np.array([a["class"] == "vehicle" for a in header["actors"]])
        if riding.any() and not vehicle[containers[riding]].all():
            raise EventError(611, "in_vehicle container must be a vehicle")
    frames = _fold(frames, dtype)
    if header.get("n_ticks") != n_t or len(header.get("actors", [])) != n_a:
        raise EventError(611, "header n_ticks / actors do not match the frame array")
    hb = jcs(header)
    nb = (n_t + BLOCK_TICKS - 1) // BLOCK_TICKS
    raws = []
    for b in range(nb):
        t0, t1 = b * BLOCK_TICKS, min((b + 1) * BLOCK_TICKS, n_t)
        raws += [_encode_sub(frames[t0:t1, :n_scripted]), _encode_sub(frames[t0:t1, n_scripted:]) if n_a > n_scripted else b""]

    def _z(raw: bytes) -> bytes:
        # 每个子块独立成帧、参数相同，压缩字节与执行次序和线程数无关；zstandard 在压缩期间释放 GIL
        return zstandard.ZstdCompressor(level=level, write_checksum=True, write_content_size=True).compress(raw) if raw else b""

    workers = min(len(raws), _workers())
    if workers > 1 and sum(map(len, raws)) >= PARALLEL_MIN_BYTES:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as ex:
            comp = list(ex.map(_z, raws))
    else:
        comp = [_z(r) for r in raws]
    blocks, index = [], []
    off = len(MAGIC) + 4 + len(hb) + 4 + nb * _IDX.size
    for b in range(nb):
        t0, t1 = b * BLOCK_TICKS, min((b + 1) * BLOCK_TICKS, n_t)
        rs, rb, cs, cb = raws[2 * b], raws[2 * b + 1], comp[2 * b], comp[2 * b + 1]
        index.append(_IDX.pack(off, len(cs), len(rs), off + len(cs), len(cb), len(rb), t0, t1 - t0, 0))
        blocks += [cs, cb]
        off += len(cs) + len(cb)
    body = frames.tobytes()                                     # 结构化数组按记录紧排即 tick 主序的规范帧数据
    data = b"".join([magic, struct.pack("<I", len(hb)), hb, struct.pack("<I", nb), *index, *blocks])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)
    return {"content_sha256": hashlib.sha256(hb + body).hexdigest(), "body_sha256": hashlib.sha256(body).hexdigest(),
            "header_sha256": hashlib.sha256(hb).hexdigest(), "file_sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


@dataclass
class ActorStates:
    """SoA：pos_enu_m (n, 3) float64、yaw_rad (n,)、vel_enu_mps (n, 3)、state (n,) u8、container (n,) u8。"""
    ids: np.ndarray
    pos_enu_m: np.ndarray
    yaw_rad: np.ndarray
    vel_enu_mps: np.ndarray
    state: np.ndarray
    container: np.ndarray
    pitch_rad: np.ndarray | None = None
    roll_rad: np.ndarray | None = None


@dataclass(frozen=True)
class EnvSample:
    """M25 §6.3.1 `env_at`：风为参考高度风速（来向，度，顺时针自北），`wind_enu_mps` 为去向矢量；`scalars` 为 M07 的 21 维标量。"""
    t_ns: int
    wind_enu_mps: np.ndarray
    wind_speed_mps: float
    wind_dir_from_deg: float
    gust_sigma_mps: float
    rain_mm_h: float
    visibility_m: float
    fog_top_agl_m: float
    cloud: float
    scalars: np.ndarray


@dataclass(frozen=True)
class _Block:
    off_s: int
    comp_s: int
    raw_s: int
    off_b: int
    comp_b: int
    raw_b: int
    tick0: int
    n_ticks: int


class TapeReader:
    """只读访问 `tape.bin`：懒解码（按需只解 S 或 B 子块，块缓存 LRU ≤ 3），`actor_states` 按 M23 §6.8 插值。"""

    def __init__(self, path: Path) -> None:
        import json

        self.path = Path(path)
        self._buf = self.path.read_bytes()
        b = self._buf
        if b[:8] not in (MAGIC, MAGIC_V2):
            raise EventError(620, "not a drone4d tape (magic)")
        (hl,) = struct.unpack_from("<I", b, 8)
        self.header_bytes = b[12:12 + hl]
        self.header = json.loads(self.header_bytes)
        self.version = 1 if b[:8] == MAGIC else 2
        self.record = RECORD if self.version == 1 else RECORD_V2
        self.no_container = NO_CONTAINER if self.version == 1 else NO_CONTAINER_V2
        if self.header.get("schema") != f"drone4d.tape.v{self.version}":
            raise EventError(620, f"unsupported tape schema {self.header.get('schema')!r}")
        if self.header.get("record") != record_spec(self.version):
            raise EventError(620, "record layout does not match schema")
        p = 12 + hl
        (nb,) = struct.unpack_from("<I", b, p)
        p += 4
        self.blocks = [_Block(*_IDX.unpack_from(b, p + i * _IDX.size)[:8]) for i in range(nb)]
        self.n_ticks = int(self.header["n_ticks"])
        self.actors = self.header["actors"]
        self.n_actors = len(self.actors)
        self.n_scripted = sum(1 for a in self.actors if a["scripted"])
        self._cache: OrderedDict[tuple[int, str], np.ndarray] = OrderedDict()
        self._env: list | None = None

    # ---------------------------------------------------------------- 解码与校验
    def _sub(self, bi: int, which: str) -> np.ndarray:
        import zstandard

        key = (bi, which)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        blk = self.blocks[bi]
        n_a = self.n_scripted if which == "S" else self.n_actors - self.n_scripted
        off, comp, raw = (blk.off_s, blk.comp_s, blk.raw_s) if which == "S" else (blk.off_b, blk.comp_b, blk.raw_b)
        if n_a == 0:
            arr = np.zeros((blk.n_ticks, 0), self.record)
        else:
            try:
                data = zstandard.ZstdDecompressor().decompress(self._buf[off:off + comp])
            except zstandard.ZstdError as e:
                raise EventError(614, f"block {bi} {which}: {e}") from None
            if len(data) != raw:
                raise EventError(614, f"block {bi} {which}: raw length {len(data)} != {raw}")
            arr = _decode_sub(data, blk.n_ticks, n_a, self.record)
        self._cache[key] = arr
        while len(self._cache) > 6:                              # 3 个块 × S、B 两个子块
            self._cache.popitem(last=False)
        return arr

    def block_frames(self, bi: int, *, background: bool = True) -> np.ndarray:
        s = self._sub(bi, "S")
        return np.concatenate([s, self._sub(bi, "B")], axis=1) if background and self.n_actors > self.n_scripted else s

    def hashes(self) -> dict:
        """流式复算 content、body、header 哈希（常量内存：逐块）。"""
        hc, hb = hashlib.sha256(self.header_bytes), hashlib.sha256()
        for bi in range(len(self.blocks)):
            f = self.block_frames(bi).tobytes()
            hc.update(f)
            hb.update(f)
        return {"content_sha256": hc.hexdigest(), "body_sha256": hb.hexdigest(), "header_sha256": hashlib.sha256(self.header_bytes).hexdigest(),
                "file_sha256": hashlib.sha256(self._buf).hexdigest()}

    def scripted_sha256(self) -> str:
        """脚本演员列哈希：SHA-256(脚本演员表的 JCS ‖ 脚本演员子块的规范帧数据（tick 主序，idx < n_s）)，只解 S 子块。
        背景人数、背景抽样与背景子块的编码不影响该值（D3-AC-07"背景 0 / 40 / 200 / 1,000 人时脚本演员列哈希相同"）。"""
        h = hashlib.sha256(jcs(self.actors[:self.n_scripted]))
        for bi in range(len(self.blocks)):
            h.update(self.block_frames(bi, background=False).tobytes())
        return h.hexdigest()

    def verify(self, expect_content_sha256: str | None = None) -> dict:
        """校验块覆盖与内容哈希；不一致抛 614。"""
        t = 0
        for blk in self.blocks:
            if blk.tick0 != t:
                raise EventError(614, f"block index not contiguous at tick {t}")
            t += blk.n_ticks
        if t != self.n_ticks:
            raise EventError(614, f"blocks cover {t} ticks, header says {self.n_ticks}")
        h = self.hashes()
        if expect_content_sha256 and h["content_sha256"] != expect_content_sha256:
            raise EventError(614, "content_sha256 mismatch", {"expected": expect_content_sha256, "actual": h["content_sha256"]})
        return h

    def check_world(self, content_version: str, raycast_sha256: str | None = None) -> None:
        w = self.header.get("world", {})
        if w.get("content_version") != content_version or (raycast_sha256 and w.get("raycast_sha256") != raycast_sha256):
            raise EventError(615, "tape was compiled for another world version",
                             {"tape": {"content_version": w.get("content_version"), "raycast_sha256": w.get("raycast_sha256")},
                              "loaded": {"content_version": content_version, "raycast_sha256": raycast_sha256}})

    # ---------------------------------------------------------------- 演员表与通信区
    def actors_info(self, *, scripted_only: bool = False) -> list[dict]:
        """§7.1 ActorInfo：{idx, id, cls, scripted, profile, size_m, appearance, pos0_enu_m}（pos0 为第 0 帧位置）。"""
        f0 = self.frame(0, background=not scripted_only)
        out = []
        for a in self.actors[:self.n_scripted] if scripted_only else self.actors:
            r = f0[a["idx"]]
            out.append({"idx": a["idx"], "id": a["id"], "cls": a["class"], "scripted": a["scripted"], "profile": a["profile"],
                        "size_m": tuple(s / 1000.0 for s in a["size_mm"]), "appearance": a["appearance"],
                        "pos0_enu_m": (r["x_mm"] / 1000.0, r["y_mm"] / 1000.0, r["z_mm"] / 1000.0)})
        return out

    def comm_zones_at(self, t_ns: int) -> list[dict]:
        """§6.11：t ∈ [from, to) 的通信区 {id, polygon_m (K, 2), extra_loss, extra_latency_ms, bandwidth_scale, t_from_ns, t_to_ns}。"""
        out = []
        for z in (self.header.get("comms") or {}).get("zones", []):
            a, b = (int(v) * 1_000_000 for v in z["t_ms"])
            if a <= int(t_ns) < b:
                out.append({"id": z["id"], "polygon_m": np.asarray(z["polygon"], np.float64), "extra_loss": z["extra_loss"],
                            "extra_latency_ms": z["extra_latency_ms"], "bandwidth_scale": z["bandwidth_scale"], "t_from_ns": a, "t_to_ns": b})
        return out

    def _env_track(self) -> list:
        """环境轨按 M07 关键帧状态机（`KeyframeManager.apply`）依次生效：[(t_apply_ns, TransitionKf)]，与线上 `env/set` 逐条投递一致。"""
        if self._env is None:
            import copy

            from awr.environment.keyframe import EnvOp, KeyframeManager
            from awr.environment.weather.presets import presets

            P = presets()
            km = KeyframeManager(world_id=str(self.header["world"].get("world_id", "")), seed=int(self.header.get("seed", 0)), epoch=0,
                                 config=copy.deepcopy(P.doc["defaults"]["config"]), initial=P.default_vector(), initial_preset=None,
                                 bounds_min=(0.0, 0.0), bounds_max=(1.0, 1.0), P=P)
            out = []
            for e in self.header.get("env_track") or []:
                op = e["op"]
                t = int(e["t_ms"]) * 1_000_000
                kf = km.apply(EnvOp("set", patch=op["patch"], mode=op["mode"], duration_s=float(op["duration_s"])), t, km.kf.anchors)
                out.append((t, kf.transition()))
            self._env = out or [(0, km.kf.transition())]          # 无环境轨：M07 缺省标量
        return self._env

    def env_at(self, t_ns: int) -> EnvSample:
        """事件时刻 t 的环境（M23 §6.11 环境轨经 M07 `eval_env` 求值）；t < 0 取首帧。"""
        import bisect
        import math

        from awr.environment.weather import presets as wp
        from awr.environment.weather.transitions import eval_env

        trk = self._env_track()
        i = max(bisect.bisect_right([t for t, _ in trk], int(t_ns)) - 1, 0)
        S = np.asarray(eval_env(trk[i][1], int(t_ns), np.empty(wp.NF)), np.float64)
        spd, d = float(S[wp.SPEED_REF]), float(S[wp.DIR])
        a = math.radians(d)
        w = np.array([-spd * math.sin(a), -spd * math.cos(a), float(S[wp.W_MEAN])])
        return EnvSample(int(t_ns), w, spd, d, float(S[wp.SIGMA_REF]), float(S[wp.RAIN]), float(S[wp.MOR_BG]), float(S[wp.FOG_TOP]),
                         float(S[wp.COVER]), S)

    # ---------------------------------------------------------------- 读取与插值
    def frame(self, k: int, *, background: bool = True) -> np.ndarray:
        k = min(max(int(k), 0), self.n_ticks - 1)
        bi, j = divmod(k, BLOCK_TICKS)
        return self.block_frames(bi, background=background)[j]

    def motion_distance(self, t_ns: int, ids=None) -> np.ndarray:
        """Arc length from canonical visible positions; deterministic even after seeking.
        Hidden transitions never contribute to gait or wheel phase.
        """
        if not hasattr(self, '_motion_distances'):
            d = np.zeros((self.n_ticks, self.n_actors), np.float32)
            prev = None
            offset = 0
            for bi in range(len(self.blocks)):
                f = self.block_frames(bi)
                frames = np.concatenate([prev[None], f]) if prev is not None else f
                p = np.stack([frames['x_mm'], frames['y_mm']],-1).astype(float)*.001
                vis = np.isin(frames['state'], [1,2])
                step = np.linalg.norm(np.diff(p,axis=0),axis=-1)*(vis[1:] & vis[:-1])
                start = offset if prev is not None else 1
                d[start:offset+len(f)] = step
                offset += len(f)
                prev = f[-1]
            self._motion_distances = np.cumsum(d, axis=0)
        k = min(max(t_ns/TICK_NS,0.),self.n_ticks-1)
        lo = int(k)
        hi = min(lo + 1, self.n_ticks - 1)
        d = self._motion_distances[lo]*(1-(k-lo))+self._motion_distances[hi]*(k-lo)
        return d if ids is None else d[ids]

    def actor_states(self, t_ns: int, ids=None) -> ActorStates:
        """M23 §6.8：k = ⌊t / 100 ms⌋；可见性类别或容器变化时取阶跃（用第 k 帧），移动与等待之间以带速度的 Hermite 插值，
        yaw 走最短弧；t 超过末帧保持末帧；t < 0 返回第 0 帧且 state = 0。仅请求脚本演员时不解 B 子块。"""
        from awr.sim.fleet.stages.kinematic import hermite

        sel = np.arange(self.n_actors) if ids is None else np.asarray(ids, np.int64)
        bg = bool(len(sel)) and int(sel.max()) >= self.n_scripted
        t_ns = int(t_ns)
        if t_ns < 0:
            st = _unpack(sel, self.frame(0, background=bg)[sel])
            st.state[:] = 0
            st.vel_enu_mps[:] = 0.0
            return st
        k, rem = divmod(t_ns, TICK_NS)
        if k >= self.n_ticks - 1:
            return _unpack(sel, self.frame(self.n_ticks - 1, background=bg)[sel])
        r0 = self.frame(k, background=bg)[sel]
        st = _unpack(sel, r0)
        if rem == 0:
            return st
        r1 = self.frame(k + 1, background=bg)[sel]
        m = (np.isin(r0["state"], _VIS) & np.isin(r1["state"], _VIS) & (r0["container"] == r1["container"]))
        if m.any():
            a, b = _unpack(sel[m], r0[m]), _unpack(sel[m], r1[m])
            n = int(m.sum())
            h = TICK_NS / 1e9
            p, v = hermite(np.zeros(n), np.full(n, h), a.pos_enu_m, b.pos_enu_m, a.vel_enu_mps, b.vel_enu_mps, np.full(n, rem / 1e9))
            st.pos_enu_m[m] = p
            st.vel_enu_mps[m] = v
            y0 = r0["yaw_cdeg"][m].astype(np.float64)
            dy = (r1["yaw_cdeg"][m].astype(np.float64) - y0 + 18000.0) % 36000.0 - 18000.0
            st.yaw_rad[m] = np.radians((y0 + dy * (rem / TICK_NS)) / 100.0)
            for field in ('pitch_rad','roll_rad'):
                if getattr(st,field) is not None:
                    getattr(st,field)[m] = getattr(a,field)*(1-rem/TICK_NS)+getattr(b,field)*(rem/TICK_NS)
        return st


_VIS = np.array(sorted(STATE_VISIBLE), np.uint8)


def _unpack(sel: np.ndarray, r: np.ndarray) -> ActorStates:
    """记录 → SI 单位；不可见状态速度记 0。"""
    vis = np.isin(r["state"], _VIS)
    pos = np.stack([r["x_mm"], r["y_mm"], r["z_mm"]], 1).astype(np.float64) / 1000.0
    vel = np.stack([r["vx_cmps"], r["vy_cmps"], r["vz_cmps"]], 1).astype(np.float64) / 100.0
    vel[~vis] = 0.0
    return ActorStates(np.asarray(sel), pos, np.radians(r["yaw_cdeg"].astype(np.float64) / 100.0), vel, r["state"].copy(), r["container"].copy(),
                       np.radians(r["pitch_cdeg"]*.01) if "pitch_cdeg" in r.dtype.names else np.zeros(len(r)),
                       np.radians(r["roll_cdeg"]*.01) if "roll_cdeg" in r.dtype.names else np.zeros(len(r)))


def manifest(hashes: dict, header: dict, *, provenance: dict, report: dict | None = None) -> dict:
    """tape.json（drone4d.tape_manifest.v1，M23 §6.7.6）：不进入内容哈希，tape.bin 内的规范头为准。"""
    return {"schema": "drone4d.tape_manifest.v1", **{k: hashes[k] for k in ("content_sha256", "body_sha256", "header_sha256", "file_sha256")},
            "bytes": hashes["bytes"], "n_actors": len(header["actors"]), "n_ticks": header["n_ticks"], "header": header,
            "provenance": provenance,
            "report": report or {"arrivals": [], "boards": [], "warnings": [], "min_dist_m": {"scripted_scripted": None, "bg_scripted": None}}}


def open_tape(d: Path, *, verify: bool = True) -> TapeReader:
    """打开事件版本目录（tape.bin + tape.json），按清单中的 content_sha256 校验（614）。"""
    import json

    d = Path(d)
    r = TapeReader(d / "tape.bin")
    if verify:
        m = json.loads((d / "tape.json").read_text())
        r.verify(m["content_sha256"])
    return r
