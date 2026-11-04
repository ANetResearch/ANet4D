"""受邀提交容器红队用例（D3-AC-59 负向用例："越权系统调用、联网尝试、fork 炸弹、写满 tmpfs、读取 `/data/d3/hidden` 的尝试被加固策略
阻断（691 并审计）且不影响同机其他作业"；§5.6.7 容器加固、ADR-164；M26 §8.1）。

加固策略与审计来源见 `awr.agentsdk.env.containers` 模块说明：seccomp 配置 `d3-agent-v1`（逃逸类系统调用 KILL_PROCESS）、
`--network none`、`--pids-limit 256`、`/tmp` tmpfs 256 MB、只读根、`--cap-drop ALL`、非 root；`/data/d3/hidden` 只挂载只读诱饵。
运行期审计（`harden_audit=True`）不提权：退出码 159（SIGSYS）、监视容器读取 Agent 网络命名空间的 `OutNoRoutes`、Engine API 统计的
进程数与 `shmem` 峰值、宿主侧对诱饵目录的 inotify；每个触发的向量写一条 691 审计记录（`agents/<uav>/hardening.jsonl` 与 `audit_sink`）。

前半部分不需要 docker：seccomp 配置结构、启动参数、判据函数、诱饵监视与审计记录（docker 替身）。`test_dmax_*` 以真实镜像
（`AWR_B24_AGENT_IMAGE`，缺省 `d3-agent-py:dev`）经生产启动器 `AgentContainers` 在 dmax CPU 池运行红队 Agent：每个向量一个容器，
断言被阻断、审计记录恰为该向量，良性对照无记录；同机邻居作业（另一容器：fork 32 个子进程计算、写 64 MB tmpfs）在红队容器
并发运行时功能正常且耗时 ≤ max(1.5 × 单独运行耗时, 单独运行耗时 + 2 s)。docker 或镜像不可用时跳过（platform 层；dmax 命令见
results-B04/B24.md）。`AWR_B24_OUT` 给出时写 `redteam_report.json`（逐向量证据、审计记录、邻居耗时、UTC 时刻）。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import pytest

from awr.agentsdk.env import containers as C
from awr.agentsdk.env.containers import (
    HARDEN_CODE,
    HIDDEN_PATH,
    PIDS_LIMIT,
    SECCOMP_KILL,
    SECCOMP_NAME,
    SIGSYS_EXIT,
    TMPFS_MB,
    AgentContainers,
    HardeningMonitor,
    check_launch,
    hardening_findings,
    make_decoy,
    net_noroutes,
    parse_docker_run,
    seccomp_path,
    seccomp_profile,
    write_seccomp,
)

IMAGE = os.environ.get("AWR_B24_AGENT_IMAGE", "d3-agent-py:dev")
DOCKER = os.environ.get("AWR_B24_DOCKER", "docker")
OUT = os.environ.get("AWR_B24_OUT")
RUN_ID = "sub-4f1a2b3c4d5e-r0"

# dmax 实测（2026-10-19，Docker 29.1.4）：Agent 退出后监视容器读到的计数（连接 1.1.1.1:443 一次）
SNMP = """Ip: Forwarding DefaultTTL InReceives InHdrErrors InAddrErrors ForwDatagrams InUnknownProtos InDiscards InDelivers OutRequests OutDiscards OutNoRoutes ReasmTimeout ReasmReqds ReasmOKs ReasmFails FragOKs FragFails FragCreates
Ip: 1 64 16 0 0 0 0 0 16 16 4 1 0 0 0 0 0 0 0
Icmp: InMsgs InErrors
Icmp: 0 0
Ip6InReceives                   \t0
Ip6OutNoRoutes                  \t3
"""


# ====================================================================== 本机：seccomp 配置与启动参数

def test_seccomp_profile_denies_by_default_and_kills_escape_syscalls() -> None:
    p = seccomp_profile()
    assert p["defaultAction"] == "SCMP_ACT_ERRNO" and p["defaultErrnoRet"] == 1
    assert {a["architecture"] for a in p["archMap"]} == {"SCMP_ARCH_X86_64", "SCMP_ARCH_AARCH64"}
    by_action: dict[str, set[str]] = {}
    for r in p["syscalls"]:
        by_action.setdefault(r["action"], set()).update(r["names"])
    kill = by_action["SCMP_ACT_KILL_PROCESS"]
    assert kill == set(SECCOMP_KILL)
    for n in ("mount", "umount2", "unshare", "setns", "pivot_root", "chroot", "init_module", "finit_module", "kexec_load", "ptrace",
              "process_vm_readv", "open_by_handle_at", "fsopen", "move_mount", "open_tree", "reboot"):
        assert n in kill, n
    unconditional = set(p["syscalls"][0]["names"])
    assert not unconditional & kill
    for n in ("bpf", "perf_event_open", "keyctl", "add_key", "userfaultfd", "io_uring_setup", "clone", "clone3", "socket", "personality"):
        assert n not in unconditional, n                                 # 不在无条件放行集合：缺省 EPERM 或带参数条件
    for n in ("read", "write", "openat", "mmap", "futex", "execve", "wait4", "socketpair", "connect", "rt_sigreturn", "exit_group"):
        assert n in unconditional, n
    clone = [r for r in p["syscalls"] if r["names"] == ["clone"]]
    assert clone and clone[0]["args"] == [{"index": 0, "value": 0x7E020000, "valueTwo": 0, "op": "SCMP_CMP_MASKED_EQ"}]
    clone3 = [r for r in p["syscalls"] if r["names"] == ["clone3"]]
    assert clone3 and clone3[0]["action"] == "SCMP_ACT_ERRNO" and clone3[0]["errnoRet"] == 38   # ENOSYS：glibc 回退到 clone


def test_launcher_argv_carries_seccomp_profile_without_run_identity(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    run_dir = tmp_path / "dataset" / "handoff-07" / "runs" / RUN_ID
    ac = AgentContainers(run_dir, RUN_ID, ["uav-01"], IMAGE, cpus=2.0, mem_gb=8.0, harden_audit=False)
    argv = ac.argv("uav-01", Path("/run/d3/agents/0123456789abcdef/sock"))
    a = " ".join(argv)
    sp = seccomp_path()
    for flag in ("--network none", "--read-only", f"--tmpfs /tmp:rw,size={TMPFS_MB}m", "--cap-drop ALL", "--security-opt no-new-privileges",
                 f"--security-opt seccomp={sp}", f"--pids-limit {PIDS_LIMIT}", "--user 65534:65534"):
        assert flag in a, flag
    assert SECCOMP_NAME in sp.name and RUN_ID not in str(sp) and "handoff-07" not in str(sp)
    assert [x for x in check_launch(parse_docker_run(argv, "agent", component="agent", run=RUN_ID)) if x.level == "error"] == []
    assert [argv[i + 1] for i, x in enumerate(argv) if x == "-v"] == ["/run/d3/agents/0123456789abcdef/sock:/run/drone4d"]
    p = write_seccomp()
    assert p == sp and json.loads(p.read_text()) == seccomp_profile()
    m0 = p.stat().st_mtime_ns
    assert write_seccomp() == p and p.stat().st_mtime_ns == m0                  # 内容相同不重写
    # 审计开启：诱饵目录只读挂载到隐藏存储路径
    argv2 = ac.argv("uav-01", Path("/run/d3/agents/0123456789abcdef/sock"), decoy=Path("/run/d3/agents/0123456789abcdef/sock-hid"))
    assert f"/run/d3/agents/0123456789abcdef/sock-hid:{HIDDEN_PATH}:ro" in argv2
    assert [x for x in check_launch(parse_docker_run(argv2, "agent", component="agent", run=RUN_ID)) if x.level == "error"] == []


def test_monitor_container_is_unprivileged_and_passes_manifest_rules() -> None:
    """监视容器持有 Agent 的网络命名空间：自身 `--network none`、非 root、无能力；Agent 以 `container:<监视容器>` 加入（同样只有 lo）。"""
    mon = HardeningMonitor(f"d3-agent-{RUN_ID}-uav-01", image="d3-agent-py:dev", labels=[f"d3.run={RUN_ID}"])
    argv = mon.monitor_argv()
    a = " ".join(argv)
    assert mon.network == f"container:d3-agent-{RUN_ID}-uav-01-mon"
    for flag in ("--network none", "--read-only", "--cap-drop ALL", "--security-opt no-new-privileges",
                 "--user 65534:65534", "--pids-limit 8"):
        assert flag in a, flag
    assert "--privileged" not in argv and "-v" not in argv and "--pid" not in argv and "--cgroupns" not in argv
    errs = [x for x in check_launch(parse_docker_run(argv, "agent", component="agent", run=RUN_ID)) if x.level == "error"]
    assert errs == []
    ac = AgentContainers(Path("/x"), RUN_ID, ["uav-01"], IMAGE)
    sock = Path("/run/d3/agents/0123456789abcdef/sock")
    assert f"--network {mon.network}" in " ".join(ac.argv("uav-01", sock, network=mon.network))
    for bad in ("host", "bridge", "container:d3-bench", "d3-net"):
        with pytest.raises(ValueError):
            ac.argv("uav-01", sock, network=bad)


# ====================================================================== 本机：判据与审计记录

def test_net_noroutes_parses_snmp_and_snmp6() -> None:
    assert net_noroutes(SNMP) == {"ip_out_no_routes": 1, "ip6_out_no_routes": 3}
    assert net_noroutes("") == {}


@pytest.mark.parametrize(("ev", "vectors"), [
    ({"exit_code": 0, "pids_peak": 3, "shmem_peak_b": 1 << 20, "net": {"ip_out_no_routes": 0, "ip6_out_no_routes": 0}, "hidden": []}, []),
    ({"exit_code": SIGSYS_EXIT}, ["syscall"]),
    ({"exit_code": 0, "net": {"ip_out_no_routes": 2, "ip6_out_no_routes": 0}}, ["network"]),
    ({"exit_code": 137, "pids_peak": PIDS_LIMIT}, ["fork_bomb"]),
    ({"exit_code": 0, "pids_peak": PIDS_LIMIT - 1}, []),
    ({"exit_code": 0, "shmem_peak_b": (TMPFS_MB << 20) - (4 << 20)}, ["tmpfs_fill"]),
    ({"exit_code": 0, "shmem_peak_b": (TMPFS_MB << 20) // 2}, []),
    ({"exit_code": 0, "hidden": [{"name": ".", "op": "open", "dir": True}], "hidden_events": 1}, ["hidden_read"]),
    ({"exit_code": SIGSYS_EXIT, "net": {"ip_out_no_routes": 1}, "pids_peak": PIDS_LIMIT, "shmem_peak_b": TMPFS_MB << 20,
      "hidden": [{"name": "index.json", "op": "read", "dir": False}]}, ["syscall", "network", "fork_bomb", "tmpfs_fill", "hidden_read"]),
])
def test_hardening_findings(ev: dict, vectors: list[str]) -> None:
    assert [v for v, _ in hardening_findings(ev)] == vectors


def test_decoy_watch_records_open_and_read(tmp_path: Path) -> None:
    d = make_decoy(tmp_path / "hid")
    assert sorted(x.name for x in d.iterdir()) == ["index.json"] and (d / "index.json").read_text() == "{}\n"
    assert oct(d.stat().st_mode & 0o777) == "0o755" and oct((d / "index.json").stat().st_mode & 0o777) == "0o444"
    stub = tmp_path / "docker"                                           # the docker-free half: the monitor container is a stand-in
    stub.write_text("#!/bin/sh\nexit 0\n")
    stub.chmod(0o755)
    quiet = HardeningMonitor("d3-agent-x-uav-01", docker=str(stub), decoy=make_decoy(tmp_path / "quiet"), stats=lambda n: None)
    quiet.watch()
    mon = HardeningMonitor("d3-agent-x-uav-02", docker=str(stub), decoy=d, stats=lambda n: None)
    mon.watch()
    subprocess.run([sys.executable, "-c", f"import os; print(os.listdir({str(d)!r})); open({str(d / 'index.json')!r}).read()"], check=True,
                   capture_output=True)
    time.sleep(0.5)
    ev, ev_quiet = mon.finish(0), quiet.finish(0)
    assert ev_quiet["hidden"] == [] and ev_quiet["errors"] == []
    names = {(e["name"], e["op"]) for e in ev["hidden"]}
    assert (".", "open") in names and ("index.json", "open") in names and ("index.json", "read") in names
    assert [v for v, _ in hardening_findings(ev)] == ["hidden_read"]


_FAKE = r'''#!/bin/sh
# 测试用 docker 替身：run/rm/wait/kill/logs 成功；inspect 返回 $FAKE_EXIT；exec 返回 snmp 文本
case "$1" in
  run) echo "$@" >> "$FAKE_STATE/runs" ;;
  inspect) printf "%s" "$FAKE_EXIT" ;;
  exec) cat "$FAKE_STATE/snmp" ;;
  logs) echo "agent log" ;;
esac
exit 0
'''


def test_audit_records_written_and_forwarded(tmp_path: Path, monkeypatch) -> None:
    """启动器审计链路（docker 替身）：主进程 SIGSYS 退出、联网计数、进程数与 shmem 峰值、诱饵读取 → 五条 691 记录，
    写入 `agents/<uav>/hardening.jsonl` 并逐条交给 `audit_sink`（与 bench `AuditLog.write` 同签名）。"""
    st = tmp_path / "state"
    st.mkdir()
    (st / "snmp").write_text(SNMP)
    fake = tmp_path / "docker"
    fake.write_text(_FAKE)
    fake.chmod(0o755)
    monkeypatch.setenv("FAKE_STATE", str(st))
    monkeypatch.setenv("FAKE_EXIT", str(SIGSYS_EXIT))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    monkeypatch.setenv("AWR_AGENT_SOCK_ROOT", str(tmp_path / "as"))
    monkeypatch.setattr(C, "engine_stats", lambda name: {"pids_stats": {"current": PIDS_LIMIT, "limit": PIDS_LIMIT},
                                                          "memory_stats": {"stats": {"shmem": TMPFS_MB << 20}}})
    sink: list[tuple[str, dict]] = []
    ac = AgentContainers(tmp_path / "run", RUN_ID, ["uav-01"], IMAGE, docker=str(fake), harden_audit=True,
                         audit_sink=lambda action, **kw: sink.append((action, kw)))
    ac.start()
    decoy = ac.monitors["uav-01"].decoy
    assert decoy is not None
    (decoy / "index.json").read_text()                                  # 模拟容器内读取诱饵
    time.sleep(0.6)
    codes = ac.stop()
    assert codes == {"uav-01": SIGSYS_EXIT}
    runs = (st / "runs").read_text().splitlines()
    assert any(f"{decoy}:{HIDDEN_PATH}:ro" in r for r in runs)
    assert "-mon " in runs[0] and "--network none" in runs[0]                 # 监视容器先于 Agent 启动并持有网络命名空间
    assert f"--network container:{ac.name('uav-01')}-mon" in runs[1]
    recs = ac.hardening["uav-01"]
    assert [r["vector"] for r in recs] == ["syscall", "network", "fork_bomb", "tmpfs_fill", "hidden_read"]
    assert all(r["code"] == HARDEN_CODE and r["action"] == "deny" and r["why"] == "hardening" and r["run_id"] == RUN_ID for r in recs)
    lines = [json.loads(x) for x in (tmp_path / "run/agents/uav-01/hardening.jsonl").read_text().splitlines()]
    assert lines[0]["kind"] == "evidence" and lines[0]["net"] == {"ip_out_no_routes": 1, "ip6_out_no_routes": 3}
    assert [x["vector"] for x in lines[1:]] == [r["vector"] for r in recs]
    assert [a for a, _ in sink] == ["deny"] * 5 and [kw["vector"] for _, kw in sink] == [r["vector"] for r in recs]
    assert all(kw["code"] == HARDEN_CODE for _, kw in sink)


def test_audit_disabled_by_default_keeps_single_mount(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("AWR_AGENT_HARDEN_AUDIT", raising=False)
    assert AgentContainers(tmp_path, RUN_ID, ["uav-01"], IMAGE).harden_audit is False
    monkeypatch.setenv("AWR_AGENT_HARDEN_AUDIT", "1")
    assert AgentContainers(tmp_path, RUN_ID, ["uav-01"], IMAGE).harden_audit is True


# ====================================================================== dmax：真实镜像红队

def _docker_ok() -> str | None:
    if shutil.which(DOCKER) is None:
        return "docker CLI not found"
    r = subprocess.run([DOCKER, "image", "inspect", IMAGE], capture_output=True, check=False)
    return None if r.returncode == 0 else f"image {IMAGE} unavailable"


_SKIP = _docker_ok()
dmax = pytest.mark.skipif(_SKIP is not None, reason=f"[platform] {_SKIP}")

_REPORT: dict = {"image": IMAGE, "vectors": {}}


@pytest.fixture(autouse=True)
def _short_sock_root(request, monkeypatch):
    """真实容器用例：套接字路径受 AF_UNIX 长度限制，未设 `AWR_AGENT_SOCK_ROOT` 时取短临时目录（dmax 入口脚本另设映射目录）。"""
    if not request.node.name.startswith("test_dmax") or os.environ.get("AWR_AGENT_SOCK_ROOT"):
        yield
        return
    d = tempfile.mkdtemp(prefix="b24-", dir="/tmp")
    monkeypatch.setenv("AWR_AGENT_SOCK_ROOT", d)
    yield
    shutil.rmtree(d, ignore_errors=True)

# 红队 Agent 脚本（容器内只用标准库；结果以 `B24 ` 前缀写到标准输出，经 container.log 带回）
_PRE = "import ctypes, errno, json, os, socket, sys, time\ndef out(**k):\n    print('B24 ' + json.dumps(k), flush=True)\n"

SCRIPTS = {
    "benign": """
import subprocess, threading
r = []
t = threading.Thread(target=lambda: r.append(sum(range(100000)))); t.start(); t.join()
p = subprocess.run(['sh', '-c', 'echo child'], capture_output=True, text=True)
a, b = socket.socketpair(); a.send(b'x')
with open('/tmp/scratch.bin', 'wb') as f:
    f.write(os.urandom(16 << 20))
os.remove('/tmp/scratch.bin')
out(ok=True, thread=r[0], child=p.stdout.strip(), pair=b.recv(1).decode(), root=sorted(os.listdir('/'))[:5])
time.sleep(2)
""",
    "syscall_unshare": "out(stage='before')\nctypes.CDLL(None, use_errno=True).unshare(0x10000000)\nout(stage='after')\n",
    "syscall_mount": "out(stage='before')\nctypes.CDLL(None, use_errno=True).mount(b'none', b'/tmp', b'tmpfs', 0, None)\nout(stage='after')\n",
    "syscall_ptrace": "out(stage='before')\nctypes.CDLL(None, use_errno=True).ptrace(0, 0, None, None)\nout(stage='after')\n",
    "syscall_setns": "out(stage='before')\nfd = os.open('/proc/self/ns/net', os.O_RDONLY)\nctypes.CDLL(None, use_errno=True).setns(fd, 0)\nout(stage='after')\n",
    "network": """
res = {}
for host, port in (('1.1.1.1', 443), ('8.8.8.8', 53), ('223.5.5.5', 53), ('2606:4700:4700::1111', 443), ('2400:3200::1', 53)):
    try:
        socket.create_connection((host, port), timeout=1.5).close(); res[f'tcp {host}'] = 'ok'
    except OSError as e:
        res[f'tcp {host}'] = errno.errorcode.get(e.errno or 0, str(e))
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.sendto(b'\\0' * 12, ('8.8.8.8', 53)); res['udp 8.8.8.8'] = 'sent'
except OSError as e:
    res['udp 8.8.8.8'] = errno.errorcode.get(e.errno or 0, str(e))
for name in ('pypi.org', 'modelscope.cn'):
    try:
        socket.getaddrinfo(name, 443); res[f'dns {name}'] = 'ok'
    except OSError as e:
        res[f'dns {name}'] = f'gaierror {e.args[0]}'
out(res=res, ifaces=sorted(os.listdir('/sys/class/net')))
time.sleep(3)
""",
    "fork_bomb": """
out(stage='start', pid=os.getpid())
fails = 0
t0 = time.monotonic()
while time.monotonic() - t0 < 30:
    try:
        if os.fork() == 0:
            while True:
                try:
                    os.fork()
                except OSError:
                    time.sleep(0.01)
    except OSError:
        fails += 1
        if fails in (1, 1000):
            out(stage='fork_failed', fails=fails)
        time.sleep(0.01)
""",
    "tmpfs_fill": """
n, err = 0, None
blk = b'\\xa5' * (1 << 20)
try:
    with open('/tmp/fill.bin', 'wb') as f:
        while n < (2 << 30):
            f.write(blk); f.flush(); n += len(blk)
except OSError as e:
    err = errno.errorcode.get(e.errno or 0, str(e))
st = os.statvfs('/tmp')
try:
    open('/fill-root.bin', 'wb').write(b'x'); root = 'ok'
except OSError as e:
    root = errno.errorcode.get(e.errno or 0, str(e))
out(written=n, err=err, tmp_total=st.f_blocks * st.f_frsize, root=root)
time.sleep(3)
""",
    "hidden_read": """
res = {}
try:
    res['listdir'] = sorted(os.listdir('/data/d3/hidden'))
except OSError as e:
    res['listdir'] = errno.errorcode.get(e.errno or 0, str(e))
for p in ('/data/d3/hidden/index.json', '/data/d3/hidden/test/truth/run_0001.mcap', '/data/d3/hidden/events/handoff-07/event.json',
          '/data/d3/evalstore/seeds.json', '/data/d3/hidden/../evalstore/seeds.json'):
    try:
        with open(p, 'rb') as f:
            res[p] = f'read {len(f.read())}'
    except OSError as e:
        res[p] = errno.errorcode.get(e.errno or 0, str(e))
try:
    open('/data/d3/hidden/planted.txt', 'w').write('x'); res['write'] = 'ok'
except OSError as e:
    res['write'] = errno.errorcode.get(e.errno or 0, str(e))
mounts = [ln.split()[4] for ln in open('/proc/self/mountinfo')]
out(res=res, data_mounts=sorted(m for m in mounts if m.startswith('/data')))
time.sleep(2)
""",
}


def _agent(tmp_path: Path, key: str, *, wait_s: float = 4.0, sink: list | None = None, tag: str = "") -> dict:
    """以生产启动器运行一个红队 Agent（不做线协议握手）：启动 → 等待 `wait_s` 或容器退出 → stop → 证据与审计记录。"""
    run_id = f"b24-rt-{key.replace('_', '-')}"
    ac = AgentContainers(tmp_path / key, run_id, ["uav-01"], IMAGE, cmd=["python", "-c", _PRE + textwrap.dedent(SCRIPTS[key])],
                         cpus=1.0, mem_gb=2.0, docker=DOCKER, harden_audit=True, monitor_image=IMAGE,
                         audit_sink=(lambda action, **kw: sink.append((action, kw))) if sink is not None else None)
    ac.start()
    name = ac.names["uav-01"]
    t0 = time.monotonic()
    while time.monotonic() - t0 < wait_s:
        r = subprocess.run([DOCKER, "inspect", "-f", "{{.State.Running}}", name], capture_output=True, text=True, check=False)
        if r.stdout.strip() != "true":
            break
        time.sleep(0.25)
    codes = ac.stop()
    log = (tmp_path / key / "agents/uav-01/container.log").read_text(errors="replace")
    msgs = [json.loads(x[4:]) for x in log.splitlines() if x.startswith("B24 ")]
    lines = [json.loads(x) for x in (tmp_path / key / "agents/uav-01/hardening.jsonl").read_text().splitlines()]
    ev = lines[0]
    recs = ac.hardening["uav-01"]
    left = subprocess.run([DOCKER, "ps", "-a", "--format", "{{.Names}}", "--filter", f"name={name}"], capture_output=True, text=True,
                          check=False).stdout.split()
    res = {"exit_code": codes["uav-01"], "msgs": msgs, "evidence": ev, "records": recs, "vectors": [r["vector"] for r in recs],
           "leftover": left, "log_tail": log[-2000:]}
    _REPORT["vectors"][tag + key] = {k: v for k, v in res.items() if k != "log_tail"}
    return res


def _write_report() -> None:
    if OUT:
        _REPORT["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        Path(OUT).mkdir(parents=True, exist_ok=True)
        (Path(OUT) / "redteam_report.json").write_text(json.dumps(_REPORT, ensure_ascii=False, indent=1, sort_keys=True))


@dmax
def test_dmax_benign_control_runs_under_profile_without_findings(tmp_path: Path) -> None:
    r = _agent(tmp_path, "benign")
    assert r["exit_code"] == 0, r["log_tail"]
    m = r["msgs"][-1]
    assert m["ok"] and m["child"] == "child" and m["pair"] == "x"
    assert r["vectors"] == [] and r["evidence"]["samples"] > 0 and r["evidence"]["errors"] == []
    assert r["evidence"]["net"] is not None and r["leftover"] == []
    _write_report()


@dmax
@pytest.mark.parametrize("key", ["syscall_unshare", "syscall_mount", "syscall_ptrace", "syscall_setns"])
def test_dmax_privileged_syscall_killed_and_audited(tmp_path: Path, key: str) -> None:
    sink: list = []
    r = _agent(tmp_path, key, sink=sink)
    assert r["exit_code"] == SIGSYS_EXIT, r["log_tail"]
    stages = [m.get("stage") for m in r["msgs"]]
    assert stages == ["before"]                                            # 调用即终止，后续语句未执行
    assert r["vectors"] == ["syscall"]
    assert r["records"][0]["code"] == HARDEN_CODE and r["records"][0]["evidence"]["seccomp"] == SECCOMP_NAME
    assert [kw["vector"] for _, kw in sink] == ["syscall"] and r["leftover"] == []
    assert r["evidence"]["errors"] == [] and r["evidence"]["net"] == {"ip_out_no_routes": 0, "ip6_out_no_routes": 0}
    _write_report()


@dmax
def test_dmax_network_attempts_fail_and_audited(tmp_path: Path) -> None:
    r = _agent(tmp_path, "network")
    assert r["exit_code"] == 0, r["log_tail"]
    m = r["msgs"][-1]
    assert m["ifaces"] == ["lo"]
    bad = {k: v for k, v in m["res"].items() if v in ("ok", "sent")}
    assert bad == {}, bad
    assert all(v in ("ENETUNREACH", "EHOSTUNREACH", "EADDRNOTAVAIL") for k, v in m["res"].items() if k.startswith(("tcp", "udp"))), m["res"]
    assert all(v.startswith("gaierror") for k, v in m["res"].items() if k.startswith("dns")), m["res"]
    assert r["vectors"] == ["network"]
    net = r["records"][0]["evidence"]
    assert net["ip_out_no_routes"] >= 4 and net["ip6_out_no_routes"] >= 2, net    # 4 次 IPv4（3 TCP + 1 UDP）、2 次 IPv6 尝试
    _write_report()


@dmax
def test_dmax_fork_bomb_capped_and_audited(tmp_path: Path) -> None:
    r = _agent(tmp_path, "fork_bomb", wait_s=6.0)
    ev = r["evidence"]
    assert ev["pids_peak"] == PIDS_LIMIT, ev                               # 达到上限且从未超过
    assert any(m.get("stage") == "fork_failed" for m in r["msgs"]), r["log_tail"]
    assert r["vectors"] == ["fork_bomb"] and r["leftover"] == []
    _write_report()


@dmax
def test_dmax_tmpfs_fill_capped_and_audited(tmp_path: Path) -> None:
    r = _agent(tmp_path, "tmpfs_fill")
    assert r["exit_code"] == 0, r["log_tail"]
    m = r["msgs"][-1]
    assert m["err"] == "ENOSPC" and m["written"] <= TMPFS_MB << 20 and m["tmp_total"] == TMPFS_MB << 20, m
    assert m["root"] in ("EROFS", "EACCES"), m                              # 只读根文件系统
    assert r["vectors"] == ["tmpfs_fill"], r["evidence"]
    _write_report()


@dmax
def test_dmax_hidden_store_unreachable_and_audited(tmp_path: Path) -> None:
    r = _agent(tmp_path, "hidden_read")
    assert r["exit_code"] == 0, r["log_tail"]
    res = r["msgs"][-1]["res"]
    assert res["listdir"] == ["index.json"]                                # 只有诱饵：真实隐藏存储从不挂载
    assert res["/data/d3/hidden/index.json"] == "read 3"
    for p in ("/data/d3/hidden/test/truth/run_0001.mcap", "/data/d3/hidden/events/handoff-07/event.json", "/data/d3/evalstore/seeds.json",
              "/data/d3/hidden/../evalstore/seeds.json"):
        assert res[p] == "ENOENT", (p, res[p])
    assert res["write"] in ("EROFS", "EACCES")
    assert r["msgs"][-1]["data_mounts"] == [HIDDEN_PATH]
    assert r["vectors"] == ["hidden_read"]
    ops = {(e["name"], e["op"]) for e in r["records"][0]["evidence"]["events"]}
    assert (".", "open") in ops and ("index.json", "read") in ops
    _write_report()


NEIGHBOR = r"""
import json, os, time
t0 = time.monotonic()
kids = []
for i in range(32):
    pid = os.fork()
    if pid == 0:
        s = 0
        for k in range(3_000_000):
            s += k * k
        os._exit(0)
    kids.append(pid)
ok = all(os.waitpid(p, 0)[1] == 0 for p in kids)
with open('/tmp/n.bin', 'wb') as f:
    for _ in range(64):
        f.write(os.urandom(1 << 20))
    f.flush(); os.fsync(f.fileno())
print(json.dumps({"ok": ok, "forks": len(kids), "wrote_mb": 64, "elapsed_s": time.monotonic() - t0}))
"""


def _neighbor(tag: str) -> dict:
    name = f"d3-test-b24-neighbor-{tag}"
    subprocess.run([DOCKER, "rm", "-f", name], capture_output=True, check=False)
    r = subprocess.run([DOCKER, "run", "--rm", "--name", name, "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=128m",
                        "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "128", "--cpus", "4", "--memory", "1g",
                        "--user", "65534:65534", IMAGE, "python", "-c", NEIGHBOR], capture_output=True, text=True, timeout=600, check=False)
    assert r.returncode == 0, r.stderr[-1000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@dmax
def test_dmax_attacks_do_not_affect_neighbor_job(tmp_path: Path) -> None:
    """同机邻居作业：单独运行两次取较小耗时为基线；fork 炸弹、写满 tmpfs 与越权系统调用三个红队容器并发运行时再跑一次。"""
    import threading
    base = min(_neighbor("base1")["elapsed_s"], _neighbor("base2")["elapsed_s"])
    results: dict[str, dict] = {}
    errs: list[BaseException] = []

    def go(key: str, wait_s: float) -> None:
        try:
            results[key] = _agent(tmp_path / "concurrent", key, wait_s=wait_s, tag="concurrent/")
        except BaseException as e:                                         # 线程内失败转交主线程断言
            errs.append(e)
    ts = [threading.Thread(target=go, args=a) for a in (("fork_bomb", 12.0), ("tmpfs_fill", 8.0), ("syscall_unshare", 4.0))]
    for t in ts:
        t.start()
    time.sleep(3.0)                                                        # 炸弹达到上限、tmpfs 写满之后启动邻居
    during = _neighbor("during")
    for t in ts:
        t.join(timeout=120)
    assert not errs, errs
    assert during["ok"] and during["forks"] == 32
    limit = max(1.5 * base, base + 2.0)
    _REPORT["neighbor"] = {"baseline_s": base, "during_s": during["elapsed_s"], "limit_s": limit}
    _write_report()
    assert during["elapsed_s"] <= limit, _REPORT["neighbor"]
    assert results["fork_bomb"]["vectors"] == ["fork_bomb"]
    assert results["tmpfs_fill"]["vectors"] == ["tmpfs_fill"]
    assert results["syscall_unshare"]["vectors"] == ["syscall"]
