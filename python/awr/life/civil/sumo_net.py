"""Convert the world's lane graph (``awr.navgraph.v1`` ``graph.geojson``) into a SUMO network.

    python -m awr.life.civil.sumo_net <world_dir_or_graph.geojson> --out <dir>

Writes ``nodes.nod.xml``, ``edges.edg.xml`` (plain XML) and runs ``netconvert`` into ``net.net.xml`` with sidewalks and
pedestrian crossings guessed, keeping the world ENU coordinates (no normalisation) so that SUMO positions are world
positions. Vehicle classes follow the graph's edge kinds:

    road / service   → all motorised classes + bicycle (sidewalk added for pedestrians)
    path (track)     → pedestrian + bicycle
    footway          → pedestrian

Junction rules from the graph map to SUMO node types (priority / right_before_left / traffic_light).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from xml.sax.saxutils import quoteattr

MOTOR = "passenger delivery taxi bus truck emergency motorcycle moped bicycle"
ALLOW = {"road": MOTOR, "service": MOTOR, "path": "pedestrian bicycle", "footway": "pedestrian"}
NODE_TYPE = {
    "priority": "priority",
    "equal": "right_before_left",
    "first_come": "right_before_left",
    "traffic_light": "traffic_light",
    "stop": "allway_stop",
}


def sumo_home() -> Path:
    if os.environ.get("SUMO_HOME"):
        return Path(os.environ["SUMO_HOME"])
    import sumo  # the eclipse-sumo wheel

    return Path(sumo.__file__).parent


def load_graph(src: Path) -> dict:
    p = Path(src)
    if p.is_dir():
        p = p / "geometry" / "nav" / "graph.geojson"
        if not p.exists():
            p = Path(src) / "graph.geojson"
    return json.loads(p.read_text(encoding="utf-8"))


def _shape(coords: list, z: list | None) -> str:
    pts = []
    for i, c in enumerate(coords):
        zz = c[2] if len(c) > 2 else (z[i] if z and i < len(z) else 0.0)
        pts.append(f"{c[0]:.2f},{c[1]:.2f},{zz:.2f}")
    return " ".join(pts)


def build_plain(graph: dict, out: Path) -> dict:
    """Write nodes/edges plain XML; returns {edge_id: {'f': sumo_edge, 'b': sumo_edge}} mapping of the graph edges."""
    feats = graph["features"]
    junction_rule = {
        f["properties"]["node"]: f["properties"].get("rule", "priority")
        for f in feats
        if f["properties"].get("feature") == "junction"
    }
    nodes: dict[str, tuple[float, float, float]] = {}
    edges_xml, mapping = [], {}
    for f in feats:
        p = f["properties"]
        if p.get("feature") != "edge":
            continue
        kind = p.get("kind", "road")
        coords = f["geometry"]["coordinates"]
        z = p.get("z_m")
        a, b = p["from_node"], p["to_node"]
        for nid, c, zz in ((a, coords[0], (z or [0.0])[0]), (b, coords[-1], (z or [0.0])[-1])):
            nodes.setdefault(nid, (c[0], c[1], c[2] if len(c) > 2 else zz))
        allow = ALLOW.get(kind, MOTOR)
        motor = kind in ("road", "service")
        speed = float(p.get("speed_mps", 13.889)) if motor else (5.0 if kind == "path" else 1.4)
        nf = int(p.get("lanes_forward", 1) or 1) if motor else 1
        nb = int(p.get("lanes_backward", 1) or 1) if motor else 1
        width = float(p.get("width_m", 6.4))
        lane_w = min(3.5, max(2.0, width / max(1, nf + nb))) if motor else max(1.5, width)
        prio = {"road": 3, "service": 1, "path": 1, "footway": 1}.get(kind, 1)
        eid = p["edge_id"]
        fwd = f"{eid}-f"
        edges_xml.append(
            f'<edge id={quoteattr(fwd)} from={quoteattr(a)} to={quoteattr(b)} numLanes="{nf}" speed="{speed:.3f}" width="{lane_w:.2f}" '
            f'priority="{prio}" allow={quoteattr(allow)} shape={quoteattr(_shape(coords, z))} type={quoteattr(kind)}/>'
        )
        mapping[eid] = {"f": fwd}
        if not p.get("oneway", False):
            bwd = f"{eid}-b"
            edges_xml.append(
                f'<edge id={quoteattr(bwd)} from={quoteattr(b)} to={quoteattr(a)} numLanes="{nb}" speed="{speed:.3f}" width="{lane_w:.2f}" '
                f'priority="{prio}" allow={quoteattr(allow)} shape={quoteattr(_shape(coords[::-1], (z or [])[::-1]))} type={quoteattr(kind)}/>'
            )
            mapping[eid]["b"] = bwd
    out.mkdir(parents=True, exist_ok=True)
    with (out / "nodes.nod.xml").open("w", encoding="utf-8") as fh:
        fh.write('<?xml version="1.0" encoding="UTF-8"?>\n<nodes>\n')
        for nid, (x, y, z) in nodes.items():
            t = NODE_TYPE.get(junction_rule.get(nid, "priority"), "priority")
            fh.write(f'  <node id={quoteattr(nid)} x="{x:.2f}" y="{y:.2f}" z="{z:.2f}" type="{t}"/>\n')
        fh.write("</nodes>\n")
    with (out / "edges.edg.xml").open("w", encoding="utf-8") as fh:
        fh.write('<?xml version="1.0" encoding="UTF-8"?>\n<edges>\n')
        for e in edges_xml:
            fh.write("  " + e + "\n")
        fh.write("</edges>\n")
    (out / "edge_map.json").write_text(json.dumps(mapping, indent=0), encoding="utf-8")
    return mapping


def netconvert(out: Path, *, sidewalks: bool = True, crossings: bool = True) -> Path:
    from sumolib import checkBinary

    net = out / "net.net.xml"
    cmd = [
        checkBinary("netconvert"),
        "--node-files",
        str(out / "nodes.nod.xml"),
        "--edge-files",
        str(out / "edges.edg.xml"),
        "--output-file",
        str(net),
        "--offset.disable-normalization",
        "true",
        "--no-turnarounds.except-deadend",
        "true",
        "--junctions.corner-detail",
        "5",
        "--geometry.min-radius.fix",
        "true",
        "--no-internal-links",
        "false",
        "--roundabouts.guess",
        "true",
        "--junctions.join",
        "false",
        "--default.sidewalk-width",
        "2.0",
        "--default.crossing-width",
        "4.0",
        "--seed",
        "1",
        "--no-warnings",
        "true",
    ]
    if sidewalks:
        cmd += ["--sidewalks.guess", "true", "--sidewalks.guess.min-speed", "0", "--sidewalks.guess.max-speed", "20"]
    if crossings:
        cmd += ["--crossings.guess", "true", "--crossings.guess.speed-threshold", "14", "--walkingareas", "true"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    (out / "netconvert.log").write_text(r.stdout + r.stderr, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(f"netconvert failed: {r.stderr[-2000:]}")
    return net


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("world")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--no-sidewalks", action="store_true")
    a = ap.parse_args()
    g = load_graph(Path(a.world))
    m = build_plain(g, a.out)
    net = netconvert(a.out, sidewalks=not a.no_sidewalks)
    print(f"edges {len(m)} → {net}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
