"""Command line of the scene principle package.

python -m awr.life.principle check DIR [--json]
python -m awr.life.principle pack --world W --nav N --semantics S [--lane-network L] [--outdoor-places P] [--config C]
                                  [--capabilities caps.json] [--sources sources.json] [--notes TEXT] --out DIR
python -m awr.life.principle describe DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from awr.life.principle import check, load, pack


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="awr.life.principle")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="admission check of a principle package")
    c.add_argument("dir", type=Path)
    c.add_argument("--json", action="store_true", help="print the full report as JSON")
    c.add_argument("--out", type=Path, help="write the report to this file")
    p = sub.add_parser("pack", help="build a principle package from legacy inputs")
    p.add_argument("--world", type=Path, required=True)
    p.add_argument("--nav", type=Path, required=True)
    p.add_argument("--semantics", type=Path, required=True)
    p.add_argument("--lane-network", type=Path)
    p.add_argument("--outdoor-places", type=Path)
    p.add_argument(
        "--config", type=Path, help="recipe config; lane_network, outdoor_places and life.unavailable are read from it"
    )
    p.add_argument("--capabilities", type=Path, help="JSON object overriding the derived capabilities")
    p.add_argument("--sources", type=Path, help="JSON object: layer name -> {source, verified, method}")
    p.add_argument("--name")
    p.add_argument("--name-zh")
    p.add_argument("--notes")
    p.add_argument("--out", type=Path, required=True)
    d = sub.add_parser("describe", help="print the manifest")
    d.add_argument("dir", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "check":
        rep = check(a.dir)
        if a.out:
            a.out.write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        if a.json:
            print(json.dumps(rep, indent=1, ensure_ascii=False))
        else:
            print(f"{'PASS' if rep['pass'] else 'FAIL'} {rep.get('id')} {str(rep.get('content_version'))[:12]}")
            for e in rep["errors"]:
                print("error:", e)
            for w in rep["warnings"]:
                print("warning:", w)
            for k, v in rep["stats"].items():
                print(k, json.dumps(v, ensure_ascii=False))
        return 0 if rep["pass"] else 2
    if a.cmd == "pack":
        caps = json.loads(a.capabilities.read_text(encoding="utf-8")) if a.capabilities else None
        srcs = json.loads(a.sources.read_text(encoding="utf-8")) if a.sources else None
        doc = pack(
            a.out,
            world=a.world,
            nav=a.nav,
            semantics=a.semantics,
            lane_network=a.lane_network,
            outdoor_places=a.outdoor_places,
            config=a.config,
            capabilities=caps,
            sources=srcs,
            name=a.name,
            name_zh=a.name_zh,
            notes=a.notes,
        )
        print(f"packed {doc['id']} {doc['content_version'][:12]} -> {a.out}")
        rep = check(a.out)
        print("check", "PASS" if rep["pass"] else "FAIL", *rep["errors"])
        return 0 if rep["pass"] else 2
    print(json.dumps(load(a.dir), indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
