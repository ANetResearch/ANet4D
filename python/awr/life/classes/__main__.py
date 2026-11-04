"""``python -m awr.life.classes check RECIPE.json``: resolve and print the entity classes and population of a recipe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from awr.life.classes import EntityClassError, describe


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="awr.life.classes")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="resolve the classes and population a recipe asks for")
    c.add_argument("recipe", type=Path, nargs="?", help="recipe config JSON; omitted: the built-in catalog")
    c.add_argument("--people", type=int, default=24, help="legacy people count when the recipe has no population")
    a = ap.parse_args(argv)
    cfg = json.loads(a.recipe.read_text(encoding="utf-8")) if a.recipe else {}
    try:
        print(json.dumps(describe(cfg, a.people), ensure_ascii=False, indent=1))
    except EntityClassError as e:
        print("error:", e)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
