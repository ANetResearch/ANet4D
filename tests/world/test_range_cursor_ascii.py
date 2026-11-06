"""字节区间头与分页游标只认 ASCII 数字（RFC 9110 §14.1.1）；int() 额外接受的写法一律当作无效。"""

from __future__ import annotations

import base64

import pytest

from awr.api.problem import ApiProblem
from awr.api.rest.worlds import _offset
from awr.api.static import parse_range


@pytest.mark.parametrize("header, want", [
    ("bytes=0-9", (0, 10)),
    ("bytes=5-", (5, 100)),
    ("bytes=-10", (90, 100)),
    ("bytes=200-300", "416"),
    ("bytes=1_0-2_0", None),  # int("1_0") == 10
    ("bytes=+5-9", None),
    ("bytes=5 -9", None),
    ("bytes=²-9", None),
    ("bytes=\u0663-9", None),  # int("\u0663") == 3（阿拉伯-印度数字三）
    ("bytes=-1_0", None),
])
def test_parse_range_ascii_only(header, want):
    assert parse_range(header, 100) == want


def _cursor(raw: str) -> str:
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def test_cursor_round_trip():
    assert _offset(None) == 0 and _offset(_cursor("o0")) == 0 and _offset(_cursor("o250")) == 250


@pytest.mark.parametrize("raw", ["o1_0", "o+5", "o-5", "o \u0663", "o²", "o"])
def test_cursor_ascii_only(raw):
    with pytest.raises(ApiProblem):
        _offset(_cursor(raw))
