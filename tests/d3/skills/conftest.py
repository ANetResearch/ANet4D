"""tests/d3/skills：技能词表、执行器与指挥网关。研究者 SDK 源码目录加入路径，供动作构造的对拍用例导入 `anet4d.agent`。"""

import sys
from pathlib import Path

SDK_SRC = Path(__file__).resolve().parents[3] / "tools" / "sdk" / "anet4d" / "src"
if str(SDK_SRC) not in sys.path:
    sys.path.insert(0, str(SDK_SRC))
