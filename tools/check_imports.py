"""Static check of the tutorial pipeline's imports. Needs no third-party package installed.

For every .py file in stockpred/ and pipeline/:
  - each `stockpred...` import (absolute or relative) must resolve to a file in the repo;
  - nothing may import the pre-2026 module names (utilW3, features_W3_old, _KEYS_DICT, CONFIG,
    Utils), which now live only under legacy/.

Exit code 1 lists every problem. Run from anywhere: python tools/check_imports.py
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKED = ["stockpred", "pipeline"]
OLD_NAMES = {"utilW3", "features_W3_old", "features_W3", "_KEYS_DICT", "CONFIG", "Utils", "LogRoot"}


def resolves(dotted: str) -> bool:
    """True if `a.b.c` is a module or package file under ROOT, or a name defined in one."""
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        base = ROOT.joinpath(*parts[:i])
        if base.with_suffix(".py").is_file() or (base / "__init__.py").is_file():
            return True
    return False


def package_of(path: Path) -> list:
    return list(path.relative_to(ROOT).parent.parts)


problems = []
files = [p for d in CHECKED for p in (ROOT / d).rglob("*.py")]
for path in files:
    rel = path.relative_to(ROOT)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError as e:
        problems.append(f"{rel}:{e.lineno}: syntax error")
        continue
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Import):
            targets = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative: from .x import y
                pkg = package_of(path)
                pkg = pkg[: len(pkg) - (node.level - 1)] if node.level > 1 else pkg
                mod = ".".join(pkg + ([node.module] if node.module else []))
                if not resolves(mod) and not all(resolves(f"{mod}.{a.name}") for a in node.names):
                    problems.append(f"{rel}:{node.lineno}: relative import '{'.' * node.level}{node.module or ''}' does not resolve")
                continue
            targets = [node.module] if node.module else []
        for t in targets:
            top = t.split(".")[0]
            if top in OLD_NAMES:
                problems.append(f"{rel}:{node.lineno}: imports old module '{t}' (moved to legacy/)")
            elif top == "stockpred" and not resolves(t):
                problems.append(f"{rel}:{node.lineno}: '{t}' does not resolve")

print(f"checked {len(files)} files in {', '.join(CHECKED)}")
if problems:
    print("\n".join(problems))
    sys.exit(1)
print("all stockpred imports resolve; no old module names")
