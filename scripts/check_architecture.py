"""Check the maintained file inventory against ARCHITECTURE.md (no hashes)."""
from pathlib import Path
import re
import os

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".agents", ".aws", ".codex", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}


def maintained_files():
    files = set()
    for base, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in names:
            path = Path(base) / name
            relative = path.relative_to(ROOT).as_posix()
            if relative.startswith("checkpoints/") and name.endswith(".pth"):
                continue
            files.add(relative)
    return files



def main():
    inventory = (ROOT / "ARCHITECTURE.md").read_text()
    listed = set(re.findall(r"^\| `([^`]+)` \|", inventory, re.M))
    local_only = set(re.findall(r"^\| `([^`]+)` \|.*\| 本机文档交付 \|$", inventory, re.M))
    actual = maintained_files()
    unknown, missing = sorted(actual - listed), sorted(listed - local_only - actual)
    for path in unknown:
        print("UNLISTED:", path)
    for path in missing:
        print("MISSING:", path)
    if unknown or missing:
        raise SystemExit(1)
    print(f"PASS: {len(listed - local_only)} deployment files listed; {len(actual & local_only)} local document(s) present.")


if __name__ == "__main__":
    main()
