"""data/raw/ の各ファイルが CHECKSUMS.txt と一致するか検証する。

使い方(リポジトリのルートから):
    python src/data/verify_raw.py
"""
import hashlib
import sys
from pathlib import Path

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    bad = 0
    lines = [l for l in (RAW / "CHECKSUMS.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    for line in lines:
        expected, rel = line.split(None, 1)
        target = RAW / rel.strip()
        if not target.exists():
            print(f"MISSING   {rel}")
            bad += 1
        elif md5(target) != expected:
            print(f"MISMATCH  {rel}")
            bad += 1
    print(f"{len(lines) - bad}/{len(lines)} files OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
