"""data/raw/ の各ファイルが CHECKSUMS.txt と一致するか検証する。

data/raw/ 直下の CHECKSUMS.txt に加えて、取得フォルダ(prices_yfinance/<日付>/)内の CHECKSUMS.txt も検証する。

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
    bad = total = 0
    for sums in sorted(RAW.rglob("CHECKSUMS.txt")):
        base = sums.parent
        for line in sums.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, rel = line.split(None, 1)
            target = base / rel.strip()
            total += 1
            if not target.exists():
                print(f"MISSING   {target.relative_to(RAW)}")
                bad += 1
            elif md5(target) != expected:
                print(f"MISMATCH  {target.relative_to(RAW)}")
                bad += 1
    print(f"{total - bad}/{total} files OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
