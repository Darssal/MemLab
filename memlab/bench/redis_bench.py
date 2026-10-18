from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6379)
    ap.add_argument("--clients", type=int, default=100)
    ap.add_argument("--requests", type=int, default=200000)
    ap.add_argument("--out", type=str, default="logs/redis_benchmark.csv")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = shutil.which("redis-benchmark")
    if not cmd:
        raise SystemExit("redis-benchmark not found in PATH")

    run = [
        cmd,
        "-h",
        args.host,
        "-p",
        str(args.port),
        "-c",
        str(args.clients),
        "-n",
        str(args.requests),
        "-t",
        "get,set",
        "--csv",
    ]
    proc = subprocess.run(run, capture_output=True, text=True, check=True)
    out.write_text(proc.stdout, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
