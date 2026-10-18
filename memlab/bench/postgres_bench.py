from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg-uri", type=str, default=os.getenv("PG_URI", ""))
    ap.add_argument("--duration-s", type=int, default=300)
    ap.add_argument("--clients", type=int, default=64)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", type=str, default="logs/pgbench.txt")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pgbench = shutil.which("pgbench")
    if not pgbench:
        raise SystemExit("pgbench not found in PATH")

    init = subprocess.run([pgbench, "-i", args.pg_uri], capture_output=True, text=True, check=True)
    run = subprocess.run(
        [pgbench, "-T", str(args.duration_s), "-c", str(args.clients), "-j", str(args.threads), args.pg_uri],
        capture_output=True,
        text=True,
        check=True,
    )
    out.write_text(init.stdout + "\n" + run.stdout, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
