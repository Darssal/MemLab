from __future__ import annotations

import importlib
import socket


def check_import(name: str) -> tuple[str, bool, str]:
    try:
        importlib.import_module(name)
        return name, True, ""
    except Exception as exc:
        return name, False, str(exc)


def check_port(host: str, port: int, timeout: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def main() -> None:
    modules = ["yaml", "pydantic", "numpy", "matplotlib", "redis", "asyncpg", "httpx"]
    print("== Python dependencies ==")
    for m, ok, err in (check_import(x) for x in modules):
        status = "OK" if ok else f"MISSING ({err})"
        print(f"{m:12} {status}")

    print("\n== Service reachability ==")
    for host, port, name in [
        ("127.0.0.1", 6379, "redis"),
        ("127.0.0.1", 5432, "postgres"),
        ("127.0.0.1", 6333, "qdrant"),
    ]:
        print(f"{name:12} {'UP' if check_port(host, port) else 'DOWN'} ({host}:{port})")


if __name__ == "__main__":
    main()
