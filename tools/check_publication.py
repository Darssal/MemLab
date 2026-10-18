"""Check a source-only repository boundary without printing matched content.

Run with PYTHONDONTWRITEBYTECODE=1. Reports must be outside the repository.
Default: strict working-tree scan. --staged: scan actual Git index blobs.
Neither mode can establish authorship or data provenance or inspect past commits.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile
import unicodedata
from typing import Any


@dataclass(frozen=True)
class Finding:
    path: str
    rule: str
    line: int | None = None


@dataclass
class ScanResult:
    files_checked: int
    findings: list[Finding]


def load_policy(path: Path) -> dict[str, Any]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy.get("schema_version") != 1:
        raise ValueError("Unsupported publication policy schema")
    return policy


def _is_link_or_reparse(path: Path) -> bool:
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & 0x400
    )


def _scope(relative: PurePosixPath, policy: dict[str, Any]) -> tuple[str, ...] | None:
    parts = relative.parts
    if len(parts) == 1:
        return () if parts[0] in policy["approved_root_files"] else None
    if parts[0] in policy["approved_top_level_directories"]:
        return parts
    for project in policy["approved_project_roots"]:
        prefix = PurePosixPath(project).parts
        if parts[:len(prefix)] == prefix:
            remaining = parts[len(prefix):]
            if len(remaining) == 1:
                return () if remaining[0] in policy["approved_project_root_names"] else None
            if remaining and remaining[0] in policy["approved_project_directories"]:
                return remaining
    return None


def _synthetic_scope(parts: tuple[str, ...]) -> bool:
    lowered = tuple(part.lower() for part in parts)
    return any(
        (parent == "examples" and child.startswith("synthetic"))
        or (parent == "tests" and child.startswith("fixtures"))
        for parent, child in zip(lowered, lowered[1:])
    )


def _allowed_directory(relative: PurePosixPath, policy: dict[str, Any]) -> bool:
    parts = relative.parts
    if parts[0] in policy["approved_top_level_directories"]:
        return True
    for project in policy["approved_project_roots"]:
        prefix = PurePosixPath(project).parts
        if len(parts) <= len(prefix) and parts == prefix[:len(parts)]:
            return True
        if parts[:len(prefix)] == prefix:
            remaining = parts[len(prefix):]
            return bool(remaining and remaining[0] in policy["approved_project_directories"])
    return False


def _allowed_file(relative: PurePosixPath, size: int, policy: dict[str, Any]) -> str | None:
    parts = _scope(relative, policy)
    if parts is None:
        return "PATH_NOT_ALLOWLISTED"
    suffix = relative.suffix.lower()
    if suffix in policy["forbidden_extensions"]:
        return "FORBIDDEN_FILE_TYPE"
    if size > policy["max_file_bytes"]:
        return "FILE_TOO_LARGE"
    if not parts:
        return None
    if suffix in policy["approved_source_extensions"]:
        return None
    if suffix == ".json" and parts[0] == "configs":
        return None
    if relative.name in {".gitignore", ".gitattributes", "CMakeLists.txt", "Makefile", "LICENSE", "NOTICE"}:
        return None
    if relative.name in {"requirements.txt", "requirements-dev.txt"}:
        return None
    if suffix in policy["synthetic_data_extensions"]:
        if not _synthetic_scope(parts):
            return "DATA_OUTSIDE_SYNTHETIC_SCOPE"
        if size > policy["max_synthetic_bytes"]:
            return "SYNTHETIC_FILE_TOO_LARGE"
        return None
    return "EXTENSION_NOT_ALLOWLISTED"


def _secret_patterns() -> list[tuple[str, re.Pattern[str]]]:
    # Prefixes are constructed so the checker does not contain token fixtures.
    return [
        ("GITHUB_TOKEN", re.compile("gh" + r"[pousr]_[A-Za-z0-9_]{20,}")),
        ("GITHUB_TOKEN", re.compile("github" + r"_pat_[A-Za-z0-9_]{20,}")),
        ("API_SECRET", re.compile("s" + r"k-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}")),
        ("CLOUD_ACCESS_KEY", re.compile(r"(?:AK" + r"IA|AS" + r"IA)[A-Z0-9]{16}")),
        ("PRIVATE_KEY", re.compile("-----BEGIN " + r"(?:RSA |EC |OPENSSH |DSA )?" + "PRIVATE KEY-----")),
        ("URL_CREDENTIALS", re.compile(r"https?://[^/\s:@]+:[^/\s@]+@", re.I)),
    ]


_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)\b(?:password|passwd|api[_-]?key|access[_-]?token|secret[_-]?key|client[_-]?secret|bearer[_-]?token)"
    r"[\"']?\s*(?:=|:)\s*[\"']([^\"'\\\r\n]{12,})[\"']"
)
_EMAIL = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])")
_PRIVATE_IP = re.compile(
    r"(?<![\d.])(?:10\.(?:\d{1,3}\.){2}\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|"
    r"172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(?![\d.])"
)
_ABSOLUTE_PATH = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/][^\"'\r\n`<>|]+|/(?:home|Users)/[^\"'\r\n`<>|]+)"
)
_PHONE = re.compile(
    r"(?i)(?:\+86[ -]?|\b(?:phone|mobile|tel(?:ephone)?)\s*[:=]?\s*[\"']?|"
    r"(?:手机|电话|手机号)\s*[:：=]?\s*[\"']?)(1[3-9][0-9]{9})(?![0-9])"
)


def _placeholder(value: str) -> bool:
    normalized = value.strip().lower()
    return bool(
        normalized.startswith(("your_", "your-", "replace_", "replace-", "dummy", "example", "placeholder", "test-", "test_", "not-a-", "<", "${"))
        or normalized in {"changeme", "none", "null", "password"}
    )


def _sensitive_lines(text: str, policy: dict[str, Any]) -> list[tuple[str, int]]:
    matches: set[tuple[str, int]] = set()
    patterns = _secret_patterns()
    allowed_emails = [re.compile(pattern, re.I) for pattern in policy.get("allowed_email_patterns", [])]
    prefixes = tuple(prefix.lower() for prefix in policy.get("allowed_absolute_path_prefixes", []))
    for number, line in enumerate(text.splitlines(), 1):
        for rule, pattern in patterns:
            if pattern.search(line):
                matches.add((rule, number))
        for assignment in _CREDENTIAL_ASSIGNMENT.finditer(line):
            if not _placeholder(assignment.group(1)):
                matches.add(("CREDENTIAL_ASSIGNMENT", number))
        for address in _EMAIL.finditer(line):
            if not any(pattern.fullmatch(address.group()) for pattern in allowed_emails):
                matches.add(("PERSONAL_EMAIL", number))
        for address in _PRIVATE_IP.finditer(line):
            if all(int(part) <= 255 for part in address.group().split(".")):
                matches.add(("PRIVATE_NETWORK_ADDRESS", number))
        for path in _ABSOLUTE_PATH.finditer(line):
            normalized = path.group().replace("\\", "/").lower()
            if not normalized.startswith(prefixes):
                matches.add(("PRIVATE_ABSOLUTE_PATH", number))
        if _PHONE.search(line):
            matches.add(("PERSONAL_PHONE", number))
    return sorted(matches, key=lambda item: (item[1], item[0]))


def scan_repository(root: Path, policy: dict[str, Any] | None = None) -> ScanResult:
    root = Path(root).absolute()
    if policy is None:
        policy = load_policy(Path(__file__).resolve().parents[1] / "publication-policy.json")
    findings: list[Finding] = []
    count = 0
    if _is_link_or_reparse(root):
        return ScanResult(0, [Finding(".", "SYMLINK_OR_REPARSE_POINT")])
    forbidden_dirs = {name.casefold() for name in policy["forbidden_directory_names"]}

    def visit(directory: Path) -> None:
        nonlocal count
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name.casefold())
        except OSError:
            findings.append(Finding(directory.relative_to(root).as_posix(), "UNREADABLE_DIRECTORY"))
            return
        for entry in entries:
            path = Path(entry.path)
            relative = PurePosixPath(path.relative_to(root).as_posix())
            rendered = relative.as_posix()
            try:
                if _is_link_or_reparse(path):
                    findings.append(Finding(rendered, "SYMLINK_OR_REPARSE_POINT"))
                    continue
                if entry.name.casefold() == ".git":
                    if len(relative.parts) != 1 or entry.name != ".git":
                        findings.append(Finding(rendered, "NESTED_GIT"))
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if entry.name.casefold() in forbidden_dirs:
                        findings.append(Finding(rendered, "FORBIDDEN_DIRECTORY"))
                        continue
                    if not _allowed_directory(relative, policy):
                        findings.append(Finding(rendered, "DIRECTORY_NOT_ALLOWLISTED"))
                        continue
                    visit(path)
                    continue
                if not entry.is_file(follow_symlinks=False):
                    findings.append(Finding(rendered, "NON_REGULAR_FILE"))
                    continue
                size = entry.stat(follow_symlinks=False).st_size
                count += 1
                rule = _allowed_file(relative, size, policy)
                if rule:
                    findings.append(Finding(rendered, rule))
                    continue
                payload = path.read_bytes()
                if b"\x00" in payload:
                    findings.append(Finding(rendered, "BINARY_CONTENT"))
                    continue
                try:
                    content = payload.decode("utf-8-sig")
                except UnicodeDecodeError:
                    findings.append(Finding(rendered, "NON_UTF8_TEXT"))
                    continue
                for sensitive_rule, number in _sensitive_lines(content, policy):
                    findings.append(Finding(rendered, sensitive_rule, number))
            except OSError:
                findings.append(Finding(rendered, "UNREADABLE_ENTRY"))

    visit(root)
    findings.sort(key=lambda finding: (finding.path, finding.line or 0, finding.rule))
    return ScanResult(count, findings)


def _git(root: Path, *arguments: str, payload: bytes | None = None) -> bytes:
    environment = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_NO_LAZY_FETCH="1")
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments], input=payload,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True,
            timeout=30, env=environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("Cannot read the local Git index or its objects") from error
    return result.stdout


def _safe_index_path(name: str) -> bool:
    if not name or name.startswith("/") or "\\" in name or ":" in name:
        return False
    if any(ord(character) < 32 or ord(character) == 127 for character in name):
        return False
    reserved = {"con", "prn", "aux", "nul"} | {
        prefix + str(number) for prefix in ("com", "lpt") for number in range(1, 10)
    }
    return all(
        part not in {"", ".", ".."}
        and not part.endswith((".", " "))
        and part.split(".", 1)[0].casefold() not in reserved
        for part in name.split("/")
    )


def _index_entries(root: Path) -> list[tuple[str, str, int, str]]:
    try:
        top = Path(_git(root, "rev-parse", "--show-toplevel").decode("utf-8").strip())
    except UnicodeDecodeError as error:
        raise ValueError("Git root must have a UTF-8 path") from error
    if top.resolve() != root.resolve():
        raise ValueError("--root must be the Git repository root for --staged")
    entries = []
    for item in _git(root, "ls-files", "--stage", "-z").split(b"\0"):
        if not item:
            continue
        try:
            metadata, raw_name = item.split(b"\t", 1)
            mode, object_id, stage = metadata.decode("ascii").split()
            if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", object_id):
                raise ValueError("Invalid object identifier")
            name = raw_name.decode("utf-8")
            entries.append((mode, object_id, int(stage), name))
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError("The Git index contains an unsupported entry") from error
    return entries


def _object_sizes(root: Path, identifiers: list[str]) -> dict[str, int | None]:
    if not identifiers:
        return {}
    raw = _git(
        root, "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)",
        payload=("\n".join(identifiers) + "\n").encode("ascii"),
    )
    sizes: dict[str, int | None] = {}
    for line in raw.decode("ascii").splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[1] == "blob" and fields[2].isdigit():
            sizes[fields[0]] = int(fields[2])
        elif fields:
            sizes[fields[0]] = None
    return sizes


def _staged_policy(root: Path, entries: list[tuple[str, str, int, str]]) -> dict[str, Any]:
    matches = [entry for entry in entries if entry[3] == "publication-policy.json"]
    if len(matches) != 1 or matches[0][0] not in {"100644", "100755"} or matches[0][2] != 0:
        raise ValueError("Stage a regular publication-policy.json before using --staged")
    identifier = matches[0][1]
    size = _object_sizes(root, [identifier]).get(identifier)
    if size is None or size > 65536:
        raise ValueError("The staged publication policy is missing or too large")
    try:
        policy = json.loads(_git(root, "cat-file", "blob", identifier).decode("utf-8-sig"))
    except (UnicodeError, ValueError) as error:
        raise ValueError("The staged publication policy is invalid") from error
    if not isinstance(policy, dict) or policy.get("schema_version") != 1:
        raise ValueError("Unsupported staged publication policy schema")
    return policy


def scan_staged(root: Path, policy: dict[str, Any] | None = None) -> ScanResult:
    """Scan complete index contents, never a working-tree substitute for a blob."""
    root = Path(root).absolute()
    entries = _index_entries(root)
    if policy is None:
        policy = _staged_policy(root, entries)
    findings: set[Finding] = set()
    candidates: list[tuple[str, str]] = []
    count = 0
    forbidden = {name.casefold() for name in policy["forbidden_directory_names"]}
    for mode, identifier, stage, name in entries:
        # Escape control characters in a rejected filename, never interpret them.
        rendered = json.dumps(name, ensure_ascii=False)[1:-1]
        if not _safe_index_path(name):
            findings.add(Finding(rendered, "UNSAFE_INDEX_PATH"))
            continue
        if stage != 0:
            findings.add(Finding(name, "UNMERGED_INDEX"))
            continue
        if mode == "120000":
            findings.add(Finding(name, "SYMLINK_INDEX_ENTRY"))
            continue
        if mode == "160000":
            findings.add(Finding(name, "SUBMODULE_INDEX_ENTRY"))
            continue
        if mode not in {"100644", "100755"}:
            findings.add(Finding(name, "UNSUPPORTED_INDEX_MODE"))
            continue
        count += 1
        relative = PurePosixPath(name)
        blocked = False
        for depth, part in enumerate(relative.parts):
            prefix = PurePosixPath(*relative.parts[:depth + 1])
            if part.casefold() == ".git":
                findings.add(Finding(prefix.as_posix(), "NESTED_GIT"))
                blocked = True
                break
            if depth == len(relative.parts) - 1:
                break
            rule = "FORBIDDEN_DIRECTORY" if part.casefold() in forbidden else (
                None if _allowed_directory(prefix, policy) else "DIRECTORY_NOT_ALLOWLISTED"
            )
            if rule:
                findings.add(Finding(prefix.as_posix(), rule))
                blocked = True
                break
        if blocked:
            continue
        rule = _allowed_file(relative, 0, policy)
        if rule:
            findings.add(Finding(name, rule))
            continue
        candidates.append((name, identifier))
    # Reject portable-path collisions before a case-insensitive temporary
    # filesystem could replace one staged blob with another.
    keyed: dict[str, str] = {}
    ambiguous: set[str] = set()
    for name, _ in candidates:
        key = unicodedata.normalize("NFC", name).casefold()
        if key in keyed:
            ambiguous.update((name, keyed[key]))
        else:
            keyed[key] = name
    for name, _ in candidates:
        parts = name.split("/")
        for depth in range(1, len(parts)):
            parent = unicodedata.normalize("NFC", "/".join(parts[:depth])).casefold()
            if parent in keyed:
                ambiguous.update((name, keyed[parent]))
    for name in ambiguous:
        findings.add(Finding(name, "INDEX_PATH_COLLISION"))
    candidates = [(name, identifier) for name, identifier in candidates if name not in ambiguous]
    sizes = _object_sizes(root, list(dict.fromkeys(identifier for _, identifier in candidates)))
    with tempfile.TemporaryDirectory(prefix="publication-staged-") as temporary:
        materialized = Path(temporary)
        for name, identifier in candidates:
            size = sizes.get(identifier)
            if size is None:
                findings.add(Finding(name, "MISSING_OR_NON_BLOB_OBJECT"))
                continue
            rule = _allowed_file(PurePosixPath(name), size, policy)
            if rule:
                findings.add(Finding(name, rule))
                continue
            # The immutable object's size was checked before requesting its body.
            payload = _git(root, "cat-file", "blob", identifier)
            if len(payload) != size:
                findings.add(Finding(name, "OBJECT_SIZE_MISMATCH"))
                continue
            destination = materialized.joinpath(*PurePosixPath(name).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        findings.update(scan_repository(materialized, policy).findings)
    return ScanResult(count, sorted(findings, key=lambda item: (item.path, item.line or 0, item.rule)))


def _outside_repository(report: Path, root: Path) -> bool:
    try:
        report.resolve().relative_to(root.resolve())
        return False
    except ValueError:
        return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--staged", action="store_true", help="Check actual Git index blobs; ignored runtime files are outside this scope")
    parser.add_argument("--report", "--json-report", type=Path, help="JSON audit destination outside the repository")
    args = parser.parse_args(argv)
    root = args.root.absolute()
    if not root.is_dir():
        parser.error("Repository root must be an existing directory")
    report = args.report or root.parent / "audit" / "publication-scan.json"
    if not _outside_repository(report, root):
        parser.error("Audit reports must be outside the repository")
    try:
        if args.staged:
            policy = load_policy(args.policy) if args.policy else None
            result = scan_staged(root, policy)
        else:
            policy = load_policy(args.policy or root / "publication-policy.json")
            result = scan_repository(root, policy)
    except (OSError, ValueError):
        parser.error("Publication policy or local Git index is missing, invalid or unavailable")
    record = {
        "schema_version": 1,
        "files_checked": result.files_checked,
        "scope": "git_index" if args.staged else "working_tree",
        "findings": [asdict(finding) for finding in result.findings],
        "limitations": ["Git history is outside this scan", "Authorship and synthetic provenance require review"],
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Checked {result.files_checked} files; findings={len(result.findings)}")
    for finding in result.findings:
        location = finding.path + (f":{finding.line}" if finding.line else "")
        print(f"{location} [{finding.rule}]")
    return 1 if result.findings else 0


if __name__ == "__main__":
    sys.exit(main())
