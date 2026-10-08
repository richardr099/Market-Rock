"""Files shared with NinjaTrader, plus the hash-chained promotion ledger.

live/params.json      canonical JSON (sorted keys, repr floats), read by NT8
live/params.sha256    hex SHA-256 of the exact bytes of params.json
live/history/<sha>.json  every version ever made live (rollback source)
state/ledger.jsonl    append-only; entry.prev = sha256 of the previous line
state/search.json     cumulative trial count + ES step size (DSR needs the
                      cumulative count, so this file must never be reset)
state/regimes.json    Bayesian regime posteriors + last processed trade time

NT8 verifies params.sha256 before every load and refuses to trade on a
mismatch, so a partially written or hand-edited file fails closed.
Writes go to a temp file then os.replace() (atomic on NTFS and POSIX);
params.json is replaced before params.sha256 so a reader racing the write sees
a hash mismatch (refuse) rather than a matching stale pair.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from . import params as P

GENESIS = "0" * 64


def canonical(p: Mapping[str, float]) -> bytes:
    v = P.validate(p)
    body = ",\n".join(f'  "{k}": {repr(float(v[k]))}' for k in sorted(v))
    return ("{\n" + body + "\n}\n").encode("ascii")


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_live(live_dir: Path, p: Mapping[str, float]) -> str:
    data = canonical(p)
    h = sha256(data)
    _atomic_write(live_dir / "history" / f"{h}.json", data)
    _atomic_write(live_dir / "params.json", data)
    _atomic_write(live_dir / "params.sha256", (h + "\n").encode())
    return h


def read_live(live_dir: Path) -> tuple[Dict[str, float], str]:
    data = (live_dir / "params.json").read_bytes()
    want = (live_dir / "params.sha256").read_text().strip()
    got = sha256(data)
    if got != want:
        raise ValueError(f"live params hash mismatch: file={got} recorded={want}")
    return P.validate(json.loads(data)), got


def read_history(live_dir: Path, h: str) -> Dict[str, float]:
    data = (live_dir / "history" / f"{h}.json").read_bytes()
    if sha256(data) != h:
        raise ValueError("history file corrupted")
    return P.validate(json.loads(data))


# --------------------------- ledger ---------------------------------------
def ledger_append(state_dir: Path, entry: Mapping) -> str:
    path = state_dir / "ledger.jsonl"
    prev = GENESIS
    if path.exists():
        lines = path.read_bytes().splitlines()
        if lines:
            prev = sha256(lines[-1])
    rec = dict(entry)
    rec["prev"] = prev
    line = json.dumps(rec, sort_keys=True, separators=(",", ":")).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab") as f:
        f.write(line + b"\n")
        f.flush()
        os.fsync(f.fileno())
    return sha256(line)


def ledger_verify(state_dir: Path) -> List[dict]:
    path = state_dir / "ledger.jsonl"
    if not path.exists():
        return []
    prev = GENESIS
    out = []
    for i, line in enumerate(path.read_bytes().splitlines()):
        rec = json.loads(line)
        if rec.get("prev") != prev:
            raise ValueError(f"ledger chain broken at line {i + 1}")
        prev = sha256(line)
        out.append(rec)
    return out


# --------------------------- small json state ------------------------------
def read_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def write_json(path: Path, obj) -> None:
    _atomic_write(path, (json.dumps(obj, sort_keys=True, indent=1) + "\n").encode())


def read_text_opt(path: Path) -> Optional[str]:
    return path.read_text() if path.exists() else None
