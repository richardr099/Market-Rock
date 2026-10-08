"""Files shared with NinjaTrader, plus the hash-chained ledger.

live/portfolio.txt     canonical text (fixed line order), read by NT8
live/portfolio.sha256  hex SHA-256 of the exact bytes of portfolio.txt
live/history/<sha>.txt every version ever made live (rollback source)
state/ledger.jsonl     append-only; entry.prev = sha256 of the previous line
state/portfolio.json   lifecycle state of every genome ever proposed
state/search.json      cumulative trial count (the DSR needs it: never reset)
state/config.json      YOUR settings (risk ceiling, limits, slots, halt)

portfolio.txt format:
  version=2
  va_pct=0.7
  n=<k>
  g<i>.id=<12 hex>          (must equal sha256(rule)[:12])
  g<i>.risk_usd=<float>
  g<i>.rule=<canonical genome text>

NT8 verifies the hash, the ids and every bound before loading and refuses
on any mismatch. Writes are temp-file + os.replace(); the text is replaced
before the hash, so a racing reader sees a mismatch, never a stale pair.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

from . import genome as G
from .strategy import VA_PCT

GENESIS = "0" * 64
MAX_GENOMES = 8
MAX_RISK_FILE = 5000.0

Entry = Tuple[str, G.Genome, float]


def canonical(entries: Sequence[Entry]) -> bytes:
    if len(entries) > MAX_GENOMES:
        raise ValueError(f"at most {MAX_GENOMES} live genomes")
    lines = ["version=2", f"va_pct={VA_PCT!r}", f"n={len(entries)}"]
    for i, (gid, g, risk) in enumerate(entries):
        G.validate(g)
        if gid != g.id:
            raise ValueError("genome id mismatch")
        risk = float(risk)
        if not 0.0 <= risk <= MAX_RISK_FILE:
            raise ValueError("risk out of range")
        lines += [f"g{i}.id={gid}", f"g{i}.risk_usd={risk!r}", f"g{i}.rule={g.text()}"]
    return ("\n".join(lines) + "\n").encode("ascii")


def parse(data: bytes) -> List[Entry]:
    kv = {}
    for line in data.decode("ascii").splitlines():
        k, v = line.split("=", 1)
        kv[k] = v
    if kv.get("version") != "2" or float(kv["va_pct"]) != VA_PCT:
        raise ValueError("unsupported portfolio file")
    out = []
    for i in range(int(kv["n"])):
        g = G.parse(kv[f"g{i}.rule"])
        out.append((kv[f"g{i}.id"], g, float(kv[f"g{i}.risk_usd"])))
    if canonical(out) != data:
        raise ValueError("non-canonical portfolio file")
    return out


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


def write_live(live_dir: Path, entries: Sequence[Entry]) -> str:
    data = canonical(entries)
    h = sha256(data)
    _atomic_write(live_dir / "history" / f"{h}.txt", data)
    _atomic_write(live_dir / "portfolio.txt", data)
    _atomic_write(live_dir / "portfolio.sha256", (h + "\n").encode())
    return h


def read_live(live_dir: Path) -> Tuple[List[Entry], str]:
    data = (live_dir / "portfolio.txt").read_bytes()
    want = (live_dir / "portfolio.sha256").read_text().strip()
    got = sha256(data)
    if got != want:
        raise ValueError(f"live portfolio hash mismatch: file={got} recorded={want}")
    return parse(data), got


def read_history(live_dir: Path, h: str) -> List[Entry]:
    data = (live_dir / "history" / f"{h}.txt").read_bytes()
    if sha256(data) != h:
        raise ValueError("history file corrupted")
    return parse(data)


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
