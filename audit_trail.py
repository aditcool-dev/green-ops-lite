import hashlib
import json
import time
from dataclasses import dataclass, asdict, field
from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AuditEntry:
    seq:          int           # monotonic sequence number
    timestamp:    float         # Unix epoch, millisecond precision
    step:         int           # env step_count at time of action
    action:       str           # raw action string e.g. "increase_cooling(1)"
    action_type:  str           # "increase_cooling" | "decrease_load" | "migrate_jobs"
    rack_index:   str           # "0" / "1" / "2" / "0→2" for migrate

    # state snapshot BEFORE the action was applied
    temps_before: list
    loads_before: list
    power_before: float
    failed_fan:   bool

    # state snapshot AFTER physics step
    temps_after:  list
    loads_after:  list
    power_after:  float

    reward:       float
    done:         bool

    # chain integrity
    entry_hash:   str = field(default="")   # SHA-256 of this entry's payload
    prev_hash:    str = field(default="")   # SHA-256 of previous entry (genesis = "0"*64)


# ──────────────────────────────────────────────────────────────────────────────
# AuditTrail
# ──────────────────────────────────────────────────────────────────────────────

class AuditTrail:
    """
    Append-only, hash-chained ledger of every agent action.

    Chain integrity:
        entry_hash = SHA256( seq | timestamp | action | temps_before |
                             temps_after | reward | prev_hash )

    Verification walks the chain and recomputes every hash. Any mismatch
    means the ledger was tampered with after the fact.
    """

    GENESIS_HASH = "0" * 64

    def __init__(self):
        self._ledger: list[AuditEntry] = []
        self._session_start: float = time.time()

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_action(action_str: str) -> tuple[str, str]:
        """Returns (action_type, rack_index_str)."""
        try:
            atype = action_str.split("(")[0].strip()
            inner = action_str.split("(")[1].rstrip(")")
            return atype, inner.replace(",", "→")
        except Exception:
            return action_str, "?"

    @staticmethod
    def _hash_entry(e: AuditEntry, prev_hash: str) -> str:
        payload = json.dumps({
            "seq":          e.seq,
            "timestamp":    e.timestamp,
            "action":       e.action,
            "temps_before": [round(t, 3) for t in e.temps_before],
            "temps_after":  [round(t, 3) for t in e.temps_after],
            "reward":       round(e.reward, 6),
            "prev_hash":    prev_hash,
        }, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    # ── public API ────────────────────────────────────────────────────────────

    def record(
        self,
        step:       int,
        action:     str,
        obs_before: Any,          # Observation pydantic model or dict
        obs_after:  Any,
        reward:     float,
        done:       bool = False,
    ) -> AuditEntry:
        """Append one action to the ledger. Returns the entry."""

        def _temps(o):
            return list(o.rack_temp) if hasattr(o, "rack_temp") else o.get("rack_temp", [])
        def _loads(o):
            return list(o.cpu_load)  if hasattr(o, "cpu_load")  else o.get("cpu_load",  [])
        def _power(o):
            return float(o.power_cost if hasattr(o, "power_cost") else o.get("power_cost", 0))
        def _fan(o):
            return bool(o.failed_fan if hasattr(o, "failed_fan") else o.get("failed_fan", False))

        seq       = len(self._ledger)
        prev_hash = self._ledger[-1].entry_hash if self._ledger else self.GENESIS_HASH
        atype, rack_idx = self._parse_action(action)

        entry = AuditEntry(
            seq          = seq,
            timestamp    = time.time(),
            step         = step,
            action       = action,
            action_type  = atype,
            rack_index   = rack_idx,
            temps_before = [round(t, 2) for t in _temps(obs_before)],
            loads_before = [round(l, 3) for l in _loads(obs_before)],
            power_before = round(_power(obs_before), 3),
            failed_fan   = _fan(obs_before),
            temps_after  = [round(t, 2) for t in _temps(obs_after)],
            loads_after  = [round(l, 3) for l in _loads(obs_after)],
            power_after  = round(_power(obs_after), 3),
            reward       = round(reward, 6),
            done         = done,
            prev_hash    = prev_hash,
        )
        entry.entry_hash = self._hash_entry(entry, prev_hash)
        self._ledger.append(entry)
        return entry

    def verify(self) -> dict:
        """
        Walk the entire chain and verify every hash.
        Returns { valid: bool, entries: int, first_tampered_seq: int|None }
        """
        if not self._ledger:
            return {"valid": True, "entries": 0, "first_tampered_seq": None,
                    "message": "Empty ledger — nothing to verify"}

        prev_hash = self.GENESIS_HASH
        for entry in self._ledger:
            expected = self._hash_entry(entry, prev_hash)
            if expected != entry.entry_hash:
                return {
                    "valid":               False,
                    "entries":             len(self._ledger),
                    "first_tampered_seq":  entry.seq,
                    "message":             f"Chain broken at seq={entry.seq}. "
                                           f"Expected {expected[:16]}… got {entry.entry_hash[:16]}…",
                }
            prev_hash = entry.entry_hash

        return {
            "valid":               True,
            "entries":             len(self._ledger),
            "first_tampered_seq":  None,
            "message":             f"All {len(self._ledger)} entries verified. Chain intact.",
            "chain_tip":           self._ledger[-1].entry_hash[:32] + "…",
        }

    def export(self) -> dict:
        """Full ledger export for /audit endpoint."""
        result = []
        for e in self._ledger:
            d = asdict(e)
            d["timestamp_iso"] = time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.gmtime(e.timestamp)
            )
            d["hash_short"]    = e.entry_hash[:16] + "…"
            d["prev_short"]    = e.prev_hash[:16]  + "…"
            result.append(d)

        return {
            "session_start": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._session_start)
            ),
            "total_entries": len(self._ledger),
            "chain_tip":     self._ledger[-1].entry_hash if self._ledger else None,
            "ledger":        result,
        }

    def latest(self, n: int = 5) -> list:
        """Last n entries as dicts, most recent first."""
        return [
            {
                "seq":        e.seq,
                "step":       e.step,
                "action":     e.action,
                "hash_short": e.entry_hash[:16] + "…",
                "reward":     e.reward,
                "temps_after": e.temps_after,
            }
            for e in reversed(self._ledger[-n:])
        ]

    def clear(self):
        """Called on /reset — starts a fresh ledger for the new episode."""
        self._ledger.clear()
        self._session_start = time.time()

    def __len__(self):
        return len(self._ledger)
