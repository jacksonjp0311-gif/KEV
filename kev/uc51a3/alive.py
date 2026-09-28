from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from kev.model_runtime import (
    configure_checkpoint_calibration,
    load_proposal_model_bytes,
    predict_proposal,
)
from kev.uc51a2.semantic_breadth import (
    INTENTS,
    load_bytes as load_semantic_bytes,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "models" / "registry.json"
VERSION = "0.52.0-alpha.1"
STATE_GROUPS = {
    "GOAL": "goals",
    "CONSTRAINT": "constraints",
    "OBSERVATION": "observations",
    "PREDICTION": "predictions",
}
_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


class IncumbentCompareAndSwapError(RuntimeError):
    """Qualification lost a race with a newer active incumbent."""

    def __init__(
        self,
        *,
        expected: Mapping[str, Any],
        current: Mapping[str, Any],
    ) -> None:
        self.expected = dict(expected)
        self.current = dict(current)
        super().__init__(
            "active incumbent changed after evaluation; refusing stale activation"
        )


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_state_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "KEV" / "alive"
    return Path.home() / ".kev" / "alive"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _observation_matches_receipt(
    observation: Mapping[str, Any], receipt: Mapping[str, Any]
) -> bool:
    """Bind a VERIFIED observation to the exact successful tool receipt.

    A receipt hash authenticates the receipt, not an arbitrary frame that
    merely cites it. Every supplied receipt reference and tool identifier
    must agree, and the observation result must be the receipt's output.
    """

    provenance = observation.get("provenance", {})
    if not isinstance(provenance, Mapping):
        return False
    source = observation.get("source", {})
    if not isinstance(source, Mapping):
        return False

    claimed_hashes = [
        value
        for value in (
            observation.get("receipt_hash"),
            provenance.get("receipt_hash"),
        )
        if value is not None
    ]
    receipt_hash = receipt.get("receipt_hash")
    if not claimed_hashes or any(value != receipt_hash for value in claimed_hashes):
        return False

    tool_references = [
        value
        for value in (
            observation.get("tool"),
            source.get("tool"),
            provenance.get("source_id"),
        )
        if value is not None
    ]
    receipt_tool = receipt.get("tool")
    if not tool_references or any(value != receipt_tool for value in tool_references):
        return False

    observed_values: list[Any] = []
    if "value" in observation:
        observed_values.append(observation["value"])
    slots = observation.get("slots", {})
    if isinstance(slots, Mapping) and "result" in slots:
        result = slots["result"]
        if isinstance(result, Mapping) and "value" in result:
            observed_values.append(result["value"])
        else:
            observed_values.append(result)
    if not observed_values:
        return False
    try:
        expected_hash = _sha256_json(receipt.get("output"))
        return all(_sha256_json(value) == expected_hash for value in observed_values)
    except (TypeError, ValueError):
        return False


def _atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _norm_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold()).strip(" .?!")


def _default_model() -> tuple[str | None, str | None]:
    if not REGISTRY_PATH.exists():
        return None, None
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    active = registry.get("active", {})
    raw_path = active.get("path")
    if not raw_path:
        return None, None
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    return str(path.resolve()), active.get("sha256")


class _FileMutex:
    """Small cross-process mutex used in addition to the in-process RLock."""

    def __init__(self, path: Path, timeout: float = 30.0) -> None:
        self.path = path
        self.timeout = timeout
        self.handle: Any = None

    def __enter__(self) -> "_FileMutex":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        self.handle.seek(0, os.SEEK_END)
        if self.handle.tell() == 0:
            self.handle.write(b"0")
            self.handle.flush()
        deadline = time.monotonic() + self.timeout
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    self.handle.seek(0)
                    msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        self.handle.close()
                        raise TimeoutError("timed out acquiring KEV state lock")
                    time.sleep(0.01)
        else:
            import importlib

            fcntl: Any = importlib.import_module("fcntl")
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.handle is None:
            return
        if os.name == "nt":
            import msvcrt

            self.handle.seek(0)
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import importlib

            fcntl: Any = importlib.import_module("fcntl")
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


class AliveStore:
    def __init__(self, directory: str | Path | None = None) -> None:
        self.dir = Path(directory or default_state_dir()).expanduser().resolve()
        self.state_path = self.dir / "state.json"
        self.ledger_path = self.dir / "ledger.jsonl"
        self.ledger_head_path = self.dir / "ledger-head.json"
        self.pending_state_event_path = self.dir / "pending-state-event.json"
        self.sessions = self.dir / "sessions"
        self.lock_path = self.dir / ".state.lock"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.sessions.mkdir(exist_ok=True)
        with _LOCKS_GUARD:
            self._thread_lock = _LOCKS.setdefault(str(self.dir), threading.RLock())
        with self._mutation_lock():
            if not self.state_path.exists():
                model_path, model_hash = _default_model()
                _atomic_json(
                    self.state_path, self._initial_state(model_path, model_hash)
                )
            if self.pending_state_event_path.exists():
                self._recover_pending_state_event_unlocked()
            state = self._read_unlocked()
            migrated = self._migrate(state)
            if migrated != state:
                self._commit_state_event_unlocked(
                    migrated, "STATE_MIGRATED", {"version": VERSION}
                )

    @staticmethod
    def _initial_state(
        model_path: str | None, model_hash: str | None
    ) -> dict[str, Any]:
        timestamp = now()
        return {
            "schema": "kev.alive-state.v2",
            "version": VERSION,
            "identity": {
                "name": "KEV",
                "description": "local teachable evidence-grounded research intelligence",
            },
            "facts": {},
            "typed_memory": [],
            "goals": [],
            "constraints": [],
            "observations": [],
            "predictions": [],
            "relations": [],
            "reviewed_lessons": [],
            "prediction_scores": [],
            "active_session": None,
            "semantic_model": model_path,
            "semantic_model_sha256": model_hash,
            "semantic_model_generation": 0,
            "model_history": [],
            "created_at": timestamp,
            "updated_at": timestamp,
        }

    @contextmanager
    def _mutation_lock(self) -> Iterator[None]:
        with self._thread_lock:
            with _FileMutex(self.lock_path):
                yield

    def _read_unlocked(self) -> dict[str, Any]:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _migrate(self, state: dict[str, Any]) -> dict[str, Any]:
        migrated = dict(state)
        migrated["schema"] = "kev.alive-state.v2"
        migrated["version"] = VERSION
        for key in (
            "typed_memory",
            "goals",
            "constraints",
            "observations",
            "predictions",
            "relations",
            "reviewed_lessons",
            "prediction_scores",
            "model_history",
        ):
            migrated.setdefault(key, [])
        migrated.setdefault("facts", {})
        migrated.setdefault("active_session", None)
        migrated.setdefault("semantic_model_sha256", None)
        # Generation is a monotonic compare-and-swap token for model
        # activation.  Existing v2 states begin at zero deterministically;
        # future qualifications increment it while holding the mutation lock.
        migrated.setdefault("semantic_model_generation", 0)
        # Earlier typed records remain evidence but are not silently promoted
        # into structured frames.
        return migrated

    def read(self) -> dict[str, Any]:
        with self._thread_lock:
            return self._read_unlocked()

    def _write_unlocked(self, state: dict[str, Any]) -> None:
        state["updated_at"] = now()
        _atomic_json(self.state_path, state)

    def _ledger_transaction_unlocked(
        self, transaction_id: str
    ) -> dict[str, Any] | None:
        if not self.ledger_path.exists():
            return None
        with self.ledger_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event = json.loads(line)
                transaction = event.get("data", {}).get("state_transaction", {})
                if transaction.get("id") == transaction_id:
                    return event
        return None

    def _commit_state_event_unlocked(
        self, state: dict[str, Any], kind: str, data: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Durably journal a state projection and its matching ledger event.

        This is recoverable rather than a claim of cross-file atomicity: a
        process crash leaves the journal, and the next store open completes
        the same state/event pair exactly once.
        """

        base_ledger_events, base_ledger_head = self._ledger_head_unlocked()
        base_ledger_size = (
            self.ledger_path.stat().st_size if self.ledger_path.exists() else 0
        )
        state["updated_at"] = now()
        transaction_id = str(uuid.uuid4())
        state_hash = _sha256_json(state)
        event_data = json.loads(
            json.dumps(dict(data), ensure_ascii=False, allow_nan=False)
        )
        event_data["state_transaction"] = {
            "id": transaction_id,
            "state_sha256": state_hash,
        }
        journal = {
            "schema": "kev.pending-state-event.v1",
            "transaction_id": transaction_id,
            "state_sha256": state_hash,
            "state": state,
            "event_kind": str(kind),
            "event_data": event_data,
            "base_ledger_events": base_ledger_events,
            "base_ledger_head": base_ledger_head,
            "base_ledger_size": base_ledger_size,
        }
        _atomic_json(self.pending_state_event_path, journal)
        _atomic_json(self.state_path, state)
        event = self._append_event_unlocked(kind, event_data)
        self.pending_state_event_path.unlink()
        return event

    def _recover_pending_state_event_unlocked(self) -> None:
        journal = json.loads(self.pending_state_event_path.read_text(encoding="utf-8"))
        if journal.get("schema") != "kev.pending-state-event.v1":
            raise RuntimeError("invalid pending state/event journal schema")
        state = journal.get("state")
        if not isinstance(state, dict) or _sha256_json(state) != journal.get(
            "state_sha256"
        ):
            raise RuntimeError("pending state/event journal hash mismatch")
        transaction_id = str(journal.get("transaction_id", ""))
        event_data = journal.get("event_data")
        transaction = (
            event_data.get("state_transaction", {})
            if isinstance(event_data, dict)
            else {}
        )
        if not transaction_id or transaction.get("id") != transaction_id:
            raise RuntimeError("pending state/event journal transaction mismatch")
        if transaction.get("state_sha256") != journal["state_sha256"]:
            raise RuntimeError("pending state/event journal state reference mismatch")

        self._repair_partial_ledger_tail_unlocked(journal)
        if (
            not self.state_path.exists()
            or _sha256_json(self._read_unlocked()) != journal["state_sha256"]
        ):
            _atomic_json(self.state_path, state)
        existing = self._ledger_transaction_unlocked(transaction_id)
        if existing is None:
            self._append_event_unlocked(str(journal["event_kind"]), event_data)
        elif (
            existing.get("kind") != journal["event_kind"]
            or existing.get("data") != event_data
        ):
            raise RuntimeError("pending state/event journal conflicts with ledger")
        else:
            verification = self._verify_ledger_unlocked()
            if not verification["valid"]:
                raise RuntimeError(
                    f"cannot recover state/event journal over invalid ledger: {verification.get('reason')}"
                )
            _atomic_json(
                self.ledger_head_path,
                {
                    "schema": "kev.ledger-head.v1",
                    "events": verification["events"],
                    "head": verification["head"],
                    "ledger_size": self.ledger_path.stat().st_size,
                    "updated_at": now(),
                },
            )
        self.pending_state_event_path.unlink()

    def _repair_partial_ledger_tail_unlocked(self, journal: Mapping[str, Any]) -> None:
        """Discard only a crash-truncated append after the journal's verified base."""

        base_size = journal.get("base_ledger_size")
        base_events = journal.get("base_ledger_events")
        base_head = journal.get("base_ledger_head")
        # Journals created before these anchors were added retain their prior
        # recovery behavior; guessing a truncation point would be unsafe.
        if (
            not isinstance(base_size, int)
            or isinstance(base_size, bool)
            or base_size < 0
            or not isinstance(base_events, int)
            or isinstance(base_events, bool)
            or base_events < 0
            or not isinstance(base_head, str)
        ):
            return

        payload = self.ledger_path.read_bytes() if self.ledger_path.exists() else b""
        if len(payload) <= base_size:
            if len(payload) < base_size:
                raise RuntimeError("ledger is shorter than pending transaction base")
            return
        verification = self._verify_ledger_bytes(payload)
        if verification["valid"]:
            return
        suffix = payload[base_size:]
        if verification.get("reason") != "parse" or suffix.endswith(b"\n"):
            return
        prefix_verification = self._verify_ledger_bytes(payload[:base_size])
        if (
            not prefix_verification["valid"]
            or prefix_verification["events"] != base_events
            or prefix_verification["head"] != base_head
        ):
            raise RuntimeError("pending transaction ledger base does not verify")

        with self.ledger_path.open("r+b") as handle:
            handle.truncate(base_size)
            handle.flush()
            os.fsync(handle.fileno())
        _atomic_json(
            self.ledger_head_path,
            {
                "schema": "kev.ledger-head.v1",
                "events": base_events,
                "head": base_head,
                "ledger_size": base_size,
                "updated_at": now(),
            },
        )

    def write(self, state: dict[str, Any]) -> None:
        with self._mutation_lock():
            self._commit_state_event_unlocked(
                state, "STATE_REPLACED", {"state_sha256": _sha256_json(state)}
            )

    def _ledger_head_unlocked(self) -> tuple[int, str]:
        verification = self._verify_ledger_unlocked()
        if not verification["valid"]:
            raise RuntimeError(f"ledger is invalid: {verification.get('reason')}")
        ledger_size = (
            self.ledger_path.stat().st_size if self.ledger_path.exists() else 0
        )
        if self.ledger_head_path.exists():
            try:
                head = json.loads(self.ledger_head_path.read_text(encoding="utf-8"))
                if (
                    head.get("ledger_size") != ledger_size
                    or head.get("events") != verification["events"]
                    or head.get("head") != verification["head"]
                ):
                    raise RuntimeError("ledger head anchor does not match ledger")
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                raise RuntimeError("ledger head anchor is invalid") from error
        return int(verification["events"]), str(verification["head"])

    def _append_event_unlocked(
        self, kind: str, data: Mapping[str, Any]
    ) -> dict[str, Any]:
        sequence, previous = self._ledger_head_unlocked()
        body = {
            "schema": "kev.alive-event.v2",
            "sequence": sequence + 1,
            "id": str(uuid.uuid4()),
            "time": now(),
            "kind": str(kind),
            "data": json.loads(json.dumps(data, ensure_ascii=False, allow_nan=False)),
            "prev_hash": previous,
        }
        body["hash"] = hashlib.sha256(_canonical_bytes(body)).hexdigest()
        line = (
            json.dumps(body, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        )
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        _atomic_json(
            self.ledger_head_path,
            {
                "schema": "kev.ledger-head.v1",
                "events": sequence + 1,
                "head": body["hash"],
                "ledger_size": self.ledger_path.stat().st_size,
                "updated_at": now(),
            },
        )
        return body

    def append_event(self, kind: str, data: Mapping[str, Any]) -> dict[str, Any]:
        with self._mutation_lock():
            return self._append_event_unlocked(kind, data)

    def recover_pending_state_event(self) -> dict[str, Any] | None:
        """Complete one interrupted state/event pair and report what recovered."""

        with self._mutation_lock():
            if not self.pending_state_event_path.exists():
                return None
            journal = json.loads(
                self.pending_state_event_path.read_text(encoding="utf-8")
            )
            summary = {
                "transaction_id": journal.get("transaction_id"),
                "event_kind": journal.get("event_kind"),
                "state_sha256": journal.get("state_sha256"),
            }
            self._recover_pending_state_event_unlocked()
            return summary

    def session_path(self, session_id: str) -> Path:
        if not isinstance(session_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", session_id
        ):
            raise ValueError("invalid session id")
        candidate = (self.sessions / f"{session_id}.jsonl").resolve()
        try:
            candidate.relative_to(self.sessions.resolve())
        except ValueError as error:
            raise ValueError("session path escapes state directory") from error
        return candidate

    def new_session(self) -> str:
        session_id = (
            time.strftime("%Y%m%d-%H%M%S", time.localtime())
            + "-"
            + uuid.uuid4().hex[:8]
        )
        with self._mutation_lock():
            state = self._read_unlocked()
            state["active_session"] = session_id
            self._commit_state_event_unlocked(
                state, "SESSION_CREATED", {"session_id": session_id}
            )
        return session_id

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        meta: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if role not in {"user", "assistant", "system"}:
            raise ValueError("invalid message role")
        record = {
            "schema": "kev.alive-message.v2",
            "time": now(),
            "role": role,
            "content": str(content),
            "meta": dict(meta or {}),
        }
        path = self.session_path(session_id)
        with self._mutation_lock():
            with path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
        return record

    def history(self, session_id: str, limit: int = 40) -> list[dict[str, Any]]:
        path = self.session_path(session_id)
        if not path.exists():
            return []
        records: deque[dict[str, Any]] = deque(maxlen=max(0, limit))
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        return list(records)

    def remember_fact(
        self, key: str, value: str, source: str = "user", correction: bool = False
    ) -> dict[str, Any]:
        normalized = _norm_key(key)
        clean_value = str(value).strip()
        if not normalized or not clean_value:
            raise ValueError("fact key and value are required")
        with self._mutation_lock():
            state = self._read_unlocked()
            old = state["facts"].get(normalized)
            revision = (old or {}).get("revision", 0) + 1
            record = {
                "key": normalized,
                "value": clean_value,
                "revision": revision,
                "source": source,
                "updated_at": now(),
                "supersedes": old["value"] if old else None,
                "supersedes_revision": old["revision"] if old else None,
            }
            state["facts"][normalized] = record
            self._commit_state_event_unlocked(
                state, "FACT_CORRECTED" if old or correction else "FACT_ADDED", record
            )
        return record

    @staticmethod
    def _frame_dict(frame: Any) -> dict[str, Any]:
        if hasattr(frame, "to_dict"):
            frame = frame.to_dict()
        if not isinstance(frame, Mapping):
            raise TypeError("frame must be a mapping or expose to_dict()")
        result = json.loads(
            json.dumps(dict(frame), ensure_ascii=False, allow_nan=False)
        )
        kind = result.get("kind")
        if kind not in INTENTS:
            raise ValueError(f"unknown frame kind: {kind}")
        if not isinstance(result.get("slots", {}), dict):
            raise ValueError("frame slots must be an object")
        if not isinstance(result.get("provenance", {}), dict):
            raise ValueError("frame provenance must be an object")
        if not isinstance(result.get("source", {}), dict):
            raise ValueError("frame source must be an object")
        result.setdefault("id", "frame-" + uuid.uuid4().hex)
        result.setdefault("status", "PROPOSED")
        result.setdefault("time", now())
        result.setdefault("slots", {})
        result.setdefault("provenance", {})
        return result

    def add_frames(self, frames: Iterable[Any]) -> list[dict[str, Any]]:
        records = [self._frame_dict(frame) for frame in frames]
        if not records:
            return []
        with self._mutation_lock():
            from kev.actions import verify_receipt

            receipts: dict[str, dict[str, Any]] = {}
            verified_requested = any(
                record.get("status") == "VERIFIED" for record in records
            )
            if verified_requested:
                verification = self._verify_ledger_unlocked()
                if not verification["valid"]:
                    raise RuntimeError(
                        f"cannot verify an observation against invalid ledger: {verification.get('reason')}"
                    )
            if verified_requested and self.ledger_path.exists():
                with self.ledger_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        event = json.loads(line)
                        if event.get("kind") == "ACTION_RECEIPT":
                            receipt = event.get("data", {})
                            receipt_hash = receipt.get("receipt_hash")
                            if isinstance(receipt_hash, str):
                                receipts[receipt_hash] = receipt
            state = self._read_unlocked()

            def receipt_reference(observation: Mapping[str, Any]) -> Any:
                provenance = observation.get("provenance", {})
                provenance_hash = (
                    provenance.get("receipt_hash")
                    if isinstance(provenance, Mapping)
                    else None
                )
                return observation.get("receipt_hash") or provenance_hash

            consumed_receipts = {
                str(receipt_reference(observation))
                for observation in state["observations"]
                if observation.get("status") == "VERIFIED"
                and receipt_reference(observation)
            }
            for record in records:
                if record.get("status") == "VERIFIED":
                    if record.get("kind") != "OBSERVATION":
                        raise ValueError("only observations may be VERIFIED")
                    receipt_hash = receipt_reference(record)
                    receipt = receipts.get(str(receipt_hash))
                    source_type = str(
                        record.get("source", {}).get(
                            "type", record.get("provenance", {}).get("source_type", "")
                        )
                    ).casefold()
                    if (
                        source_type != "tool"
                        or receipt is None
                        or receipt.get("status") != "SUCCEEDED"
                        or not verify_receipt(receipt)
                        or not _observation_matches_receipt(record, receipt)
                    ):
                        raise ValueError(
                            "VERIFIED observation requires an exactly matching successful "
                            "ledger-backed tool receipt"
                        )
                    if str(receipt_hash) in consumed_receipts:
                        raise ValueError(
                            "VERIFIED observation receipt has already been consumed"
                        )
                    consumed_receipts.add(str(receipt_hash))
                state["typed_memory"].append(record)
                group = STATE_GROUPS.get(record["kind"], "relations")
                state[group].append(record)
            self._commit_state_event_unlocked(
                state, "FRAMES_ADDED", {"frames": records}
            )
        return records

    def add_typed(
        self, kind: str, text: str, semantic: Mapping[str, Any]
    ) -> dict[str, Any]:
        record = {
            "kind": kind,
            "relation": kind,
            "text": text,
            "slots": {},
            "status": "REPORTED" if kind == "OBSERVATION" else "PROPOSED",
            "provenance": {
                "source_type": "legacy-language",
                "semantic": dict(semantic),
            },
        }
        return self.add_frames([record])[0]

    def add_lesson(
        self, intent: str, text: str, source: str = "user"
    ) -> dict[str, Any]:
        if intent not in INTENTS:
            raise ValueError("valid intent required")
        clean_text = str(text).strip()
        if not clean_text:
            raise ValueError("lesson text required")
        record = {
            "id": "lesson-" + uuid.uuid4().hex[:12],
            "intent": intent,
            "frame_kinds": [intent],
            "text": clean_text,
            "status": "DRAFT",
            "source": source,
            "created_at": now(),
            "reviewed_by": None,
            "reviewed_at": None,
            "permission": None,
        }
        with self._mutation_lock():
            state = self._read_unlocked()
            state["reviewed_lessons"].append(record)
            self._commit_state_event_unlocked(state, "LESSON_DRAFTED", record)
        return record

    def review_lesson(
        self, lesson_id: str, reviewed_by: str, permission: str
    ) -> dict[str, Any]:
        if not str(reviewed_by).strip() or not str(permission).strip():
            raise ValueError("reviewer and permission provenance are required")
        with self._mutation_lock():
            state = self._read_unlocked()
            record = next(
                (item for item in state["reviewed_lessons"] if item["id"] == lesson_id),
                None,
            )
            if record is None:
                raise KeyError("lesson not found")
            if record.get("status") == "REVIEWED":
                raise ValueError("lesson is already reviewed")
            record["status"] = "REVIEWED"
            record["reviewed_by"] = str(reviewed_by).strip()
            record["reviewed_at"] = now()
            record["permission"] = str(permission).strip()
            self._commit_state_event_unlocked(state, "LESSON_REVIEWED", record)
        return dict(record)

    def export_lessons(self, path: str | Path) -> dict[str, Any]:
        with self._thread_lock:
            state = self._read_unlocked()
            reviewed = [
                item
                for item in state["reviewed_lessons"]
                if item.get("status") == "REVIEWED"
            ]
        output = Path(path).expanduser().resolve()
        text = "".join(
            json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n"
            for item in reviewed
        )
        _atomic_text(output, text)
        digest = sha256_file(output)
        event = self.append_event(
            "LESSONS_EXPORTED",
            {"path": str(output), "sha256": digest, "count": len(reviewed)},
        )
        return {
            "path": str(output),
            "sha256": digest,
            "count": len(reviewed),
            "ledger_event": event["hash"],
        }

    def explicit_state(self) -> dict[str, Any]:
        state = self.read()
        return {
            "facts": state["facts"],
            "goals": state["goals"],
            "constraints": state["constraints"],
            "observations": state["observations"],
            "predictions": state["predictions"],
            "reviewed_lessons": [
                item
                for item in state["reviewed_lessons"]
                if item.get("status") == "REVIEWED"
            ],
        }

    def append_receipt(self, receipt: Mapping[str, Any]) -> dict[str, Any]:
        return self.append_event("ACTION_RECEIPT", receipt)

    def record_prediction_scores(
        self,
        scores: Iterable[Mapping[str, Any]],
        *,
        observation_id: str,
        receipt_hash: str,
    ) -> list[dict[str, Any]]:
        """Persist a replayable scoring pass against one later observation."""

        records: list[dict[str, Any]] = []
        for score in scores:
            record = json.loads(
                json.dumps(dict(score), ensure_ascii=False, allow_nan=False)
            )
            record["evaluated_at"] = now()
            record["observation_id"] = observation_id
            record["receipt_hash"] = receipt_hash
            records.append(record)
        if not records:
            return []
        with self._mutation_lock():
            state = self._read_unlocked()
            state["prediction_scores"].extend(records)
            self._commit_state_event_unlocked(
                state,
                "PREDICTIONS_SCORED",
                {
                    "observation_id": observation_id,
                    "receipt_hash": receipt_hash,
                    "scores": records,
                },
            )
        return records

    def record_model_decision(
        self,
        decision: str,
        data: Mapping[str, Any],
        active_model: Mapping[str, Any] | None = None,
        expected_incumbent: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if decision not in {"MODEL_REJECTED", "MODEL_QUALIFIED"}:
            raise ValueError("invalid model decision")
        if decision == "MODEL_QUALIFIED" and expected_incumbent is None:
            raise ValueError(
                "qualified model decision requires an expected incumbent snapshot"
            )
        with self._mutation_lock():
            state = self._read_unlocked()
            if expected_incumbent is not None:
                expected_path_value = expected_incumbent.get("path")
                expected_digest = str(expected_incumbent.get("sha256", ""))
                expected_generation = expected_incumbent.get("generation")
                if (
                    not expected_path_value
                    or not re.fullmatch(r"[0-9a-f]{64}", expected_digest)
                    or isinstance(expected_generation, bool)
                    or not isinstance(expected_generation, int)
                    or expected_generation < 0
                ):
                    raise ValueError(
                        "expected incumbent requires an exact path, SHA-256, and generation"
                    )
                expected_path = Path(str(expected_path_value)).resolve()
                current_path_value = state.get("semantic_model")
                current_digest = state.get("semantic_model_sha256")
                current_generation = state.get("semantic_model_generation")
                current_path = (
                    Path(str(current_path_value)).resolve()
                    if current_path_value
                    else None
                )
                current_file_digest = (
                    sha256_file(current_path)
                    if current_path is not None and current_path.is_file()
                    else None
                )
                current = {
                    "path": str(current_path) if current_path is not None else None,
                    "sha256": current_digest,
                    "generation": current_generation,
                    "observed_file_sha256": current_file_digest,
                }
                expected = {
                    "path": str(expected_path),
                    "sha256": expected_digest,
                    "generation": expected_generation,
                }
                if (
                    current_path != expected_path
                    or current_digest != expected_digest
                    or current_generation != expected_generation
                    or current_file_digest != expected_digest
                ):
                    raise IncumbentCompareAndSwapError(
                        expected=expected,
                        current=current,
                    )
            if decision == "MODEL_QUALIFIED":
                if not active_model:
                    raise ValueError("qualified model metadata required")
                path = Path(str(active_model["path"])).resolve()
                digest = str(active_model["sha256"])
                if not path.is_file() or sha256_file(path) != digest:
                    raise ValueError("qualified model hash does not match artifact")
                old = {
                    "path": state.get("semantic_model"),
                    "sha256": state.get("semantic_model_sha256"),
                    "generation": state.get("semantic_model_generation"),
                }
                state["model_history"].append(old)
                state["semantic_model"] = str(path)
                state["semantic_model_sha256"] = digest
                state["semantic_model_generation"] = (
                    int(state["semantic_model_generation"]) + 1
                )
                event = self._commit_state_event_unlocked(state, decision, data)
            else:
                event = self._append_event_unlocked(decision, data)
        return event

    @staticmethod
    def _verify_ledger_bytes(payload: bytes) -> dict[str, Any]:
        previous = "0" * 64
        count = 0
        try:
            text = payload.decode("utf-8")
            for line_number, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                claimed_hash = record.pop("hash")
                if record.get("sequence") != count + 1:
                    return {
                        "events": count,
                        "valid": False,
                        "reason": "sequence",
                        "line": line_number,
                    }
                if record.get("prev_hash") != previous:
                    return {
                        "events": count,
                        "valid": False,
                        "reason": "prev_hash",
                        "line": line_number,
                    }
                calculated = hashlib.sha256(_canonical_bytes(record)).hexdigest()
                if calculated != claimed_hash:
                    return {
                        "events": count,
                        "valid": False,
                        "reason": "hash",
                        "line": line_number,
                    }
                previous = claimed_hash
                count += 1
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
            return {
                "events": count,
                "valid": False,
                "reason": "parse",
                "message": str(error),
            }
        return {"events": count, "valid": True, "head": previous}

    def _verify_ledger_unlocked(self) -> dict[str, Any]:
        if not self.ledger_path.exists():
            return {"events": 0, "valid": True, "head": "0" * 64}
        try:
            return self._verify_ledger_bytes(self.ledger_path.read_bytes())
        except OSError as error:
            return {
                "events": 0,
                "valid": False,
                "reason": "parse",
                "message": str(error),
            }

    def verify_ledger(self) -> dict[str, Any]:
        with self._thread_lock:
            result = self._verify_ledger_unlocked()
            if result["valid"] and self.ledger_head_path.exists():
                try:
                    anchor = json.loads(
                        self.ledger_head_path.read_text(encoding="utf-8")
                    )
                    ledger_size = (
                        self.ledger_path.stat().st_size
                        if self.ledger_path.exists()
                        else 0
                    )
                    if (
                        anchor.get("head") != result["head"]
                        or anchor.get("events") != result["events"]
                        or anchor.get("ledger_size") != ledger_size
                    ):
                        return {**result, "valid": False, "reason": "head_anchor"}
                except (OSError, ValueError, json.JSONDecodeError):
                    return {**result, "valid": False, "reason": "head_anchor_parse"}
            return result


class AliveRuntime:
    def __init__(
        self,
        state_dir: str | Path | None = None,
        model_path: str | Path | None = None,
        tool_registry: Any | None = None,
    ) -> None:
        from kev.actions import ToolRegistry

        self.store = AliveStore(state_dir)
        self.tools = tool_registry if tool_registry is not None else ToolRegistry()
        if not isinstance(self.tools, ToolRegistry):
            raise TypeError("tool_registry must be a ToolRegistry")
        state = self.store.read()
        configured = model_path or state.get("semantic_model")
        self.model_path = Path(configured).resolve() if configured else None
        self.model = None
        self.model_hash = None
        self.model_error = None
        self.calibration_error = None
        if self.model_path and self.model_path.is_file():
            checkpoint_bytes = self.model_path.read_bytes()
            self.model_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
            expected = state.get("semantic_model_sha256")
            if expected and expected != self.model_hash:
                self.model_error = "MODEL_HASH_MISMATCH"
            else:
                self.model = load_proposal_model_bytes(
                    checkpoint_bytes, semantic_loader=load_semantic_bytes
                )
                _, self.calibration_error = configure_checkpoint_calibration(
                    self.model,
                    self.model_path,
                    checkpoint_bytes,
                )
        else:
            self.model_error = "MODEL_UNAVAILABLE"

    def plan_actions(self) -> tuple[Any, ...]:
        """Plan only from the six explicit state groups."""

        from kev.actions import StatePlanner

        return StatePlanner(self.tools).plan(self.store.explicit_state())

    def execute_action(self, action: Any, inputs: Any = None) -> dict[str, Any]:
        """Execute one typed action and commit its receipt before state use."""

        from kev.actions import ActionExecutor, score_predictions

        prior_state = self.store.explicit_state()
        outcome = ActionExecutor(self.tools, self.store.append_receipt).execute(
            action,
            inputs,
            state=prior_state,
        )
        observation = None
        scores: list[dict[str, Any]] = []
        if outcome.observation is not None:
            observation = self.store.add_frames([outcome.observation])[0]
            raw_scores = score_predictions(prior_state["predictions"], [observation])
            scores = self.store.record_prediction_scores(
                raw_scores,
                observation_id=observation["id"],
                receipt_hash=str(outcome.receipt["receipt_hash"]),
            )
        return {
            "schema": "kev.action-outcome.v1",
            "receipt": dict(outcome.receipt),
            "observation": observation,
            "prediction_scores": scores,
        }

    def _proposal(self, text: str) -> dict[str, Any]:
        if self.model is None:
            return {
                "proposals": [],
                "cardinality": 0,
                "score_status": "UNAVAILABLE",
                "model_error": self.model_error,
            }
        proposal = predict_proposal(self.model, text)
        proposal.pop("embedding", None)
        proposal["model_sha256"] = self.model_hash
        if self.calibration_error is not None:
            proposal["calibration_error"] = self.calibration_error
        return proposal

    def _extract_frames(self, message: str, session_id: str) -> list[Any]:
        # Imported lazily so deterministic state/fact operations remain
        # available even when optional semantic dependencies are unavailable.
        from kev.frames import extract_frames

        proposal = self._proposal(message)
        proposal_result = {
            "kinds": [item["kind"] for item in proposal.get("proposals", [])],
            "scores": {
                item["kind"]: item["score"] for item in proposal.get("proposals", [])
            },
            "cardinality": proposal.get("cardinality", 0),
        }
        return extract_frames(
            message,
            source_type="LANGUAGE",
            source_id=session_id,
            actor="user",
            model_hash=self.model_hash,
            proposal_result=proposal_result,
        )

    def chat(self, message: str, session_id: str | None = None) -> dict[str, Any]:
        message = str(message).strip()
        if not message:
            raise ValueError("message required")
        state = self.store.read()
        active_session = (
            session_id or state.get("active_session") or self.store.new_session()
        )
        # Validate caller-controlled session ids before touching the filesystem.
        self.store.session_path(active_session)
        self.store.append_message(active_session, "user", message)
        lowered = message.casefold()
        route = "FRAME_EXTRACTION"
        meta: dict[str, Any] = {
            "proposal": self._proposal(message),
            "model": str(self.model_path) if self.model_path else None,
            "model_sha256": self.model_hash,
        }

        if re.search(r"\bwho are you\b|\bwhat are you\b", lowered):
            route = "IDENTITY_QUERY"
            response = (
                "I am KEV, an experimental local cognitive-runtime research system. "
                "I preserve explicit state and evidence; I am not AGI."
            )
        elif re.search(r"\bwhat do you remember\b|\bmemory status\b", lowered):
            route = "MEMORY_STATUS"
            current = self.store.read()
            response = (
                f"State contains {len(current['facts'])} current facts, "
                f"{len(current['typed_memory'])} structured frames, and "
                f"{sum(item.get('status') == 'REVIEWED' for item in current['reviewed_lessons'])} "
                "reviewed lessons."
            )
        else:
            remember = re.match(
                r"\s*remember that\s+(.+?)\s+is\s+(.+?)[.!]?\s*$", message, re.I
            )
            correction = re.match(
                r"\s*(?:correction|update)\s*:\s*(.+?)\s+is\s+(.+?)[.!]?\s*$",
                message,
                re.I,
            )
            if remember or correction:
                route = "FACT_CORRECTION" if correction else "FACT_WRITE"
                match = remember or correction
                assert match is not None
                record = self.store.remember_fact(
                    match.group(1), match.group(2), correction=bool(correction)
                )
                response = f"Stored fact revision {record['revision']}: {record['key']} = {record['value']}."
                meta["memory_write"] = record
            else:
                query = re.match(
                    r"\s*(?:what is|what's|tell me)\s+(.+?)[?]?\s*$", message, re.I
                )
                fact = (
                    self.store.read()["facts"].get(_norm_key(query.group(1)))
                    if query
                    else None
                )
                if fact:
                    route = "FACT_READ"
                    response = fact["value"]
                    meta["memory_read"] = fact
                else:
                    frames = self._extract_frames(message, active_session)
                    records = self.store.add_frames(frames)
                    meta["frames"] = records
                    if records:
                        kinds = ", ".join(record["kind"] for record in records)
                        response = f"Recorded {len(records)} proposed structured frame(s): {kinds}."
                    else:
                        response = "No validated cognitive frame was extracted; nothing was added to state."

        meta["route"] = route
        self.store.append_message(active_session, "assistant", response, meta)
        event = self.store.append_event(
            "CHAT_TURN",
            {
                "session_id": active_session,
                "route": route,
                "frame_ids": [item["id"] for item in meta.get("frames", [])],
            },
        )
        return {
            "schema": "kev.alive-turn.v2",
            "version": VERSION,
            "session_id": active_session,
            "response": response,
            "route": route,
            "meta": meta,
            "authority": "NONE",
            "tool_execution": "NONE",
            "ledger_event": event["hash"],
        }
