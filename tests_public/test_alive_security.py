from __future__ import annotations

import hashlib
import http.client
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from kev.uc51a3.alive import AliveStore
from kev.uc51a3.server import MAX_BODY_BYTES, make_server


def _request(
    port: int,
    method: str,
    path: str,
    *,
    body: bytes = b"",
    host: str = "127.0.0.1",
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.putrequest(method, path, skip_host=True)
    connection.putheader("Host", host)
    supplied = {name.casefold() for name in (headers or {})}
    for name, value in (headers or {}).items():
        connection.putheader(name, value)
    if body and "content-length" not in supplied:
        connection.putheader("Content-Length", str(len(body)))
    connection.endheaders(body if body else None)
    response = connection.getresponse()
    response_body = response.read()
    response_headers = {name.casefold(): value for name, value in response.getheaders()}
    status = response.status
    connection.close()
    return status, response_headers, response_body


@pytest.fixture
def local_server(tmp_path: Path):
    server = make_server(str(tmp_path / "state"), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_ledger_detects_content_tampering_and_preserved_anchor_detects_truncation(
    tmp_path: Path,
):
    tampered = AliveStore(tmp_path / "tampered")
    tampered.append_event("FIRST", {"value": 1})
    tampered.append_event("SECOND", {"value": 2})
    records = [
        json.loads(line) for line in tampered.ledger_path.read_text().splitlines()
    ]
    records[0]["data"]["value"] = 999
    tampered.ledger_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    verification = tampered.verify_ledger()
    assert verification["valid"] is False
    assert verification["reason"] == "hash"
    assert verification["line"] == 1

    truncated = AliveStore(tmp_path / "truncated")
    truncated.append_event("FIRST", {"value": 1})
    truncated.append_event("SECOND", {"value": 2})
    first_line = truncated.ledger_path.read_text(encoding="utf-8").splitlines()[0]
    truncated.ledger_path.write_text(first_line + "\n", encoding="utf-8")

    verification = truncated.verify_ledger()
    assert verification["valid"] is False
    assert verification["reason"] == "head_anchor"


def test_ledger_anchor_detects_formatting_only_byte_length_tamper(tmp_path: Path):
    store = AliveStore(tmp_path / "formatting-tamper")
    store.append_event("FIRST", {"value": 1})
    with store.ledger_path.open("ab") as handle:
        handle.write(b"\n")

    verification = store.verify_ledger()
    assert verification["valid"] is False
    assert verification["reason"] == "head_anchor"


def test_same_size_ledger_tamper_blocks_a_later_append(tmp_path: Path):
    store = AliveStore(tmp_path / "same-size-tamper")
    store.append_event("FIRST", {"value": 1})
    original = store.ledger_path.read_bytes()
    tampered = original.replace(b'"value": 1', b'"value": 2')
    assert tampered != original
    assert len(tampered) == len(original)
    store.ledger_path.write_bytes(tampered)

    with pytest.raises(RuntimeError, match="ledger is invalid"):
        store.append_event("SECOND", {"value": 2})
    assert store.ledger_path.read_bytes() == tampered


def test_concurrent_writers_preserve_every_event_and_one_linear_hash_chain(
    tmp_path: Path,
):
    stores = [AliveStore(tmp_path / "shared") for _ in range(4)]
    event_count = 40

    def append(index: int):
        return stores[index % len(stores)].append_event("CONCURRENT", {"index": index})

    with ThreadPoolExecutor(max_workers=10) as executor:
        events = list(executor.map(append, range(event_count)))

    verification = stores[0].verify_ledger()
    assert verification["valid"] is True
    assert verification["events"] == event_count
    assert sorted(event["sequence"] for event in events) == list(
        range(1, event_count + 1)
    )
    assert len({event["hash"] for event in events}) == event_count
    assert {event["data"]["index"] for event in events} == set(range(event_count))


def test_pending_state_event_recovers_a_crash_before_ledger_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    directory = tmp_path / "recover-before-append"
    store = AliveStore(directory)

    def interrupted_append(_kind, _data):
        raise OSError("simulated process interruption")

    monkeypatch.setattr(store, "_append_event_unlocked", interrupted_append)
    with pytest.raises(OSError, match="simulated"):
        store.remember_fact("codename", "Lumen")
    assert store.pending_state_event_path.is_file()

    recovered = AliveStore(directory)
    assert recovered.read()["facts"]["codename"]["value"] == "Lumen"
    verification = recovered.verify_ledger()
    assert verification["valid"] is True
    assert verification["events"] == 1
    assert not recovered.pending_state_event_path.exists()


def test_pending_state_event_does_not_duplicate_an_already_appended_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    directory = tmp_path / "recover-after-append"
    store = AliveStore(directory)
    real_append = store._append_event_unlocked

    def interrupted_after_append(kind, data):
        real_append(kind, data)
        raise OSError("simulated interruption after append")

    monkeypatch.setattr(store, "_append_event_unlocked", interrupted_after_append)
    with pytest.raises(OSError, match="after append"):
        store.remember_fact("codename", "Lumen")
    assert store.pending_state_event_path.is_file()

    recovered = AliveStore(directory)
    verification = recovered.verify_ledger()
    assert verification["valid"] is True
    assert verification["events"] == 1
    assert not recovered.pending_state_event_path.exists()


def test_pending_state_event_recovers_a_crash_truncated_ledger_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    directory = tmp_path / "recover-partial-append"
    store = AliveStore(directory)
    store.append_event("BASE", {"value": "preserved"})

    def interrupted_during_append(_kind, _data):
        with store.ledger_path.open("ab") as handle:
            handle.write(b'{"schema":"kev.alive-event.v2","sequence":2')
            handle.flush()
        raise OSError("simulated interruption during ledger append")

    monkeypatch.setattr(store, "_append_event_unlocked", interrupted_during_append)
    with pytest.raises(OSError, match="during ledger append"):
        store.remember_fact("codename", "Lumen")
    assert store.pending_state_event_path.is_file()

    recovered = AliveStore(directory)
    assert recovered.read()["facts"]["codename"]["value"] == "Lumen"
    verification = recovered.verify_ledger()
    assert verification["valid"] is True
    assert verification["events"] == 2
    events = [
        json.loads(line)
        for line in recovered.ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["kind"] for event in events] == ["BASE", "FACT_ADDED"]
    assert not recovered.pending_state_event_path.exists()


def test_reviewed_lesson_export_requires_explicit_review_and_preserves_provenance(
    tmp_path: Path,
):
    store = AliveStore(tmp_path / "state")
    draft = store.add_lesson("CONSTRAINT", "Never mutate production.", source="test")
    assert draft["status"] == "DRAFT"

    draft_export = tmp_path / "draft-export.jsonl"
    before_review = store.export_lessons(draft_export)
    assert before_review["count"] == 0
    assert draft_export.read_text(encoding="utf-8") == ""

    with pytest.raises(ValueError, match="reviewer and permission"):
        store.review_lesson(draft["id"], "", "training approved")

    reviewed = store.review_lesson(
        draft["id"],
        reviewed_by="human-reviewer",
        permission="approved for local challenger training",
    )
    assert reviewed["status"] == "REVIEWED"
    assert reviewed["reviewed_by"] == "human-reviewer"
    assert reviewed["reviewed_at"]
    assert reviewed["permission"] == "approved for local challenger training"

    with pytest.raises(ValueError, match="already reviewed"):
        store.review_lesson(draft["id"], "second-reviewer", "approved")

    reviewed_export = tmp_path / "reviewed-export.jsonl"
    export = store.export_lessons(reviewed_export)
    rows = [
        json.loads(line)
        for line in reviewed_export.read_text(encoding="utf-8").splitlines()
    ]
    assert export["count"] == 1
    assert export["sha256"] == hashlib.sha256(reviewed_export.read_bytes()).hexdigest()
    assert rows == [reviewed]
    assert store.verify_ledger()["valid"] is True


@pytest.mark.parametrize(
    "session_id",
    [
        "",
        ".",
        "..",
        "../escape",
        "..\\escape",
        "/absolute/path",
        "nested/session",
        "C:\\absolute\\path",
        "a" * 129,
    ],
)
def test_session_ids_cannot_escape_the_session_directory(
    tmp_path: Path, session_id: str
):
    store = AliveStore(tmp_path / "state")
    with pytest.raises(ValueError, match="invalid session id|escapes"):
        store.append_message(session_id, "user", "must not be written")

    assert not (tmp_path / "escape.jsonl").exists()


def test_valid_session_id_remains_inside_session_directory(tmp_path: Path):
    store = AliveStore(tmp_path / "state")
    path = store.session_path("session.Valid_123-abc")
    assert path.parent == store.sessions.resolve()
    store.append_message("session.Valid_123-abc", "user", "hello")
    assert path.is_file()


def test_http_surface_blocks_path_escape_nonlocal_origins_and_non_json_writes(
    local_server: int,
):
    status, headers, body = _request(local_server, "GET", "/")
    assert status == 200
    assert b"KEV Alive" in body
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["x-frame-options"] == "DENY"
    assert headers["cross-origin-resource-policy"] == "same-origin"
    assert headers["referrer-policy"] == "no-referrer"
    assert headers["cache-control"] == "no-store"
    assert "default-src 'self'" in headers["content-security-policy"]
    assert "frame-ancestors 'none'" in headers["content-security-policy"]

    for path in ("/../../README.md", "/%2e%2e/%2e%2e/README.md", "/..\\..\\README.md"):
        status, _, body = _request(local_server, "GET", path)
        assert status == 404, (path, body)

    status, _, _ = _request(local_server, "GET", "/api/status", host="attacker.invalid")
    assert status == 403

    status, _, _ = _request(
        local_server,
        "GET",
        "/api/status",
        headers={"Origin": "https://attacker.invalid"},
    )
    assert status == 403

    payload = json.dumps({"intent": "GOAL", "text": "poison the reviewed set"}).encode()
    status, _, response = _request(
        local_server,
        "POST",
        "/api/teach",
        body=payload,
        headers={"Content-Type": "text/plain"},
    )
    assert status == 415
    assert json.loads(response)["error"] == "JSON_REQUIRED"

    status, _, response = _request(
        local_server,
        "POST",
        "/api/teach",
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(MAX_BODY_BYTES + 1),
        },
    )
    assert status == 413
    assert json.loads(response)["error"] == "BODY_TOO_LARGE"
