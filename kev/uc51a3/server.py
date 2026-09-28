from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .alive import AliveRuntime


ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "desktop" / "v051a3"
MAX_BODY_BYTES = 1_048_576


def make_server(state_dir: str | None, port: int = 39061) -> ThreadingHTTPServer:
    runtime = AliveRuntime(state_dir)
    store = runtime.store
    web_root = WEB.resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            return

        def end_headers(self) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-store")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'",
            )
            super().end_headers()

        def _local_request(self) -> bool:
            host = self.headers.get("Host", "").split(":", 1)[0].casefold()
            if host not in {"127.0.0.1", "localhost", "[::1]"}:
                return False
            origin = self.headers.get("Origin")
            if origin:
                origin_host = (urlsplit(origin).hostname or "").casefold()
                if origin_host not in {"127.0.0.1", "localhost", "::1"}:
                    return False
            return True

        def send_json(self, obj: object, code: int = 200) -> None:
            body = json.dumps(obj, sort_keys=True).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            if not self._local_request():
                return self.send_error(403)
            path = urlsplit(self.path).path
            if path == "/api/status":
                state = store.read()
                return self.send_json(
                    {
                        "version": state["version"],
                        "active_session": state["active_session"],
                        "facts": len(state["facts"]),
                        "typed_memory": len(state["typed_memory"]),
                        "reviewed_lessons": sum(
                            item.get("status") == "REVIEWED"
                            for item in state["reviewed_lessons"]
                        ),
                        "draft_lessons": sum(
                            item.get("status") == "DRAFT"
                            for item in state["reviewed_lessons"]
                        ),
                        "ledger": store.verify_ledger(),
                        "semantic_model": state.get("semantic_model"),
                        "semantic_model_sha256": state.get("semantic_model_sha256"),
                    }
                )
            if path == "/api/history":
                session_id = store.read().get("active_session")
                return self.send_json(
                    {
                        "session_id": session_id,
                        "messages": store.history(session_id, 80) if session_id else [],
                    }
                )

            relative = (
                "index.html"
                if path in {"/", "/index.html"}
                else unquote(path).lstrip("/")
            )
            candidate = (web_root / relative).resolve()
            try:
                candidate.relative_to(web_root)
            except ValueError:
                return self.send_error(404)
            if not candidate.is_file():
                return self.send_error(404)
            content_types = {
                ".html": "text/html; charset=utf-8",
                ".js": "text/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8",
            }
            body = candidate.read_bytes()
            self.send_response(200)
            self.send_header(
                "Content-Type",
                content_types.get(candidate.suffix, "application/octet-stream"),
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            if not self._local_request():
                return self.send_error(403)
            if self.headers.get_content_type() != "application/json":
                return self.send_json({"error": "JSON_REQUIRED"}, 415)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0 or length > MAX_BODY_BYTES:
                    return self.send_json({"error": "BODY_TOO_LARGE"}, 413)
                body = json.loads(self.rfile.read(length) or b"{}")
                path = urlsplit(self.path).path
                if path == "/api/chat":
                    return self.send_json(
                        runtime.chat(body.get("message", ""), body.get("session_id"))
                    )
                if path == "/api/new-session":
                    return self.send_json({"session_id": store.new_session()})
                if path == "/api/teach":
                    lesson = store.add_lesson(
                        body.get("intent", ""),
                        body.get("text", ""),
                        source="desktop",
                    )
                    return self.send_json(lesson, 201)
                if path == "/api/review":
                    return self.send_json(
                        store.review_lesson(
                            body.get("lesson_id", ""),
                            body.get("reviewed_by", ""),
                            body.get("permission", ""),
                        )
                    )
                return self.send_error(404)
            except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
                return self.send_json(
                    {"error": type(error).__name__, "message": str(error)}, 400
                )
            except Exception:
                return self.send_json({"error": "INTERNAL_ERROR"}, 500)

    class LocalThreadingHTTPServer(ThreadingHTTPServer):
        daemon_threads = True

    return LocalThreadingHTTPServer(("127.0.0.1", port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir")
    parser.add_argument("--port", type=int, default=39061)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = make_server(args.state_dir, args.port)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(url, flush=True)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
