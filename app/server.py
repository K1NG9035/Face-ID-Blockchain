from __future__ import annotations

import argparse
import cgi
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import sys
from typing import Any
from urllib.parse import parse_qs, urlparse

from .database import MatchDatabase
from .fingerprint import metadata_hash, sha256_file
from .service import run_pipeline_service

STATIC_DIR = Path("static")
OUTPUT_DIR = Path("output")


class PipelineApiHandler(SimpleHTTPRequestHandler):
    """Zero-dependency HTTP request handler for the FaceWebChain Web App & API."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def _send_json(self, data: Any, status: int = HTTPStatus.OK) -> None:
        payload = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/health":
            db = MatchDatabase()
            self._send_json({
                "status": "healthy",
                "service": "FaceWebChain Pipeline Engine",
                "total_records": db.count(),
                "static_ready": STATIC_DIR.is_dir(),
            })
            return

        if path == "/api/records":
            db = MatchDatabase()
            query = parse_qs(parsed.query)
            limit = int(query.get("limit", [50])[0])
            records = [r.to_dict() for r in db.list_records(limit=limit)]
            self._send_json({"records": records, "count": len(records)})
            return

        if path.startswith("/api/records/"):
            artifact_hash = path.removeprefix("/api/records/").strip()
            db = MatchDatabase()
            record = db.get_by_artifact_hash(artifact_hash)
            if record:
                self._send_json({"found": True, "record": record.to_dict()})
            else:
                self._send_json({"found": False, "error": "Record not found"}, status=HTTPStatus.NOT_FOUND)
            return

        # Serve generated output artifacts like candidate_1_annotated.jpg
        if path.startswith("/artifacts/"):
            artifact_filename = path.removeprefix("/artifacts/").strip()
            file_path = OUTPUT_DIR / artifact_filename
            if file_path.is_file():
                content_type, _ = mimetypes.guess_type(str(file_path))
                data = file_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type or "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
                return
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Artifact not found")
                return

        # Fallback to static files
        if not STATIC_DIR.is_dir():
            STATIC_DIR.mkdir(parents=True, exist_ok=True)
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_length)

        if path == "/api/scan":
            try:
                body = json.loads(post_body.decode("utf-8")) if post_body else {}
                image_input = body.get("image")
                if not image_input:
                    self._send_json({"error": "Missing 'image' in request body"}, status=HTTPStatus.BAD_REQUEST)
                    return

                threshold = float(body.get("threshold", 0.5))
                detector_model = body.get("model", "hog")
                upsample_times = int(body.get("upsample_times", 0))
                mock_dir = Path(body["mock_dir"]) if body.get("mock_dir") else None
                skip_blockchain = bool(body.get("skip_blockchain", True))
                require_liveness = bool(body.get("require_liveness", False))

                dossier = run_pipeline_service(
                    image_input=image_input,
                    threshold=threshold,
                    detector_model=detector_model,
                    upsample_times=upsample_times,
                    mock_dir=mock_dir,
                    skip_blockchain=skip_blockchain,
                    require_liveness=require_liveness,
                )
                self._send_json({"success": True, "dossier": dossier.to_dict()})
            except Exception as exc:
                self._send_json({"success": False, "error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if path == "/api/verify":
            try:
                body = json.loads(post_body.decode("utf-8")) if post_body else {}
                artifact_hash = body.get("artifact_hash")
                metadata_hash_val = body.get("metadata_hash")
                if not artifact_hash or not metadata_hash_val:
                    self._send_json({"error": "artifact_hash and metadata_hash required"}, status=HTTPStatus.BAD_REQUEST)
                    return

                db = MatchDatabase()
                record = db.get_by_artifact_hash(artifact_hash)
                if not record:
                    self._send_json({"verified": False, "reason": "Hash not found in registry"})
                    return

                is_verified = (record.metadata_hash == metadata_hash_val)
                self._send_json({
                    "verified": is_verified,
                    "status": "VERIFIED" if is_verified else "TAMPERED",
                    "record": record.to_dict(),
                })
            except Exception as exc:
                self._send_json({"success": False, "error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "API endpoint not found")


def start_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, PipelineApiHandler)
    print(f"FaceWebChain Web & API Server active at: http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Start FaceWebChain web & API server")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    args = parser.parse_args()
    start_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
