from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Settings, load_dotenv_if_available
from .fingerprint import metadata_hash, sha256_file
from .pipeline import find_match
from .web_search import GoogleVisionSearch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="facewebchain", description="Face discovery, local verification, and integrity proof")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--image", type=Path, required=True)
    run.add_argument("--threshold", type=float, default=0.5)
    run.add_argument("--output", type=Path, default=Path("artifacts"))
    verify = commands.add_parser("verify")
    verify.add_argument("--record-id", type=int, required=True)
    verify.add_argument("--artifact", type=Path, required=True)
    verify.add_argument("--metadata", type=Path, required=True)
    return parser


def run_command(args: argparse.Namespace) -> int:
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    result = find_match(args.image, GoogleVisionSearch(), args.threshold, args.output)
    (args.output / "last_metadata.json").write_text(json.dumps(result.metadata, indent=2), encoding="utf-8")
    settings = Settings.from_environment()
    rpc, key, address = settings.require_blockchain()
    from .blockchain import MatchRegistryClient
    from .deploy_contract import load_abi
    record_id, tx_hash = MatchRegistryClient(rpc, key, address, load_abi()).record(result.artifact_hash, result.metadata_hash, result.candidate.url)
    (args.output / "last_run.json").write_text(json.dumps({"record_id": record_id, "transaction": tx_hash, "source_url": result.candidate.url}, indent=2), encoding="utf-8")
    print("[1/4] FACE ENCODING: complete")
    print(f"[2/4] WEB DISCOVERY: {result.candidate.url}")
    print(f"[3/4] FACE VERIFICATION: MATCH ({result.match.distance:.3f} <= {args.threshold:.3f})")
    print(f"[4/4] ARTIFACT SHA-256: {result.artifact_hash}\nMETADATA SHA-256: {result.metadata_hash}")
    print(f"BLOCKCHAIN: Sepolia record {record_id} ({settings.explorer_base_url}/tx/{tx_hash})")
    return 0


def verify_command(args: argparse.Namespace) -> int:
    settings = Settings.from_environment()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    local_artifact = sha256_file(args.artifact)
    local_metadata = metadata_hash(metadata)
    from .deploy_contract import load_abi
    from .blockchain import MatchRegistryClient
    rpc, key, address = settings.require_blockchain()
    record = MatchRegistryClient(rpc, key, address, load_abi()).read(args.record_id)
    verified = record.artifact_hash == local_artifact and record.metadata_hash == local_metadata
    print("VERIFIED" if verified else "TAMPERED")
    return 0 if verified else 1


def main() -> int:
    load_dotenv_if_available()
    args = build_parser().parse_args()
    try:
        return run_command(args) if args.command == "run" else verify_command(args)
    except (FileNotFoundError, LookupError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
