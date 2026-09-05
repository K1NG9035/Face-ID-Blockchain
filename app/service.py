from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import io
import json
from pathlib import Path
import tempfile
from typing import Any

from .config import Settings
from .liveness import LivenessResult, evaluate_liveness
from .pipeline import SearchProvider, find_match
from .social_resolver import SocialPostMetadata, resolve_social_post
from .web_search import GoogleVisionSearch, LocalDirectorySearch


@dataclass(frozen=True)
class PipelineDossier:
    status: str
    liveness: dict[str, Any]
    social_post: dict[str, Any]
    match_metrics: dict[str, Any]
    evidence_hashes: dict[str, str]
    blockchain_proof: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _decode_image_input(image_input: str | bytes | Path, temp_dir: Path) -> Path:
    """Resolve input whether it is a file path, raw bytes, or base64 string."""
    if isinstance(image_input, Path):
        if not image_input.is_file():
            raise FileNotFoundError(f"Input image not found: {image_input}")
        return image_input

    if isinstance(image_input, str):
        # Check if it is a filepath on disk
        potential_path = Path(image_input)
        if potential_path.is_file():
            return potential_path

        # Handle base64 string (including data:image/...;base64, prefixes)
        b64_data = image_input
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        image_bytes = base64.b64decode(b64_data)
        out_file = temp_dir / "input_scan.jpg"
        out_file.write_bytes(image_bytes)
        return out_file

    if isinstance(image_input, bytes):
        out_file = temp_dir / "input_scan.jpg"
        out_file.write_bytes(image_input)
        return out_file

    raise ValueError("Unsupported image input format")


def run_pipeline_service(
    image_input: str | bytes | Path,
    output_dir: Path = Path("output"),
    threshold: float = 0.5,
    detector_model: str = "hog",
    upsample_times: int = 0,
    search_provider: SearchProvider | None = None,
    mock_dir: Path | None = None,
    skip_blockchain: bool = False,
    require_liveness: bool = False,
) -> PipelineDossier:
    """High-level service orchestrator for face ingestion, search, and blockchain verification."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_str:
        temp_dir = Path(tmp_str)
        input_image_path = _decode_image_input(image_input, temp_dir)

        # 1. Passive Liveness & Anti-Spoofing Check
        liveness: LivenessResult = evaluate_liveness(input_image_path)
        if require_liveness and not liveness.is_live:
            raise ValueError(f"Face failed liveness verification: {', '.join(liveness.reasons)}")

        # 2. Select Search Provider
        provider: SearchProvider
        if search_provider is not None:
            provider = search_provider
        elif mock_dir is not None:
            provider = LocalDirectorySearch(mock_dir)
        else:
            settings = Settings.from_environment()
            default_mock = Path("mock_candidates")
            if not settings.google_credentials and default_mock.is_dir():
                provider = LocalDirectorySearch(default_mock)
            else:
                provider = GoogleVisionSearch()

        # 3. Discover, Download & Verify Candidate Face
        verified = find_match(
            image_path=input_image_path,
            search_provider=provider,
            threshold=threshold,
            output_dir=output_dir,
            upsample_times=upsample_times,
            detector_model=detector_model,
            annotate=True,
        )

        # 4. Social Media Context Resolution
        social_post: SocialPostMetadata = resolve_social_post(
            candidate_url=verified.candidate.url,
            page_url=verified.candidate.page_url,
        )

        # 5. Enrich Metadata with Social & Liveness context
        enriched_metadata = dict(verified.metadata)
        enriched_metadata["liveness"] = liveness.to_dict()
        enriched_metadata["social_post"] = social_post.to_dict()
        (output_dir / "last_metadata.json").write_text(
            json.dumps(enriched_metadata, indent=2), encoding="utf-8"
        )

        # 6. Blockchain Anchoring
        blockchain_proof: dict[str, Any] = {}
        if skip_blockchain:
            blockchain_proof = {
                "status": "skipped",
                "network": "local/simulated",
                "message": "Blockchain registration bypassed for development/offline test",
            }
            (output_dir / "last_run.json").write_text(
                json.dumps(
                    {
                        "status": "skipped",
                        "artifact_hash": verified.artifact_hash,
                        "metadata_hash": verified.metadata_hash,
                        "source_url": verified.candidate.url,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            settings = Settings.from_environment()
            try:
                rpc, key, address = settings.require_blockchain()
                from .blockchain import MatchRegistryClient
                from .deploy_contract import load_abi

                client = MatchRegistryClient(rpc, key, address, load_abi())
                record_id, tx_hash = client.record(
                    verified.artifact_hash,
                    verified.metadata_hash,
                    verified.candidate.url,
                )
                blockchain_proof = {
                    "status": "anchored",
                    "network": "Ethereum Sepolia",
                    "contract_address": address,
                    "record_id": record_id,
                    "transaction_hash": tx_hash,
                    "explorer_url": f"{settings.explorer_base_url}/tx/{tx_hash}",
                }
            except RuntimeError as exc:
                # Graceful fallback to offline simulation proof if credentials aren't set in .env
                import hashlib
                sim_tx = "0x" + hashlib.sha256((verified.artifact_hash + verified.metadata_hash).encode()).hexdigest()
                blockchain_proof = {
                    "status": "simulated",
                    "network": "Ethereum Sepolia (Simulated)",
                    "contract_address": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
                    "record_id": 1,
                    "transaction_hash": sim_tx,
                    "explorer_url": f"https://sepolia.etherscan.io/tx/{sim_tx}",
                    "note": f"Offline simulated attestation ({exc})",
                }

            (output_dir / "last_run.json").write_text(
                json.dumps(blockchain_proof, indent=2),
                encoding="utf-8",
            )

        dossier = PipelineDossier(
            status="VERIFIED_MATCH",
            liveness=liveness.to_dict(),
            social_post=social_post.to_dict(),
            match_metrics={
                "distance": round(verified.match.distance, 4),
                "threshold": threshold,
                "confidence": round(verified.match.confidence, 2),
                "cosine_similarity": round(getattr(verified.match, "cosine_similarity", 0.0), 4),
                "matched": verified.match.matched,
                "detector_model": detector_model,
                "face_location": verified.face_location,
                "artifact_path": str(verified.artifact_path),
                "annotated_artifact": str(verified.annotated_path) if verified.annotated_path else None,
                "embedding_tensor": verified.reference_encoding,
            },
            evidence_hashes={
                "artifact_sha256": verified.artifact_hash,
                "metadata_sha256": verified.metadata_hash,
            },
            blockchain_proof=blockchain_proof,
        )

        try:
            from .database import insert_record
            insert_record(dossier, db_path=output_dir / "evidence_vault.db")
        except Exception:
            pass

        return dossier
