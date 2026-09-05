from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Settings, load_dotenv_if_available
from .fingerprint import metadata_hash, sha256_file
from .pipeline import find_match
from .web_search import GoogleVisionSearch, LocalDirectorySearch

INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def find_input_image(input_dir: Path = INPUT_DIR) -> Path:
    images = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"Place one JPG, JPEG, PNG, or WEBP image in {input_dir}")
    if len(images) > 1:
        names = ", ".join(path.name for path in images)
        raise ValueError(f"Place only one input image in {input_dir}; found: {names}")
    return images[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="facewebchain", description="Face discovery, local verification, model training, and integrity proof")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Train a face recognition model from a directory of labeled photos")
    train.add_argument("--dataset", type=Path, required=True, help="Directory containing labeled face images")
    train.add_argument("--model-output", type=Path, default=Path("models/face_model.pkl"), help="Path to save trained model (default: models/face_model.pkl)")
    train.add_argument("--model", choices=["hog", "cnn"], default="hog", help="Detection model (default: hog)")
    train.add_argument("--threshold", type=float, default=0.5, help="Match threshold (default: 0.5)")
    train.add_argument("--upsample-times", type=int, default=0, help="Extra face-detection scales (default: 0)")

    run = commands.add_parser("run", help="Run face discovery, local verification, and blockchain recording")
    run.add_argument("--image", type=Path, help=f"Reference image; defaults to the only image in {INPUT_DIR}")
    run.add_argument("--threshold", type=float, default=0.5)
    run.add_argument("--upsample-times", type=int, default=0, help="Extra face-detection scales; use 1 for small faces")
    run.add_argument("--model", choices=["hog", "cnn"], default="hog", help="Detection model: hog (fast CPU) or cnn (deep/GPU) (default: hog)")
    run.add_argument("--mock-dir", type=Path, help="Run offline using a local directory of candidate images instead of Google Vision")
    run.add_argument("--no-annotate", action="store_true", help="Disable visual bounding box annotation artifact")
    run.add_argument("--skip-blockchain", action="store_true", help="Skip Sepolia on-chain registration (offline/local mode)")
    run.add_argument("--output", type=Path, default=OUTPUT_DIR, help=f"Folder for candidate image and JSON details (default: {OUTPUT_DIR})")

    verify = commands.add_parser("verify", help="Verify local artifact against Ethereum Sepolia record")
    verify.add_argument("--record-id", type=int, required=True)
    verify.add_argument("--artifact", type=Path, required=True)
    verify.add_argument("--metadata", type=Path, required=True)
    return parser


def train_command(args: argparse.Namespace) -> int:
    from .classifier import train_face_classifier
    print(f"[1/2] TRAINING MODEL on dataset: {args.dataset}")
    classifier = train_face_classifier(
        dataset_dir=args.dataset,
        output_model_path=args.model_output,
        model=args.model,
        upsample_times=args.upsample_times,
        threshold=args.threshold,
    )
    print(f"[2/2] MODEL TRAINED & SAVED: {args.model_output}")
    print(f"      Recognized Identities: {', '.join(classifier.classes)}")
    print(f"      Total Trained Samples: {len(classifier.labels)}")
    return 0


def run_command(args: argparse.Namespace) -> int:
    args.image = args.image or INPUT_DIR
    if args.image.is_dir():
        args.image = find_input_image(args.image)
    if not args.image.is_file():
        raise FileNotFoundError(args.image)

    search_provider = LocalDirectorySearch(args.mock_dir) if args.mock_dir else GoogleVisionSearch()
    annotate = not args.no_annotate

    result = find_match(
        args.image,
        search_provider,
        args.threshold,
        args.output,
        args.upsample_times,
        detector_model=args.model,
        annotate=annotate,
    )
    (args.output / "last_metadata.json").write_text(json.dumps(result.metadata, indent=2), encoding="utf-8")

    print(f"[1/4] FACE ENCODING: complete (model: {args.model})")
    print(f"[2/4] DISCOVERY SOURCE: {result.candidate.url}")
    print(f"[3/4] FACE VERIFICATION: MATCH ({result.match.distance:.3f} <= {args.threshold:.3f}) | Confidence: {result.match.confidence:.1f}%")
    if result.annotated_path:
        print(f"      ANNOTATED ARTIFACT: {result.annotated_path}")
    print(f"[4/4] ARTIFACT SHA-256: {result.artifact_hash}\n      METADATA SHA-256: {result.metadata_hash}")

    if args.skip_blockchain:
        print("BLOCKCHAIN: Skipped (--skip-blockchain requested)")
        (args.output / "last_run.json").write_text(
            json.dumps({
                "status": "skipped",
                "artifact_hash": result.artifact_hash,
                "metadata_hash": result.metadata_hash,
                "source_url": result.candidate.url,
            }, indent=2),
            encoding="utf-8",
        )
        return 0

    settings = Settings.from_environment()
    rpc, key, address = settings.require_blockchain()
    from .blockchain import MatchRegistryClient
    from .deploy_contract import load_abi
    record_id, tx_hash = MatchRegistryClient(rpc, key, address, load_abi()).record(result.artifact_hash, result.metadata_hash, result.candidate.url)
    (args.output / "last_run.json").write_text(json.dumps({"record_id": record_id, "transaction": tx_hash, "source_url": result.candidate.url}, indent=2), encoding="utf-8")
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
        if args.command == "train":
            return train_command(args)
        elif args.command == "run":
            return run_command(args)
        else:
            return verify_command(args)
    except (FileNotFoundError, LookupError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
