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
    run.add_argument("--require-liveness", action="store_true", help="Abort if face scan fails liveness/anti-spoofing check")
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
    from .service import run_pipeline_service
    args.image = args.image or INPUT_DIR
    if args.image.is_dir():
        args.image = find_input_image(args.image)
    if not args.image.is_file():
        raise FileNotFoundError(args.image)

    dossier = run_pipeline_service(
        image_input=args.image,
        output_dir=args.output,
        threshold=args.threshold,
        detector_model=args.model,
        upsample_times=args.upsample_times,
        mock_dir=args.mock_dir,
        skip_blockchain=args.skip_blockchain,
        require_liveness=args.require_liveness,
    )

    liv = dossier.liveness
    liv_status = "PASS" if liv["is_live"] else "WARNING/FAIL"
    print(f"[1/5] LIVENESS CHECK: {liv_status} ({liv['score']}%) | {', '.join(liv['reasons'])}")
    print(f"[2/5] FACE ENCODING: complete (model: {dossier.match_metrics['detector_model']})")

    post = dossier.social_post
    post_label = f"[{post['platform']}]"
    if post.get("author"):
        post_label += f" Author: {post['author']}"
    print(f"[3/5] SOCIAL DISCOVERY: {post_label}")
    print(f"      Image URL: {post['image_url']}")
    if post.get("post_url"):
        print(f"      Post URL:  {post['post_url']}")
    if post.get("caption"):
        print(f"      Caption:   {post['caption']}")

    metrics = dossier.match_metrics
    print(f"[4/5] FACE VERIFICATION: MATCH ({metrics['distance']:.3f} <= {metrics['threshold']:.3f}) | Confidence: {metrics['confidence']:.1f}%")
    if metrics.get("annotated_artifact"):
        print(f"      ANNOTATED ARTIFACT: {metrics['annotated_artifact']}")

    hashes = dossier.evidence_hashes
    print(f"[5/5] ARTIFACT SHA-256: {hashes['artifact_sha256']}\n      METADATA SHA-256: {hashes['metadata_sha256']}")

    proof = dossier.blockchain_proof
    if proof.get("status") == "anchored":
        print(f"BLOCKCHAIN: Sepolia record {proof['record_id']} ({proof['explorer_url']})")
    else:
        print("BLOCKCHAIN: Skipped (--skip-blockchain requested)")
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
