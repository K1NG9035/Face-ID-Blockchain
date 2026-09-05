# FaceWebChain

FaceWebChain is a consent-based demonstration pipeline that discovers image references with Google Cloud Vision Web Detection, verifies candidate faces locally, and records only SHA-256 fingerprints on Ethereum Sepolia.

The code is released under the [MIT License](LICENSE). Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Data flow

`authorized image -> local face encoding -> live web discovery -> downloaded candidate -> local distance check -> artifact and metadata hashes -> Sepolia record -> independent verification`

The chain never receives a face image or face embedding. A face-distance match is not proof of identity, ownership, or authorship; it only reports the configured model's comparison result.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Fill `.env` with Google credentials, a dedicated Sepolia test wallet, an RPC URL, and the deployed contract address. Deploy with `python -m app.deploy_contract`, then copy the generated ABI/address into the project configuration as appropriate.

## Run & Train

### 1. Train the Face Model
You can train a face recognition classifier from a directory of labeled photos:

```powershell
python -m app.cli train --dataset data\authorized_faces --model-output models\face_model.pkl
```

### 2. Run Face Discovery & Verification Pipeline

```powershell
Copy-Item C:\Photos\authorized-face.jpg input\reference.jpg
python -m app.cli run
python -m app.cli verify --record-id 1 --artifact output\candidate_1.jpg --metadata output\last_metadata.json
```

For offline development and testing without live Google Vision or Sepolia funds:

```powershell
python -m app.cli run --mock-dir path\to\test_candidates --skip-blockchain
```

Place an authorized JPG, JPEG, PNG, or WEBP image in `input/`. The face model reads that image, tests candidate faces (including multi-face/crowd photos), and outputs:

- `output/candidate_1.jpg`: raw matching candidate image
- `output/candidate_1_annotated.jpg`: visual verification artifact with high-contrast bounding box, distance, and confidence
- `output/last_metadata.json`: source URL, face distance, confidence score, detector model, and bounding box coordinates
- `output/last_run.json`: blockchain record ID, transaction hash, and source URL

## Test

```powershell
python -m pytest -q
```

### CMake build targets

CMake can create an isolated build environment and expose the setup, test, contract deployment, and pipeline commands:

```powershell
cmake -S . -B build
cmake --build build --target setup
cmake --build build --target test
cmake --build build --target deploy-contract
cmake --build build --target run
```

The CMake `run` target reads the image from `input/` and writes details to `output/` by default. Use `-DFACE_IMAGE=C:/Photos/face.jpg` to override the input image. The `deploy-contract` and `run` targets require the same `.env` credentials described above. The `test` target is offline.

Tests for the pure hashing, URL, threshold, and Web3 codec logic run without external credentials or network access.

## Open architecture

The live integrations are deliberately replaceable. Implement the `SearchProvider` protocol in `app/pipeline.py` to use another authorized discovery service, or inject a deterministic fake for local development. The face and blockchain modules similarly keep vendor-specific imports at runtime, so contributors can test the pure pipeline logic without Google Cloud, Ethereum, or private keys.

GitHub Actions runs the offline test suite on every push and pull request. The workflow intentionally does not use credentials or download biometric data.

## Responsible use

Only process images you are authorized to use. Do not use this project for surveillance or to claim a person's identity. See [docs/SECURITY.md](docs/SECURITY.md) for credential and privacy guidance.

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community expectations.