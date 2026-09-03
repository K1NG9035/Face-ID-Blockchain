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

## Run

```powershell
python -m app.cli run --image examples/input.jpg --threshold 0.50
python -m app.cli verify --record-id 1 --artifact artifacts/candidate_1.jpg --metadata artifacts/last_metadata.json
```

The first command requires live Google Vision and Sepolia access. It writes `artifacts/last_metadata.json` and `artifacts/last_run.json`; generated artifacts and credentials are ignored by Git.

## Test

```powershell
python -m pytest -q
```

Tests for the pure hashing, URL, threshold, and Web3 codec logic run without external credentials or network access.

## Open architecture

The live integrations are deliberately replaceable. Implement the `SearchProvider` protocol in `app/pipeline.py` to use another authorized discovery service, or inject a deterministic fake for local development. The face and blockchain modules similarly keep vendor-specific imports at runtime, so contributors can test the pure pipeline logic without Google Cloud, Ethereum, or private keys.

GitHub Actions runs the offline test suite on every push and pull request. The workflow intentionally does not use credentials or download biometric data.

## Responsible use

Only process images you are authorized to use. Do not use this project for surveillance or to claim a person's identity. See [docs/SECURITY.md](docs/SECURITY.md) for credential and privacy guidance.

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community expectations.