# Architecture

`app.face` owns local face detection and distance comparison. `app.web_search` owns the live Google Vision adapter and bounded candidate download. `app.fingerprint` defines canonical JSON and SHA-256 behavior. `app.blockchain` contains bytes32 conversion and the optional Web3 client. `app.pipeline` composes discovery, local verification, and fingerprinting without knowing provider credentials. `app.cli` is the operator-facing entry point.

External services are deliberately optional at import time, which makes pure unit tests deterministic and keeps configuration failures explicit at runtime.