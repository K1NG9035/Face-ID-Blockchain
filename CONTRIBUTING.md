# Contributing

Thanks for helping improve FaceWebChain.

## Development

1. Create a branch for your change.
2. Create a virtual environment and install the project requirements.
3. Run `python -m pytest -q` before opening a pull request.
4. Keep changes focused and explain behavior changes in the pull request.

Tests must not require Google credentials, private keys, network access, or personal biometric images. Add deterministic tests for new pure logic and use fakes for external providers.

## Pull requests

Please include a concise problem statement, the approach taken, validation performed, and any privacy or security implications. Do not include credentials, downloaded images, face embeddings, or generated artifacts in commits.

By contributing, you agree that your contributions are provided under the MIT License in this repository.
