# Demo Script

1. Show an authorized input image and the repository.
2. Run the `app.cli run` command with the configured threshold.
3. Show the live candidate URL, local distance, generated hashes, transaction hash, and Sepolia explorer URL.
4. Run `app.cli verify` and show `VERIFIED`.
5. Change one byte in the downloaded artifact and run verification again; the expected result is `TAMPERED`.

Never show credentials during recording. A live run depends on Google Vision quotas, reachable image URLs, dlib installation, funded Sepolia ETH, and an RPC endpoint.