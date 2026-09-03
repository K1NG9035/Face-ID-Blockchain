# Security and Privacy

- Use a dedicated Sepolia wallet with no production funds.
- Never commit `.env`, private keys, service-account JSON, or RPC credentials.
- The contract stores hashes, a source URL, timestamp, and submitter address. Hashes can still be correlatable, so do not fingerprint sensitive material without consent.
- Face images and embeddings remain off-chain and are used locally for comparison.
- A match is a model output, not identity proof, ownership proof, or authorship proof.
- Review candidate URLs and obey provider terms, robots rules, copyright, and applicable privacy law.