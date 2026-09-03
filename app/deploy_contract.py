from __future__ import annotations

import json
from pathlib import Path

from .config import Settings


def load_abi() -> list[dict]:
    return json.loads(Path("contract_abi.json").read_text(encoding="utf-8"))


def main() -> None:
    try:
        from solcx import compile_source, install_solc
        from web3 import Web3
    except ImportError as exc:
        raise SystemExit("Install web3 and py-solc-x before deploying") from exc
    settings = Settings.from_environment()
    if not settings.sepolia_rpc_url or not settings.private_key:
        raise SystemExit("SEPOLIA_RPC_URL and SEPOLIA_PRIVATE_KEY are required")
    install_solc("0.8.20")
    source = Path("contracts/MatchRegistry.sol").read_text(encoding="utf-8")
    compiled = compile_source(source, output_values=["abi", "bin"], solc_version="0.8.20")
    artifact = next(iter(compiled.values()))
    web3 = Web3(Web3.HTTPProvider(settings.sepolia_rpc_url))
    account = web3.eth.account.from_key(settings.private_key)
    factory = web3.eth.contract(abi=artifact["abi"], bytecode=artifact["bin"])
    transaction = factory.constructor().build_transaction({"from": account.address, "nonce": web3.eth.get_transaction_count(account.address), "chainId": 11155111})
    signed = account.sign_transaction(transaction)
    tx_hash = web3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    Path("contract_abi.json").write_text(json.dumps(artifact["abi"], indent=2), encoding="utf-8")
    Path("contract_address.json").write_text(json.dumps({"address": receipt.contractAddress}, indent=2), encoding="utf-8")
    print(f"Deployed to {receipt.contractAddress}: {settings.explorer_base_url}/tx/{tx_hash.hex()}")


if __name__ == "__main__":
    main()
