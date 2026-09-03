from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def as_bytes32(hex_hash: str) -> bytes:
    value = hex_hash.removeprefix("0x")
    if len(value) != 64:
        raise ValueError("A SHA-256 hash must contain exactly 64 hexadecimal characters")
    try:
        return bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError("Hash is not hexadecimal") from exc


def from_bytes32(value: bytes) -> str:
    if len(value) != 32:
        raise ValueError("Expected 32 bytes")
    return "0x" + value.hex()


@dataclass(frozen=True)
class ChainRecord:
    artifact_hash: str
    metadata_hash: str
    source_url: str
    timestamp: int
    submitter: str


class MatchRegistryClient:
    def __init__(self, rpc_url: str, private_key: str, contract_address: str, abi: list[dict[str, Any]]):
        try:
            from web3 import Web3
        except ImportError as exc:
            raise RuntimeError("Install web3 to use Sepolia recording") from exc
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self.web3.is_connected():
            raise RuntimeError("Unable to connect to the configured Sepolia RPC")
        account = self.web3.eth.account.from_key(private_key)
        self.account = account
        self.contract = self.web3.eth.contract(address=Web3.to_checksum_address(contract_address), abi=abi)

    def record(self, artifact_hash: str, metadata_hash: str, source_url: str) -> tuple[int, str]:
        nonce = self.web3.eth.get_transaction_count(self.account.address)
        transaction = self.contract.functions.recordMatch(as_bytes32(artifact_hash), as_bytes32(metadata_hash), source_url).build_transaction({"from": self.account.address, "nonce": nonce, "chainId": 11155111})
        signed = self.account.sign_transaction(transaction)
        tx_hash = self.web3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        return int(receipt["logs"][0]["topics"][1].hex(), 16), tx_hash.hex()

    def read(self, record_id: int) -> ChainRecord:
        values = self.contract.functions.getMatch(record_id).call()
        return ChainRecord(from_bytes32(values[0]), from_bytes32(values[1]), values[2], int(values[3]), values[4])
