from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    google_credentials: str | None
    sepolia_rpc_url: str | None
    private_key: str | None
    contract_address: str | None
    explorer_base_url: str

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            google_credentials=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            sepolia_rpc_url=os.getenv("SEPOLIA_RPC_URL"),
            private_key=os.getenv("SEPOLIA_PRIVATE_KEY"),
            contract_address=os.getenv("CONTRACT_ADDRESS"),
            explorer_base_url=os.getenv("EXPLORER_BASE_URL", "https://sepolia.etherscan.io"),
        )

    def require_blockchain(self) -> tuple[str, str, str]:
        values = (self.sepolia_rpc_url, self.private_key, self.contract_address)
        if any(value is None for value in values):
            raise RuntimeError("SEPOLIA_RPC_URL, SEPOLIA_PRIVATE_KEY, and CONTRACT_ADDRESS are required")
        return values  # type: ignore[return-value]


def load_dotenv_if_available(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"'))
