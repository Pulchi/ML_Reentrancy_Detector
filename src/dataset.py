"""Dataset loading – walks the nested folder structure and yields labelled samples."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .config import DATASET_ROOT, VERSION_FOLDERS, SAFE_FOLDER, REENTRANT_FOLDERS, PRAGMA_MAP


@dataclass
class ContractSample:
    path: Path
    source: str
    label: int              # 0 = safe, 1 = reentrant
    version_folder: str     # "0_4", "0_5", "0_8"
    solc_version: str       # "0.4.24", "0.5.0", "0.8.0"
    category: str           # "always-safe", "cross-contract", etc.


def iter_contracts(limit: int = 0) -> Iterator[ContractSample]:
    """Yield every .sol file under the dataset with its label."""
    count = 0
    for ver in VERSION_FOLDERS:
        ver_dir = DATASET_ROOT / ver
        if not ver_dir.exists():
            continue
        solc_ver = PRAGMA_MAP[ver]
        # Safe contracts
        safe_dir = ver_dir / SAFE_FOLDER
        if safe_dir.exists():
            for sol in safe_dir.rglob("*.sol"):
                source = sol.read_text(encoding="utf-8", errors="ignore")
                yield ContractSample(sol, source, 0, ver, solc_ver, SAFE_FOLDER)
                count += 1
                if limit and count >= limit:
                    return
        # Reentrant contracts
        for cat in REENTRANT_FOLDERS:
            cat_dir = ver_dir / cat
            if not cat_dir.exists():
                continue
            for sol in cat_dir.rglob("*.sol"):
                source = sol.read_text(encoding="utf-8", errors="ignore")
                yield ContractSample(sol, source, 1, ver, solc_ver, cat)
                count += 1
                if limit and count >= limit:
                    return


def load_all(limit: int = 0) -> list[ContractSample]:
    return list(iter_contracts(limit))
