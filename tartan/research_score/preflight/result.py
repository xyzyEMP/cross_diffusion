from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class CheckResult:
    id: str
    severity: str
    status: str
    evidence: Dict[str, Any]
    remediation: str = ""

    def to_dict(self):
        return asdict(self)
