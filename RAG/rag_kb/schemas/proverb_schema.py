from .base_schema import BaseEntry
from typing import Optional


class ProverbEntry(BaseEntry):
    literal_meaning: str
    real_meaning: str
    when_used: str
    msa_equivalent: Optional[str] = None
