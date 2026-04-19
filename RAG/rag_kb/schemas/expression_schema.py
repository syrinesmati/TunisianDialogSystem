from .base_schema import BaseEntry
from typing import Optional


class ExpressionEntry(BaseEntry):
    origin: Optional[str] = None
    severity: str
    gender_sensitive: bool
