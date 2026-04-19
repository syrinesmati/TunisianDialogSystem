from pydantic import BaseModel
from typing import List, Optional


class BaseEntry(BaseModel):
    id: str
    type: str

    term_arabic: str
    term_arabizi: str

    meaning: str
    meaning_fr: Optional[str] = None

    example: str
    usage_context: str

    region: str
    register: str
    generation: str

    scripts: List[str]

    source: str
    last_updated: str
