from dataclasses import dataclass
from typing import Optional

@dataclass
class CandidateProfile:
    resume: str
    linkedin_url: Optional[str] = None
    license_number: Optional[str] = None
    state: Optional[str] = None
    current_location: Optional[str] = None
