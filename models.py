from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class User:
    """Simple user model used for demo and session data."""

    email: str
    role: str = "viewer"
    files: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def add_file(self, filename: str, meta: Optional[Dict[str, Any]] = None) -> None:
        """Attach uploaded file metadata to the user object."""
        self.files.append(
            {
                "filename": filename,
                "meta": meta or {},
                "uploaded_at": datetime.utcnow().isoformat(),
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the user object to a dictionary."""
        return asdict(self)


@dataclass
class FileRecord:
    """Metadata for one uploaded file."""

    filename: str
    rows: int
    columns: List[str]
    uploaded_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert the file record to a dictionary."""
        return asdict(self)


@dataclass
class ForecastPoint:
    """A single forecast record."""

    date: str
    forecast: float
    lower: float
    upper: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert forecast point to a dictionary."""
        return asdict(self)


@dataclass
class AnalysisResult:
    """Container for analysis output."""

    summary: Dict[str, Any]
    kpis: Dict[str, Any]
    insights: List[Dict[str, Any]]
    forecast: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis result to a dictionary."""
        return asdict(self)