""""Dataclasses to represent Search Results and Documents."""

from pydantic import BaseModel


class Document(BaseModel):
    """Dataclass for Documents."""
    text: str
    id: str

class SearchResult(BaseModel):
    """Dataclass for SearchResults."""
    
    document: Document
    score: float
