from pydantic import BaseModel


class Document(BaseModel):
    text: str
    id: str

class SearchResult(BaseModel):
    document: Document
    score: float

