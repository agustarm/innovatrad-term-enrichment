from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class DocIn(BaseModel):
    text: str = Field(..., min_length=1)

class DocOut(BaseModel):
    keywords: List[str]

class MetaOut(BaseModel):
    method: str
    k: int
    model: str
    device: str
    extra: Dict[str, Any] = {}

class PredictResponse(BaseModel):
    results: List[DocOut]
    meta: MetaOut