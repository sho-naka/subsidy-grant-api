from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class GrantType(str, Enum):
    SUBSIDY = "補助金"
    GRANT = "助成金"


class SearchRequest(BaseModel):
    prefecture: str = Field(..., description="都道府県")
    municipality: Optional[str] = Field(None, description="市区町村")
    industry: Optional[str] = Field(None, description="業種")
    keywords: Optional[str] = Field(None, description="任意キーワード")
    top_k: int = Field(10, ge=1, le=20, description="返却件数")


class GrantItem(BaseModel):
    title: str
    summary: str
    source_url: str
    grant_type: GrantType = Field(..., description="補助金または助成金の分類")
    deadline: Optional[str] = None
    amount_max: Optional[int] = None
    rate_max: Optional[float] = None
    area: Optional[str] = None
    municipality: Optional[str] = None
    industry: Optional[str] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    reasons: List[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    items: List[GrantItem]
    took_ms: int
