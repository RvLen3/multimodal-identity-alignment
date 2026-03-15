from typing import List, Optional

from pydantic import BaseModel, Field


class IdentityRef(BaseModel):
    platform: str
    account: str


class SearchRequest(BaseModel):
    source: IdentityRef
    target_platforms: List[str] = Field(default_factory=list)
    top_k: int = 5


class VerifyRequest(BaseModel):
    source: IdentityRef
    target: IdentityRef


class Candidate(BaseModel):
    platform: str
    account: str
    score: float


class SearchResponse(BaseModel):
    task_id: str
    mode: str = "search"
    source: IdentityRef
    candidates: List[Candidate]
    found_count: int
    status: str = "ok"


class ModalityScore(BaseModel):
    name: str
    desc: str
    weight: str
    score: float


class VerifyResponse(BaseModel):
    task_id: str
    mode: str = "verify"
    is_match: bool
    confidence: float
    source: IdentityRef
    target: IdentityRef
    modalities: List[ModalityScore]


class StatItem(BaseModel):
    label: str
    value: str
    trend: str


class TaskTarget(BaseModel):
    name: str
    platform: str


class TaskItem(BaseModel):
    id: str
    targetA: TaskTarget
    targetB: TaskTarget
    status: str
    score: Optional[float] = None
    timestamp: str


class ProfileItem(BaseModel):
    username: str
    platform: str
    id: str
    bio: str
    location: str
    avatar: str


class DetailItem(BaseModel):
    id: str
    overallScore: float
    profileA: ProfileItem
    profileB: ProfileItem
    modalities: List[ModalityScore]
    decision_lines: List[str] = Field(default_factory=list)


class OverviewResponse(BaseModel):
    stats: List[StatItem]
    tasks: List[TaskItem]
    selected_task_id: Optional[str] = None


class UserSuggestResponse(BaseModel):
    platform: str
    query: str
    items: List[str]
