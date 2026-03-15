from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch.nn.functional as F

from .mock_data import add_search_task, get_detail, get_stats, list_tasks
from .schemas import (
    Candidate,
    DetailItem,
    IdentityRef,
    OverviewResponse,
    SearchRequest,
    SearchResponse,
    UserSuggestResponse,
    VerifyRequest,
    VerifyResponse,
)
from .user_index import normalize_platform, suggest_user_ids
from .vector_store import compute_similarity, get_user_vector

app = FastAPI(title="IdentityAlign Backend", version="0.1.0")
VALID_PLATFORMS = {"bili", "douyin", "weibo"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/search", response_model=SearchResponse)
def search_identities(req: SearchRequest) -> SearchResponse:
    if req.top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be >= 1")

    source_platform = normalize_platform(req.source.platform)
    source_uid = str(req.source.account)
    source_ref = IdentityRef(platform=source_platform, account=source_uid)

    if req.target_platforms:
        target_platforms = [normalize_platform(p) for p in req.target_platforms]
    else:
        target_platforms = [p for p in ["bili", "douyin", "weibo"] if p != source_platform]

    target_platforms = [p for p in target_platforms if p in VALID_PLATFORMS and p != source_platform]
    if not target_platforms:
        raise HTTPException(status_code=400, detail="target_platforms cannot be empty")

    try:
        source_vec = get_user_vector(user_id=source_uid, platform=source_platform)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"source user not found: {source_platform}/{source_uid}") from exc

    raw_results = []
    for platform in target_platforms:
        raw_results.extend(compute_similarity(source_vec, target_platform=platform, top_k=req.top_k))

    deduped = {}
    for item in raw_results:
        key = (item["platform"], item["uid"])
        if key not in deduped or item["score"] > deduped[key]["score"]:
            deduped[key] = item

    raw_results = sorted(deduped.values(), key=lambda x: x["score"], reverse=True)[: req.top_k]
    candidates = [
        Candidate(
            platform=item["platform"],
            account=item["uid"],
            score=item["score"],
        )
        for item in raw_results
    ]

    if not candidates:
        raise HTTPException(status_code=404, detail="no candidates found for target platforms")

    best = candidates[0]
    task_id = add_search_task(
        source=source_ref,
        target=IdentityRef(platform=best.platform, account=best.account),
        score=best.score,
    )
    return SearchResponse(
        task_id=task_id,
        source=source_ref,
        candidates=candidates,
        found_count=len(candidates),
    )


@app.post("/api/verify", response_model=VerifyResponse)
def verify_identity(req: VerifyRequest) -> VerifyResponse:
    source_platform = normalize_platform(req.source.platform)
    target_platform = normalize_platform(req.target.platform)
    source_uid = str(req.source.account)
    target_uid = str(req.target.account)

    if source_platform not in VALID_PLATFORMS or target_platform not in VALID_PLATFORMS:
        raise HTTPException(status_code=400, detail="invalid platform")

    try:
        source_vec = get_user_vector(source_uid, source_platform)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"source user not found: {source_platform}/{source_uid}") from exc

    try:
        target_vec = get_user_vector(target_uid, target_platform)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"target user not found: {target_platform}/{target_uid}") from exc

    cosine = F.cosine_similarity(source_vec.view(1, -1), target_vec.view(1, -1)).item()
    confidence = max(0.0, min(1.0, (cosine + 1.0) / 2.0))
    is_match = confidence >= 0.85

    source_ref = IdentityRef(platform=source_platform, account=source_uid)
    target_ref = IdentityRef(platform=target_platform, account=target_uid)
    task_id = add_search_task(
        source=source_ref,
        target=target_ref,
        score=confidence,
    )
    detail = get_detail(task_id)
    return VerifyResponse(
        task_id=task_id,
        is_match=is_match,
        confidence=confidence,
        source=source_ref,
        target=target_ref,
        modalities=detail.modalities,
    )


@app.get("/api/system/overview", response_model=OverviewResponse)
def system_overview() -> OverviewResponse:
    tasks = list_tasks()
    return OverviewResponse(
        stats=get_stats(),
        tasks=tasks,
        selected_task_id=tasks[0].id if tasks else None,
    )


@app.get("/api/tasks/{task_id}", response_model=DetailItem)
def task_detail(task_id: str) -> DetailItem:
    try:
        return get_detail(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="task not found") from exc


@app.get("/api/users/suggest", response_model=UserSuggestResponse)
def user_suggest(platform: str, q: str = "", limit: int = 8) -> UserSuggestResponse:
    normalized_platform = normalize_platform(platform)
    if normalized_platform not in {"bili", "douyin", "weibo"}:
        raise HTTPException(status_code=400, detail="invalid platform")
    items = suggest_user_ids(platform=normalized_platform, query=q, limit=max(1, min(limit, 50)))
    return UserSuggestResponse(platform=normalized_platform, query=q, items=items)
