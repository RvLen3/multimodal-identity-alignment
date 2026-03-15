from copy import deepcopy
from datetime import datetime
from itertools import count
from typing import Dict, List

from .schemas import DetailItem, IdentityRef, ModalityScore, StatItem, TaskItem

_task_counter = count(9001)
TASKS: List[TaskItem] = []
DETAILS: Dict[str, DetailItem] = {}


def _now_hms() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _make_task_id() -> str:
    return f"T-{next(_task_counter)}"


def _build_modalities(base: float) -> List[ModalityScore]:
    return [
        ModalityScore(
            name="Visual_Embedding",
            desc="Placeholder visual feature score",
            weight="0.30",
            score=min(base + 0.03, 0.999),
        ),
        ModalityScore(
            name="Semantic_NLP",
            desc="Placeholder text semantic score",
            weight="0.25",
            score=max(base - 0.02, 0.1),
        ),
        ModalityScore(
            name="Temporal_Behavior",
            desc="Placeholder activity pattern score",
            weight="0.20",
            score=max(base - 0.09, 0.1),
        ),
        ModalityScore(
            name="Topology_Graph",
            desc="Placeholder graph similarity score",
            weight="0.25",
            score=max(base - 0.01, 0.1),
        ),
    ]


def _build_detail(
    task_id: str,
    source: IdentityRef,
    target: IdentityRef,
    score: float,
) -> DetailItem:
    source_seed = f"{source.platform}_{source.account}"
    target_seed = f"{target.platform}_{target.account}"
    return DetailItem(
        id=task_id,
        overallScore=score,
        profileA={
            "username": source.account,
            "platform": source.platform,
            "id": f"{source.platform}_{source.account}",
            "bio": "Mock source profile; backend contract already wired.",
            "location": "Shanghai, China",
            "avatar": f"https://api.dicebear.com/7.x/avataaars/svg?seed={source_seed}",
        },
        profileB={
            "username": target.account,
            "platform": target.platform,
            "id": f"{target.platform}_{target.account}",
            "bio": "Mock target profile; waiting for real inference service.",
            "location": "Shanghai, China",
            "avatar": f"https://api.dicebear.com/7.x/avataaars/svg?seed={target_seed}",
        },
        modalities=_build_modalities(score),
        decision_lines=[
            "> Initiating consensus evaluation...",
            f"> Source: {source.platform}:{source.account}",
            f"> Target: {target.platform}:{target.account}",
            f"> RESULT: {'MATCH' if score >= 0.85 else 'NO_MATCH'} (placeholder)",
        ],
    )


def _seed_data() -> None:
    if TASKS:
        return

    source = IdentityRef(platform="douyin", account="AlexChen_Douyin")
    target1 = IdentityRef(platform="bilibili", account="AlexChen_Bili")
    target2 = IdentityRef(platform="weibo", account="AlexChen_微博")
    for target, score in [(target1, 0.964), (target2, 0.882)]:
        task_id = _make_task_id()
        task = TaskItem(
            id=task_id,
            targetA={"name": source.account, "platform": source.platform},
            targetB={"name": target.account, "platform": target.platform},
            status="DONE",
            score=score,
            timestamp=_now_hms(),
        )
        TASKS.insert(0, task)
        DETAILS[task_id] = _build_detail(task_id, source, target, score)


def get_stats() -> List[StatItem]:
    _seed_data()
    total = len(TASKS)
    done = len([t for t in TASKS if t.status == "DONE"])
    avg = sum([(t.score or 0.0) for t in TASKS]) / done if done else 0.0
    return [
        StatItem(label="TOTAL_TASKS", value=str(total), trend="live"),
        StatItem(label="DONE_TASKS", value=str(done), trend="+"),
        StatItem(label="AVG_CONFIDENCE", value=f"{avg:.3f}", trend="stable"),
        StatItem(label="SYS_LATENCY", value="mock", trend="normal"),
    ]


def list_tasks() -> List[TaskItem]:
    _seed_data()
    return deepcopy(TASKS)


def get_detail(task_id: str) -> DetailItem:
    _seed_data()
    if task_id not in DETAILS:
        raise KeyError(task_id)
    return deepcopy(DETAILS[task_id])


def add_search_task(source: IdentityRef, target: IdentityRef, score: float) -> str:
    _seed_data()
    task_id = _make_task_id()
    task = TaskItem(
        id=task_id,
        targetA={"name": source.account, "platform": source.platform},
        targetB={"name": target.account, "platform": target.platform},
        status="DONE",
        score=score,
        timestamp=_now_hms(),
    )
    TASKS.insert(0, task)
    DETAILS[task_id] = _build_detail(task_id, source, target, score)
    return task_id
