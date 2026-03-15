import ast
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn.functional as F

from .user_index import normalize_platform

_VECTOR_DF: Optional[pd.DataFrame] = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_vector_file() -> Path:
    root = _project_root()
    pt_path = root / 'user_vectors.pt'
    csv_path = root / 'user_vectors.csv'
    if pt_path.exists():
        return pt_path
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError('user_vectors.pt and user_vectors.csv both not found')


def _parse_csv_vector(raw) -> torch.Tensor:
    if isinstance(raw, torch.Tensor):
        return raw.float()

    text = str(raw).strip()
    if not text:
        return torch.empty(0, dtype=torch.float32)

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return torch.tensor(parsed, dtype=torch.float32)
    except (ValueError, SyntaxError):
        pass

    parts = [p.strip() for p in text.split(',') if p.strip()]
    return torch.tensor([float(x) for x in parts], dtype=torch.float32)


# 加载所有用户向量并持久化保存
def load_user_vectors(file_path: str = '') -> pd.DataFrame:
    global _VECTOR_DF

    target_file = Path(file_path) if file_path else _default_vector_file()
    if not target_file.is_absolute():
        target_file = _project_root() / target_file

    if not target_file.exists():
        raise FileNotFoundError(f'vector file not found: {target_file}')

    if target_file.suffix.lower() == '.pt':
        payload = torch.load(target_file, map_location='cpu')
        platforms = [normalize_platform(p) for p in payload['platforms']]
        uids = [str(u) for u in payload['uids']]
        vectors = payload['vectors']
        if isinstance(vectors, torch.Tensor):
            vectors = [v.float() for v in vectors]
        df = pd.DataFrame({'platform': platforms, 'uid': uids, 'vector': vectors})
    else:
        df = pd.read_csv(target_file)
        if 'platform' not in df.columns or 'uid' not in df.columns or 'vector' not in df.columns:
            raise ValueError('csv must contain platform, uid, vector columns')
        df['platform'] = df['platform'].astype(str).map(normalize_platform)
        df['uid'] = df['uid'].astype(str)
        df['vector'] = df['vector'].apply(_parse_csv_vector)

    # 去重：同平台同 uid 保留第一条
    df = df.drop_duplicates(subset=['platform', 'uid'], keep='first').reset_index(drop=True)
    _VECTOR_DF = df
    return _VECTOR_DF


def _get_vector_df() -> pd.DataFrame:
    global _VECTOR_DF
    if _VECTOR_DF is None:
        _VECTOR_DF = load_user_vectors('')
    return _VECTOR_DF


# 根据用户id和platform获取对应的vector
def get_user_vector(user_id: str, platform: Optional[str] = None) -> torch.Tensor:
    df = _get_vector_df()
    uid = str(user_id)

    target = df[df['uid'] == uid]
    if platform:
        target = target[target['platform'] == normalize_platform(platform)]

    if target.empty:
        if platform:
            raise KeyError(f'user not found: platform={platform}, uid={uid}')
        raise KeyError(f'user not found: uid={uid}')

    return target.iloc[0]['vector'].float()


# 根据获取的vector和目标平台的所有用户vector计算相似度
def compute_similarity(vec: torch.Tensor, target_platform: str, top_k: int = 1) -> List[Dict]:
    '''计算相似度并返回每个平台的top_k个候选人选'''
    df = _get_vector_df()
    plat = normalize_platform(target_platform)

    candidates = df[df['platform'] == plat]
    if candidates.empty:
        return []

    q = vec.float().view(1, -1)
    mat = torch.stack([v.float().view(-1) for v in candidates['vector'].tolist()], dim=0)

    q_norm = F.normalize(q, p=2, dim=1)
    mat_norm = F.normalize(mat, p=2, dim=1)
    sims = torch.mm(q_norm, mat_norm.t()).squeeze(0)

    k = min(max(1, int(top_k)), sims.shape[0])
    scores, idxs = torch.topk(sims, k=k)

    rows = candidates.reset_index(drop=True)
    result: List[Dict] = []
    for score, idx in zip(scores.tolist(), idxs.tolist()):
        row = rows.iloc[idx]
        result.append(
            {
                'platform': row['platform'],
                'uid': str(row['uid']),
                'score': float(score),
            }
        )
    return result
