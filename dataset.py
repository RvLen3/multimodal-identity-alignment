import json
import os
import random
from itertools import combinations
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from modelscope import AutoTokenizer
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

os.environ["MODELSCOPE_CACHE"] = "/modelscope_cache"

tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")


def _clean_id(val):
    if pd.isna(val):
        return None
    return str(val).split(".")[0]


def _to_float_count(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return 0.0
        if "万" in text:
            return float(text.replace("万", "")) * 10000.0
        if text.isdigit():
            return float(text)
        try:
            return float(text)
        except ValueError:
            return 0.0
    return 0.0


class CrossPlatformDataset(Dataset):
    def __init__(
        self,
        data_dir,
        method="pair",
        max_videos=3,
        transform=None,
        easy_neg_per_anchor=1,
        seed=42,
    ):
        self.data_dir = data_dir
        self.method = method
        self.max_videos = max_videos
        self.easy_neg_per_anchor = max(0, int(easy_neg_per_anchor))
        self.rng = random.Random(seed)

        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.data_list = self.get_user_list()

    def get_user_list(self):
        data_pair = pd.read_csv(f"{self.data_dir}/data.csv")
        data_list = []

        all_users = set()
        rows = []

        # First pass: collect anchors/hard negatives and global user pool.
        for _, row in data_pair.iterrows():
            anchors = []
            for plat, col in [("bili", "Bili_id"), ("douyin", "Douyin_id"), ("weibo", "Weibo_id")]:
                uid = _clean_id(row[col])
                if uid:
                    node = (plat, uid)
                    anchors.append(node)
                    all_users.add(node)

            hard_negs = []
            for plat, col in [("bili", "Bili_neg"), ("douyin", "Douyin_neg"), ("weibo", "Weibo_neg")]:
                uid = _clean_id(row[col])
                if uid:
                    node = (plat, uid)
                    hard_negs.append(node)
                    all_users.add(node)

            if len(anchors) >= 2:
                rows.append((anchors, hard_negs))

        # Second pass: build samples with hard + random easy negatives.
        for anchors, hard_negs in rows:
            exclude = set(anchors) | set(hard_negs)
            easy_pool = [u for u in all_users if u not in exclude]
            easy_negs = (
                self.rng.sample(easy_pool, k=min(self.easy_neg_per_anchor, len(easy_pool))) if easy_pool else []
            )
            negs = list(dict.fromkeys(hard_negs + easy_negs))

            if self.method == "pair":
                for node1, node2 in combinations(anchors, 2):
                    data_list.append([node1, node2, 1])
                for anchor in anchors:
                    for neg in negs:
                        data_list.append([anchor, neg, 0])
            elif self.method == "triplet":
                for i in range(len(anchors)):
                    for j in range(len(anchors)):
                        if i == j:
                            continue
                        for neg in negs:
                            data_list.append([anchors[i], anchors[j], neg])
            else:
                raise ValueError(f"Unsupported method: {self.method}")

        return data_list

    def __len__(self):
        return len(self.data_list)

    def _get_single_user_feat(self, plat, uid):
        user_dir = os.path.join(self.data_dir, plat, uid)
        json_path = os.path.join(user_dir, f"{uid}.json")

        empty_img = torch.zeros((3, 224, 224))
        feat_dict = {
            "name": "",
            "sign": "",
            "sex": 0.5,
            "following": 0.0,
            "follower": 0.0,
            "avatar": empty_img,
            "top_photo": empty_img,
            "video_titles": ["" for _ in range(self.max_videos)],
            "video_stats": torch.zeros((self.max_videos, 3)),
            "video_covers": torch.zeros((self.max_videos, 3, 224, 224)),
        }

        if not os.path.exists(json_path):
            return feat_dict

        with open(json_path, "r", encoding="utf-8") as f:
            info = json.load(f)

        user_info = info.get("creator_profile") or info.get("creator") or {}
        feat_dict["name"] = user_info.get("name", "")
        feat_dict["sign"] = user_info.get("sign", "")

        sex_str = str(user_info.get("sex", "")).lower()
        if sex_str in ["male", "1", "男"]:
            feat_dict["sex"] = 1.0
        elif sex_str in ["female", "0", "女"]:
            feat_dict["sex"] = 0.0

        feat_dict["following"] = _to_float_count(user_info.get("following", 0))
        feat_dict["follower"] = _to_float_count(user_info.get("follower", 0))

        try:
            feat_dict["avatar"] = self.transform(Image.open(os.path.join(user_dir, f"{uid}.jpg")).convert("RGB"))
        except Exception:
            pass

        try:
            feat_dict["top_photo"] = self.transform(Image.open(os.path.join(user_dir, "top_photo.jpg")).convert("RGB"))
        except Exception:
            pass

        video_info = info.get("videos") or info.get("notes") or []
        for i in range(min(len(video_info), self.max_videos)):
            v = video_info[i]
            vid = v.get("aweme_id") or v.get("note_id") or v.get("bvid") or ""

            title = v.get("title", "")
            desc = v.get("desc", "")
            feat_dict["video_titles"][i] = f"{title} {desc}".strip()

            likes = _to_float_count(v.get("liked_count") or v.get("likes") or 0)
            comments = _to_float_count(v.get("comment_count") or v.get("comment") or 0)
            shares = _to_float_count(v.get("share_count") or v.get("shares") or 0)
            feat_dict["video_stats"][i] = torch.tensor([likes, comments, shares], dtype=torch.float32)

            cover_name = f"{vid}_0.jpg" if plat == "weibo" else f"{vid}.jpg"
            try:
                feat_dict["video_covers"][i] = self.transform(
                    Image.open(os.path.join(user_dir, cover_name)).convert("RGB")
                )
            except Exception:
                pass

        return feat_dict

    def __getitem__(self, idx):
        data = self.data_list[idx]

        if self.method == "pair":
            (plat1, uid1), (plat2, uid2), label = data
            feat1 = self._get_single_user_feat(plat1, uid1)
            feat2 = self._get_single_user_feat(plat2, uid2)
            return feat1, feat2, torch.tensor([label], dtype=torch.float32)

        (plat_a, uid_a), (plat_p, uid_p), (plat_n, uid_n) = data
        feat_a = self._get_single_user_feat(plat_a, uid_a)
        feat_p = self._get_single_user_feat(plat_p, uid_p)
        feat_n = self._get_single_user_feat(plat_n, uid_n)
        return feat_a, feat_p, feat_n


def _tokenize_and_pack_feat(feat, titles):
    feat["name_tokens"] = tokenizer(feat["name"], padding=True, return_tensors="pt", max_length=64, truncation=True)
    feat["sign_tokens"] = tokenizer(feat["sign"], padding=True, return_tensors="pt", max_length=256, truncation=True)

    batch_size = len(titles)
    video_size = len(titles[0]) if batch_size > 0 else 0
    flat_titles = [title for user_titles in titles for title in user_titles]

    flat_tokens = tokenizer(flat_titles, padding=True, truncation=True, max_length=64, return_tensors="pt")
    seq_len = flat_tokens["input_ids"].shape[1]

    feat["video_title_tokens"] = {
        "input_ids": flat_tokens["input_ids"].view(batch_size, video_size, seq_len),
        "attention_mask": flat_tokens["attention_mask"].view(batch_size, video_size, seq_len),
    }
    feat["video_titles"] = titles

    feat["profile_numeric"] = torch.stack(
        [feat["sex"].float(), feat["following"].float(), feat["follower"].float()],
        dim=1,
    )
    return feat


def align_collate_fn(batch_list):
    feat1_titles = [sample[0]["video_titles"] for sample in batch_list]
    feat2_titles = [sample[1]["video_titles"] for sample in batch_list]

    for sample in batch_list:
        sample[0].pop("video_titles", None)
        sample[1].pop("video_titles", None)

    feat1, feat2, label = default_collate(batch_list)
    feat1 = _tokenize_and_pack_feat(feat1, feat1_titles)
    feat2 = _tokenize_and_pack_feat(feat2, feat2_titles)
    return feat1, feat2, label


def align_single_collate_fn(feat_list):
    titles = [feat["video_titles"] for feat in feat_list]
    for feat in feat_list:
        feat.pop("video_titles", None)
    batched_feat = default_collate(feat_list)
    return _tokenize_and_pack_feat(batched_feat, titles)


def triplet_align_collate_fn(batch_list):
    feat_a = align_single_collate_fn([sample[0] for sample in batch_list])
    feat_p = align_single_collate_fn([sample[1] for sample in batch_list])
    feat_n = align_single_collate_fn([sample[2] for sample in batch_list])
    return feat_a, feat_p, feat_n
