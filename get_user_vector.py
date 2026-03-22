import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import CrossPlatformDataset, align_single_collate_fn
from models import MultiModalAlignment

MODEL_PATH = "./saved_models/baseline.pth"
SAVE_PT_PATH = "./user_vectors.pt"

DATA_DIR = "./data"
BATCH_SIZE = 32
MAX_VIDEOS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UserInferenceDataset(Dataset):
    def __init__(self, data_dir, max_videos=3):
        self.data_dir = data_dir
        self.base_dataset = CrossPlatformDataset(
            data_dir=data_dir,
            method="pair",
            max_videos=max_videos,
            debug=False,
        )
        self.users = self._scan_users()

    def _scan_users(self):
        users = []
        for platform in ["bili", "douyin", "weibo"]:
            plat_dir = os.path.join(self.data_dir, platform)
            if not os.path.exists(plat_dir):
                continue
            for uid in os.listdir(plat_dir):
                user_dir = os.path.join(plat_dir, uid)
                if os.path.isdir(user_dir):
                    users.append((platform, uid))
        users.sort(key=lambda x: (x[0], x[1]))
        return users

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        plat, uid = self.users[idx]
        feat = self.base_dataset._get_single_user_feat(plat, uid)
        return feat, plat, uid


def inference_collate_fn(batch):
    feats = [x[0] for x in batch]
    platforms = [x[1] for x in batch]
    uids = [x[2] for x in batch]
    batched_feat = align_single_collate_fn(feats)
    return batched_feat, platforms, uids


def move_tokens_to_device(tokens_dict, dev):
    return {k: v.to(dev) for k, v in tokens_dict.items()}


def prepare_model_inputs(feat, dev):
    user_inputs = (
        feat["avatar"].to(dev),
        feat["top_photo"].to(dev),
        move_tokens_to_device(feat["name_tokens"], dev),
        move_tokens_to_device(feat["sign_tokens"], dev),
        feat["profile_numeric"].to(dev),
    )

    manu_inputs = (
        feat["video_covers"].to(dev),
        move_tokens_to_device(feat["video_title_tokens"], dev),
        feat["video_stats"].to(dev),
        feat["work_i_is_valid"].to(dev),
        feat["work_cover_is_valid"].to(dev),
    )

    return user_inputs, manu_inputs


@torch.no_grad()
def export_user_vectors():
    print("Loading model...")
    model = MultiModalAlignment().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print("Loading dataset...")
    dataset = UserInferenceDataset(DATA_DIR, MAX_VIDEOS)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=inference_collate_fn,
    )

    all_platforms = []
    all_uids = []
    all_vectors = []

    print("Computing vectors...")
    for feat, platforms, uids in tqdm(dataloader):
        user_inputs, manu_inputs = prepare_model_inputs(feat, device)
        vectors = model(user_inputs, manu_inputs, platforms)
        all_vectors.append(vectors.cpu())
        all_platforms.extend(platforms)
        all_uids.extend(uids)

    if not all_vectors:
        raise RuntimeError("No users found to export. Check DATA_DIR and platform folders.")

    final_vectors = torch.cat(all_vectors, dim=0)
    save_dict = {
        "platforms": all_platforms,
        "uids": all_uids,
        "vectors": final_vectors,
    }
    torch.save(save_dict, SAVE_PT_PATH)

    print("\nDone!")
    print("Users:", len(all_uids))
    print("Vector shape:", final_vectors.shape)


def export_csv():
    data = torch.load(SAVE_PT_PATH)
    df = pd.DataFrame(
        {
            "platform": data["platforms"],
            "uid": data["uids"],
            "vector": [v.numpy().tolist() for v in data["vectors"]],
        }
    )

    csv_path = SAVE_PT_PATH.replace(".pt", ".csv")
    df.to_csv(csv_path, index=False)
    print("CSV saved:", csv_path)


if __name__ == "__main__":
    save_dir = os.path.dirname(SAVE_PT_PATH)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    export_user_vectors()
    export_csv()
