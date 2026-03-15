import torch
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd

from dataset import CrossPlatformDataset, align_collate_fn
from models import MultiModalAlignment

# ==========================================
# 配置
# ==========================================

MODEL_PATH = "./saved_models/baseline.pth"
SAVE_PT_PATH = "./user_vectors.pt"

DATA_DIR = "./data"
BATCH_SIZE = 32
MAX_VIDEOS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. 推理 Dataset (单用户)
# ==========================================

class UserInferenceDataset(Dataset):

    def __init__(self, data_dir, max_videos=3):
        self.data_dir = data_dir

        # 复用原 Dataset 的特征提取逻辑
        self.base_dataset = CrossPlatformDataset(
            data_dir=data_dir,
            method="pair",
            max_videos=max_videos
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

        return users

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):

        plat, uid = self.users[idx]

        feat = self.base_dataset._get_single_user_feat(plat, uid)

        return feat, plat, uid


# ==========================================
# 2. collate_fn
# ==========================================

def inference_collate_fn(batch):

    feats = [x[0] for x in batch]
    platforms = [x[1] for x in batch]
    uids = [x[2] for x in batch]

    # 伪造 pair 输入 (feat1, feat2)
    batch_list = [(f, f, torch.tensor([1.0])) for f in feats]

    feat1, _, _ = align_collate_fn(batch_list)

    return feat1, platforms, uids


# ==========================================
# 3. 输入整理
# ==========================================

def move_tokens_to_device(tokens_dict, dev):
    return {k: v.to(dev) for k, v in tokens_dict.items()}


def prepare_model_inputs(feat, dev):

    user_inputs = (
        feat["avatar"].to(dev),
        feat["top_photo"].to(dev),
        move_tokens_to_device(feat["name_tokens"], dev),
        move_tokens_to_device(feat["sign_tokens"], dev),
        feat["profile_numeric"].to(dev)
    )

    manu_inputs = (
        feat["video_covers"].to(dev),
        move_tokens_to_device(feat["video_title_tokens"], dev),
        feat["video_stats"].to(dev)
    )

    return user_inputs, manu_inputs


# ==========================================
# 4. 导出用户向量
# ==========================================

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
        collate_fn=inference_collate_fn
    )

    all_platforms = []
    all_uids = []
    all_vectors = []

    print("Computing vectors...")

    for feat, platforms, uids in tqdm(dataloader):

        user_inputs, manu_inputs = prepare_model_inputs(feat, device)

        vectors = model(user_inputs, manu_inputs)  # [B, D]

        all_vectors.append(vectors.cpu())

        all_platforms.extend(platforms)
        all_uids.extend(uids)

    final_vectors = torch.cat(all_vectors, dim=0)

    save_dict = {
        "platforms": all_platforms,
        "uids": all_uids,
        "vectors": final_vectors
    }

    torch.save(save_dict, SAVE_PT_PATH)

    print("\nDone!")
    print("Users:", len(all_uids))
    print("Vector shape:", final_vectors.shape)


# ==========================================
# 5. 额外导出 CSV（可选）
# ==========================================

def export_csv():

    data = torch.load(SAVE_PT_PATH)

    df = pd.DataFrame({
        "platform": data["platforms"],
        "uid": data["uids"],
        "vector": [v.numpy().tolist() for v in data["vectors"]]
    })

    csv_path = SAVE_PT_PATH.replace(".pt", ".csv")

    df.to_csv(csv_path, index=False)

    print("CSV saved:", csv_path)


# ==========================================
# main
# ==========================================

if __name__ == "__main__":

    os.makedirs(os.path.dirname(SAVE_PT_PATH), exist_ok=True)

    export_user_vectors()

    export_csv()