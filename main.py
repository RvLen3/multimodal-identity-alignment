import torch
import csv
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

# 导入你原有的模块
from dataset import CrossPlatformDataset, align_collate_fn
from models import MultiModalAlignment

# ==========================================
# 1. 配置区
# ==========================================
MODEL_PATH = './saved_models/baseline.pth'
SAVE_CSV_PATH = './user_vectors.csv'
DATA_DIR = "./data"
BATCH_SIZE = 64
MAX_VIDEOS = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_tokens_to_device(tokens_dict, dev):
    return {k: v.to(dev) for k, v in tokens_dict.items()}


def prepare_model_inputs(feat, dev):
    user_inputs = (
        feat['avatar'].to(dev),
        feat['top_photo'].to(dev),
        move_tokens_to_device(feat['name_tokens'], dev),
        move_tokens_to_device(feat['sign_tokens'], dev),
        feat['profile_numeric'].to(dev)
    )
    manu_inputs = (
        feat['video_covers'].to(dev),
        move_tokens_to_device(feat['video_title_tokens'], dev),
        feat['video_stats'].to(dev)
    )
    return user_inputs, manu_inputs


@torch.no_grad()
def export_user_vectors():
    # 1. 加载模型
    print(f"正在从 {MODEL_PATH} 加载模型...")
    model = MultiModalAlignment().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 2. 准备数据集
    # 注意：这里假设你的 dataset 在 method="inference" 或类似模式下
    # 会返回 (feat, platform, uid) 这样的结构。
    # 如果 dataset 不支持，你可能需要修改 dataset.py 里的 __getitem__
    print("正在初始化推理数据集...")
    dataset = CrossPlatformDataset(data_dir=DATA_DIR, method="inference", max_videos=MAX_VIDEOS)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=align_collate_fn  # 确保 collate_fn 兼容单人模式
    )

    # 3. 开启 CSV 写入
    print(f"开始计算 Vector 并保存至 {SAVE_CSV_PATH}...")
    with open(SAVE_CSV_PATH, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['platform', 'uid', 'vector'])

        for feat, platforms, uids in tqdm(dataloader):
            # 这里的 feat 是单个用户的特征字典
            user_inputs, manu_inputs = prepare_model_inputs(feat, device)

            # 模型前向传播获取 Embedding
            vectors = model(user_inputs, manu_inputs)  # shape: [batch, dim]

            # 转为 CPU 数组
            vectors_np = vectors.cpu().numpy()

            # 逐行写入
            for i in range(len(platforms)):
                # 将 vector 转换为逗号/空格分隔的字符串，方便存储在 CSV 单个单元格
                vec_str = ",".join(map(str, vectors_np[i].tolist()))
                writer.writerow([platforms[i], uids[i], vec_str])

    print("\n任务完成！所有用户向量已保存。")


if __name__ == "__main__":
    # 确保保存目录存在
    os.makedirs(os.path.dirname(SAVE_CSV_PATH), exist_ok=True)
    export_user_vectors()