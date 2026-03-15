import json
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from PIL import Image

data_dir = "/data"

import pandas as pd
from itertools import combinations
from torch.utils.data import Dataset

import os
import json
import torch
import pandas as pd
from PIL import Image
from itertools import combinations
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.dataloader import default_collate
os.environ["MODELSCOPE_CACHE"] = "/modelscope_cache"
from modelscope import AutoTokenizer

class CrossPlatformDataset(Dataset):
    def __init__(self, data_dir, method="pair", max_videos=3, transform=None):
        """
        :param data_dir: 数据根目录 (存放 data.csv, bili/, douyin/, weibo/)
        :param method: "pair" 或 "triplet"
        :param max_videos: 每个用户截取的最大作品数量（用于对齐 Tensor 维度）
        :param transform: 图像的预处理逻辑 (Resize + ToTensor)
        """
        self.data_dir = data_dir
        self.method = method
        self.max_videos = max_videos

        # 如果没有传入图像处理逻辑，给定一个默认的标准预处理
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.data_list = self.get_user_list()

    def get_user_list(self):
        # 读取 CSV 并构造 Pair 或 Triplet 列表 (复用之前优化好的代码)
        data_pair = pd.read_csv(f"{self.data_dir}/data.csv")
        data_list = []

        def clean_id(val):
            if pd.isna(val): return None
            return str(val).split('.')[0]

        for _, row in data_pair.iterrows():
            anchors = []
            for plat, col in [('bili', 'Bili_id'), ('douyin', 'Douyin_id'), ('weibo', 'Weibo_id')]:
                uid = clean_id(row[col])
                if uid: anchors.append((plat, uid))

            negs = []
            for plat, col in [('bili', 'Bili_neg'), ('douyin', 'Douyin_neg'), ('weibo', 'Weibo_neg')]:
                uid = clean_id(row[col])
                if uid: negs.append((plat, uid))

            if len(anchors) < 2: continue

            if self.method == "pair":
                for node1, node2 in combinations(anchors, 2):
                    data_list.append([node1, node2, 1])  # 正样本
                for anchor in anchors:
                    for neg in negs:
                        data_list.append([anchor, neg, 0])  # 负样本
            elif self.method == "triplet":
                for i in range(len(anchors)):
                    for j in range(len(anchors)):
                        if i == j: continue
                        for neg_node in negs:
                            data_list.append([anchors[i], anchors[j], neg_node])
        return data_list

    def __len__(self):
        return len(self.data_list)

    def process_follower(self, following):
        """处理粉丝数/关注数，统一转为 float"""
        if isinstance(following, (int, float)):
            return float(following)
        if isinstance(following, str):
            following = following.strip()
            if '万' in following:
                return float(following.replace('万', '')) * 10000.0
            elif following.isdigit():
                return float(following)
        return 0.0

    def _get_single_user_feat(self, plat, uid):
        """
        核心私有方法：读取单一平台单一用户的所有多模态特征
        返回一个字典，包含所有的文本(str)和图像/数值(Tensor)
        """
        user_dir = os.path.join(self.data_dir, plat, uid)
        json_path = os.path.join(user_dir, f"{uid}.json")

        # 兜底默认值 (防止文件损坏或丢失导致 DataLoader 崩溃)
        empty_img = torch.zeros((3, 224, 224))
        feat_dict = {
            'name': "", 'sign': "", 'sex': 0.0,  # sex: 1=男, 0=女, 0.5=未知
            'following': 0.0, 'follower': 0.0,
            'avatar': empty_img, 'top_photo': empty_img,
            'video_titles': ["" for _ in range(self.max_videos)],
            'video_stats': torch.zeros((self.max_videos, 3)),  # likes, comments, shares
            'video_covers': torch.zeros((self.max_videos, 3, 224, 224))
        }

        if not os.path.exists(json_path):
            return feat_dict  # 文件不存在直接返回全零占位符

        with open(json_path, 'r', encoding='utf-8') as f:
            info = json.load(f)

        # 1. 提取基础信息 (兼容不同平台的字段名)
        user_info = info.get("creator_profile") or info.get("creator") or {}
        feat_dict['name'] = user_info.get('name', "")
        feat_dict['sign'] = user_info.get('sign', "")

        sex_str = str(user_info.get('sex', '')).lower()
        if sex_str in ['male', '男', '1']:
            feat_dict['sex'] = 1.0
        elif sex_str in ['female', '女', '0']:
            feat_dict['sex'] = 0.0
        else:
            feat_dict['sex'] = 0.5 # unknown

        feat_dict['following'] = self.process_follower(user_info.get('following', 0))
        feat_dict['follower'] = self.process_follower(user_info.get('follower', 0))

        # 2. 读取静态图像 (头像、背景)
        try:
            feat_dict['avatar'] = self.transform(Image.open(os.path.join(user_dir, f"{uid}.jpg")).convert('RGB'))
        except:
            pass

        try:
            feat_dict['top_photo'] = self.transform(Image.open(os.path.join(user_dir, "top_photo.jpg")).convert('RGB'))
        except:
            pass

        # 3. 处理作品序列 (截断或填充到 max_videos 长度)
        video_info = info.get("videos") or info.get("notes") or []

        for i in range(min(len(video_info), self.max_videos)):
            v = video_info[i]
            vid = v.get('aweme_id') or v.get('note_id') or v.get('bvid') or ""

            # 文本合并
            title = v.get('title', "")
            desc = v.get('desc', "")
            feat_dict['video_titles'][i] = f"{title} {desc}".strip()

            # 交互数据
            likes = self.process_follower(v.get('liked_count') or v.get('likes') or 0)
            comments = self.process_follower(v.get('comment_count') or v.get('comment') or 0)
            shares = self.process_follower(v.get('share_count') or v.get('shares') or 0)
            feat_dict['video_stats'][i] = torch.tensor([likes, comments, shares], dtype=torch.float32)

            # 封面图 (微博有多张图时，只取 _0.jpg 第一张)
            cover_name = f"{vid}_0.jpg" if plat == 'weibo' else f"{vid}.jpg"
            try:
                feat_dict['video_covers'][i] = self.transform(
                    Image.open(os.path.join(user_dir, cover_name)).convert('RGB'))
            except:
                pass

        return feat_dict

    def __getitem__(self, idx):
        # 根据 idx，而不是用 for 循环遍历
        data = self.data_list[idx]

        if self.method == 'pair':
            (plat1, uid1), (plat2, uid2), label = data
            feat1 = self._get_single_user_feat(plat1, uid1)
            feat2 = self._get_single_user_feat(plat2, uid2)
            label_tensor = torch.tensor([label], dtype=torch.float32)
            return feat1, feat2, label_tensor

        elif self.method == 'triplet':
            (plat_a, uid_a), (plat_p, uid_p), (plat_n, uid_n) = data
            feat_a = self._get_single_user_feat(plat_a, uid_a)
            feat_p = self._get_single_user_feat(plat_p, uid_p)
            feat_n = self._get_single_user_feat(plat_n, uid_n)
            return feat_a, feat_p, feat_n


tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
import torch
from torch.utils.data.dataloader import default_collate


# 确保你的 tokenizer 在这里是可访问的，或者是作为参数传进来的

def align_collate_fn(batch_list):
    """
    针对跨平台对齐任务定制的 collate_fn
    batch_list: 包含 batch_size 个 (feat1, feat2, label) 的列表
    """
    # 1. 提取出不需要默认 collate 处理的嵌套文本（防止 list 转置问题）
    # 手动把 batch 中所有样本的 video_titles 抽出来
    feat1_titles = [sample[0]['video_titles'] for sample in batch_list]  # [batch_size, max_videos]
    feat2_titles = [sample[1]['video_titles'] for sample in batch_list]

    # 临时从字典中移除，避免 default_collate 捣乱
    for sample in batch_list:
        sample[0].pop('video_titles', None)
        sample[1].pop('video_titles', None)

    # 2. 使用 PyTorch 默认机制打包剩下的数值和图像张量
    batched_data = default_collate(batch_list)
    feat1, feat2, label = batched_data

    # ==========================================
    # 3. 核心修复点：利用 zip 将字典与刚才抽出的标题列表成对遍历
    # ==========================================
    for feat, titles in zip([feat1, feat2], [feat1_titles, feat2_titles]):
        # 处理网名和签名 (它们是普通的字符串列表，被 default_collate 打包成了 list of str)
        feat['name_tokens'] = tokenizer(feat['name'], padding=True, return_tensors='pt', max_length=64, truncation=True)
        feat['sign_tokens'] = tokenizer(feat['sign'], padding=True, return_tensors='pt', max_length=256, truncation=True)

        # 处理 video_titles (使用 zip 传进来的 titles 变量)
        batch_size = len(titles)
        video_size = len(titles[0]) if batch_size > 0 else 0  # 防御性编程

        # 展平嵌套列表 [batch_size, video_size] -> [batch_size * video_size]
        flat_titles = [title for user_titles in titles for title in user_titles]

        # 统一 Tokenize
        flat_tokens = tokenizer(
            flat_titles,
            padding=True,
            truncation=True,  # 超出则截断
            max_length=64,
            return_tensors='pt'
        )

        seq_len = flat_tokens['input_ids'].shape[1]

        # 重塑并存入 feat 字典
        feat['video_title_tokens'] = {
            'input_ids': flat_tokens['input_ids'].view(batch_size, video_size, seq_len),
            'attention_mask': flat_tokens['attention_mask'].view(batch_size, video_size, seq_len)
        }

        feat['video_titles'] = titles

    # 4. 构建数值统计特征
    feat1['profile_numeric'] = torch.stack([
        feat1['sex'].float(),
        feat1['following'].float(),
        feat1['follower'].float()
    ], dim=1)

    feat2['profile_numeric'] = torch.stack([
        feat2['sex'].float(),
        feat2['following'].float(),
        feat2['follower'].float()
    ], dim=1)

    return feat1, feat2, label

if __name__ == "__main__":
    # 1. 设置你的数据根目录 (确保路径正确)
    DATA_DIR = r"E:\BS\data"

    # 2. 实例化我们写好的 Dataset (这里以 pair 模式为例)
    print("正在初始化 Dataset...")
    try:
        dataset = CrossPlatformDataset(
            data_dir=DATA_DIR,
            method="pair",
            max_videos=3  # 每个用户最多保留 3 个作品，对齐维度
        )
        print(f"Dataset 初始化成功！共构造了 {len(dataset)} 个数据对样本。")
    except Exception as e:
        print(f"Dataset 初始化失败，请检查 data.csv 是否存在以及路径是否正确。报错信息: {e}")
        exit()

    # 3. 实例化 DataLoader
    # batch_size=4 用于测试，实际训练时可以根据显存调大到 16 或 32
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Windows 下测试时建议先设为 0，防止多进程报错；真正放到 Linux 服务器训练时再设为 4 或 8,
        collate_fn=align_collate_fn
    )

    # 4. 抽取第一个 Batch 并打印维度信息
    print("\n" + "=" * 40)
    print("开始抽取第一个 Batch 的数据...")
    print("=" * 40)

    # 获取一个 Batch 的数据
    batch_data = next(iter(dataloader))

    # 因为我们在 pair 模式下返回的是 (feat1, feat2, label)
    feat1, feat2, label = batch_data

    print(f"\n【Label 标签张量维度】: {label.shape}  -> 期望: [4, 1] (代表 4 个样本的 0 或 1)")

    print("\n【平台 A (feat1) 的张量维度解析】:")
    # 遍历字典中的所有特征，打印它们的类型和形状
    for key, value in feat1.items():
        if isinstance(value, torch.Tensor):
            print(f"- {key:<20} : Tensor {value.shape}")
        elif isinstance(value, list):
            print(f"- {key:<20} : List of strings, length = {len(value)} (对应 Batch Size)")
        # 新增：拦截字典/BatchEncoding，并打印内部的张量维度
        elif hasattr(value, 'keys'):
            print(f"- {key:<20} : Dict/BatchEncoding 包含以下内容:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, torch.Tensor):
                    print(f"      -> {sub_key:<14} : Tensor {sub_value.shape}")

    print(f"-> 图像维度 (avatar): {feat1['avatar'].shape}  [Batch大小, 通道数, 图像高度, 图像宽度]")
    print(
        f"-> 序列图像 (video_covers): {feat1['video_covers'].shape}  [Batch大小, 作品数量(max_videos), 通道数, 高, 宽]")
    print(
        f"-> 序列特征 (video_stats): {feat1['video_stats'].shape}  [Batch大小, 作品数量(max_videos), 3个统计值(赞/评/转)]")

    print("\n测试完成")