import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Sequence
from transformers import AutoModel
import os
os.environ["MODELSCOPE_CACHE"] = "/modelscope_cache"
from modelscope import T5ForConditionalGeneration, AutoTokenizer
model = T5ForConditionalGeneration.from_pretrained('google/byt5-small')
tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')




# ==========================================
# 1. 基础特征提取基座 (Backbones)
# ==========================================

class TextBackbone(nn.Module):
    """
    轻量级公共文本基座：使用 Embedding + Mean Pooling 代替庞大的 BERT 编码器
    既能使用预训练的 Tokenizer 词表，又大幅节省显存
    """

    def __init__(self, vocab_size=384, embed_dim=1472, pretrained_name="google/byt5-small"):
        super().__init__()
        # 1. 随机初始化的 Embedding (用于网名等短文本，从头学习特定领域的字面特征)
        # padding_idx=0 保证了填充位置的特征始终为 0，不参与计算和梯度更新
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 2. 预训练的 Embedding (用于标题和签名等长文本，直接借用 BERT 丰富的先验语义)
        print(f"正在抽取 {pretrained_name} 的底层词嵌入权重...")

        # Need Some Time to Load pretrain Model
        # pretrained = AutoModel.from_pretrained(pretrained_name)

        pretrained = T5ForConditionalGeneration.from_pretrained(pretrained_name)
        self.embedding_pretrained = pretrained.encoder.embed_tokens


        for param in self.embedding_pretrained.parameters():
            param.requires_grad = False

    def forward(self, idx, mask_idx, type='id'):
        # 选择对应的 Embedding 层
        Emb = self.embedding if type == 'id' else self.embedding_pretrained

        embeds = Emb(idx)
        mask_expanded = mask_idx.unsqueeze(-1).float()
        embeds = embeds * mask_expanded
        sum_embeds = embeds.sum(dim=1)
        lens = mask_idx.sum(dim=1, keepdim=True).clamp(min=1e-9)

        return sum_embeds / lens  # 输出: [Batch, EmbedDim]


class CVBackbone(nn.Module):
    """
    公共视觉基座：采用经典的 ResNet18 作为轻量级视觉特征提取器
    """

    def __init__(self, output_dim=512):
        super().__init__()
        # 工业界标准做法：加载预训练权重，去掉最后一层分类头
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # 到平均池化层结束
        self.fc = nn.Linear(resnet.fc.in_features, output_dim)

    def forward(self, x):
        # x shape: [Batch, 3, 224, 224]
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # 展平: [Batch, 512]
        return self.fc(features)  # 输出: [Batch, 512]


# ==========================================
# 2. 模态对齐子网络
# ==========================================

class UserModel(nn.Module):
    def __init__(self, CVModel, TextModel):
        super(UserModel, self).__init__()
        self.CVModel = CVModel
        self.TextModel = TextModel

        # 不同的投影头，赋予不同字段不同的语义空间
        self.name_head = nn.Linear(1472, 256)
        self.sign_head = nn.Linear(1472, 256)
        self.avatar_head = nn.Linear(512, 256)
        self.bg_head = nn.Linear(512, 256)

        # 离散/数值特征处理 (修改为3维：性别, 关注数, 粉丝数)
        self.num_head = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )

        # 聚合用户信息
        # 256(name) + 256(sign) + 256(avatar) + 256(bg) + 128(num) = 1152
        self.Aggregation = nn.Sequential(
            nn.Linear(1152, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    # 👇【FIXED Bug 2】: 接收字典类型的 name_tokens 和 sign_tokens，避免解包错误
    def forward(self, avatar, bg, name_tokens, sign_tokens, numerical_feats):
        v_avatar = self.avatar_head(self.CVModel(avatar))
        v_bg = self.bg_head(self.CVModel(bg))

        # 👇【FIXED Bug 2】: 从字典中提取 input_ids 和 attention_mask 传给 TextModel
        v_name = self.name_head(self.TextModel(name_tokens['input_ids'], name_tokens['attention_mask'], 'id'))
        v_sign = self.sign_head(self.TextModel(sign_tokens['input_ids'], sign_tokens['attention_mask'], 'sign'))

        v_num = self.num_head(numerical_feats)

        # 沿着特征维度拼接
        concat_feat = torch.cat([v_avatar, v_bg, v_name, v_sign, v_num], dim=-1)
        return self.Aggregation(concat_feat)


class ManuModel(nn.Module):
    def __init__(self, CVModel, TextModel):
        super(ManuModel, self).__init__()
        self.CVModel = CVModel
        self.TextModel = TextModel

        self.title_head = nn.Linear(1472, 256)
        self.cover_head = nn.Linear(512, 256)
        # 修改为3维：点赞, 评论, 转发
        self.stats_head = nn.Linear(3, 64)

        self.item_fusion = nn.Linear(256 + 256 + 64, 256)
        self.attention_pooling = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)

    # 👇【FIXED Bug 2】: 接收字典类型的 titles_tokens
    def forward(self, covers, titles_tokens, stats):
        # 注意：这里的输入包含作品序列维度 N
        # covers shape: [Batch, N, C, H, W]
        # stats  shape: [Batch, N, 3]

        B, N = covers.shape[0], covers.shape[1]

        # 👇【FIXED Bug 2】: 从字典中提取 titles 和 mask
        titles = titles_tokens['input_ids']
        title_mask = titles_tokens['attention_mask']

        # 1. 处理视觉特征：必须先合并 Batch 和 N 维度才能送入 CVBackbone
        C, H, W = covers.shape[2:]
        covers_flat = covers.view(B * N, C, H, W)
        v_cover_flat = self.cover_head(self.CVModel(covers_flat))
        v_cover = v_cover_flat.view(B, N, -1)  # 恢复维度: [Batch, N, 256]

        # 2. 处理文本特征：同理合并维度
        S = titles.shape[2]
        titles_flat = titles.view(B * N, S)

        # 👇【FIXED Bug 1】: 将 title_mask 也进行拍平操作，确保维度对齐
        title_mask_flat = title_mask.view(B * N, S)

        # 传入拍平后的 id 和 mask
        v_title_flat = self.title_head(self.TextModel(titles_flat, title_mask_flat, 'sign'))
        v_title = v_title_flat.view(B, N, -1)  # 恢复维度: [Batch, N, 256]

        # 3. 处理数值特征
        v_stats = self.stats_head(stats)  # Linear层自动作用于最后一个维度: [Batch, N, 64]

        # 4. 融合单条作品特征
        item_feats = F.relu(self.item_fusion(torch.cat([v_cover, v_title, v_stats], dim=-1)))  # [Batch, N, 256]

        # 5. 序列注意力池化
        attn_out, _ = self.attention_pooling(item_feats, item_feats, item_feats)
        works_feat = torch.mean(attn_out, dim=1)  # 压缩作品序列 -> [Batch, 256]

        return works_feat


# ==========================================
# 3. 最终端到端对齐网络
# ==========================================

class MultiModalAlignment(nn.Module):
    def __init__(self):
        super(MultiModalAlignment, self).__init__()
        # 实例化共享基座
        self.cv_base = CVBackbone(output_dim=512)
        self.text_base = TextBackbone(vocab_size=384, embed_dim=1472)

        # 将共享基座传入两个子模块，确保权重完全一致，且节省显存
        self.UserModel = UserModel(self.cv_base, self.text_base)
        self.ManuModel = ManuModel(self.cv_base, self.text_base)
        self.Bili_vector = torch.nn.Parameter(torch.randn(128))  # 128维的均值向量，用于消除偏置
        self.Douyin_vector = torch.nn.Parameter(torch.randn(128))
        self.Weibo_vector = torch.nn.Parameter(torch.randn(128))

        # 最终融合层
        self.Final_Aggregation = nn.Sequential(
            nn.Linear(512 + 256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128)  # 输出 128 维，用于计算相似度/Loss
        )
        # 平台适配改为“残差小修正”，避免直接映射导致跨平台空间撕裂
        self.bili_adapter = nn.Linear(128, 128, bias=False)
        self.douyin_adapter = nn.Linear(128, 128, bias=False)
        self.weibo_adapter = nn.Linear(128, 128, bias=False)
        nn.init.zeros_(self.bili_adapter.weight)
        nn.init.zeros_(self.douyin_adapter.weight)
        nn.init.zeros_(self.weibo_adapter.weight)
        self.adapter_scale = 0.1

    def forward(self, user_inputs, manu_inputs, platform):
        """
        user_inputs: Tuple (avatar, bg, name_tokens, sign_tokens, numerical_feats)
        manu_inputs: Tuple (covers, titles_tokens, stats)
        """
        # 利用解包将数据传给对应的子网络
        user_vector = self.UserModel(*user_inputs)
        works_vector = self.ManuModel(*manu_inputs)

        final_concat = torch.cat([user_vector, works_vector], dim=-1)

        # 验证一下第一个想法:针对不同平台设置一个专门的映射层，看看能不能消除平台偏置
        # 这个映射层是加在最开始还是最后?都测试一下
        final_vector = self.Final_Aggregation(final_concat)
        final_vector = F.normalize(final_vector, p=2, dim=-1)
        platform_layers = {
            "bili": self.bili_adapter,
            "douyin": self.douyin_adapter,
            "weibo": self.weibo_adapter,
        }

        if isinstance(platform, str):
            if platform not in platform_layers:
                raise ValueError(f"未知平台: {platform}")
            adapted = final_vector + self.adapter_scale * platform_layers[platform](final_vector)
            return F.normalize(adapted, p=2, dim=-1)

        if isinstance(platform, Sequence):
            if len(platform) != final_vector.size(0):
                raise ValueError("platform 列表长度必须与 batch size 一致")
            unknown = sorted(set(p for p in platform if p not in platform_layers))
            if unknown:
                raise ValueError(f"未知平台: {unknown}")

            out = torch.empty_like(final_vector)
            for plat_name, layer in platform_layers.items():
                idx = [i for i, p in enumerate(platform) if p == plat_name]
                if not idx:
                    continue
                idx_tensor = torch.tensor(idx, device=final_vector.device, dtype=torch.long)
                selected = final_vector.index_select(0, idx_tensor)
                adapted = selected + self.adapter_scale * layer(selected)
                out.index_copy_(0, idx_tensor, F.normalize(adapted, p=2, dim=-1))
            return out

        raise TypeError(f"platform 应该是 str 或 Sequence[str]，实际为: {type(platform)}")
