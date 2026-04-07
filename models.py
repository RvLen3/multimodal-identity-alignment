import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Sequence
from transformers import AutoModel, AutoTokenizer as HFAutoTokenizer
import os
os.environ["MODELSCOPE_CACHE"] = "/modelscope_cache"
from modelscope import T5ForConditionalGeneration

DEFAULT_ID_PRETRAINED = "google/byt5-small"
DEFAULT_LONG_PRETRAINED = "hfl/chinese-roberta-wwm-ext"
T5model = T5ForConditionalGeneration.from_pretrained(DEFAULT_ID_PRETRAINED)
T5tokenizer = HFAutoTokenizer.from_pretrained(DEFAULT_ID_PRETRAINED)
PretrainedTokenizer = HFAutoTokenizer.from_pretrained(DEFAULT_LONG_PRETRAINED)
PretrainedPath = DEFAULT_LONG_PRETRAINED




# 中文长文本编码基座：默认使用中文 RoBERTa
class BertBackbone(nn.Module):
    def __init__(self, pretrained_name=DEFAULT_LONG_PRETRAINED):
        super().__init__()
        print(f"正在加载 {pretrained_name} 作为文本基座...")
        self.bert = AutoModel.from_pretrained(pretrained_name)

    def forward(self, idx, mask_idx):
        # Transformers attention_mask 使用 0/1 整型更稳妥
        outputs = self.bert(input_ids=idx, attention_mask=mask_idx.long())
        return outputs.last_hidden_state[:, 0, :]  # 取 [CLS] token 的输出作为句子特征


class T5Backbone(nn.Module):
    def __init__(self, pretrained_name=DEFAULT_ID_PRETRAINED):
        super().__init__()
        print(f"正在加载 {pretrained_name} 作为 ID 文本基座...")
        self.t5 = T5model if pretrained_name == DEFAULT_ID_PRETRAINED else T5ForConditionalGeneration.from_pretrained(pretrained_name)

    def forward(self, idx, mask_idx):
        outputs = self.t5.encoder(input_ids=idx, attention_mask=mask_idx.long())
        return outputs.last_hidden_state  # [Batch, SeqLen, Hidden]


# ==========================================
# 1. 基础特征提取基座 (Backbones)
# ==========================================

class TextBackbone(nn.Module):
    """
    轻量级公共文本基座：使用 Embedding + Mean Pooling 代替庞大的 BERT 编码器
    既能使用预训练的 Tokenizer 词表，又大幅节省显存
    """

    def __init__(self, vocab_size=384, embed_dim=1472, pretrained_name=None):
        super().__init__()
        # pretrained_name 支持两种形式:
        # 1) str: 仅覆盖 id 文本模型，long 文本仍用默认中文 BERT
        # 2) dict: {"id": "...", "long": "..."} 分别覆盖两类文本模型
        if pretrained_name is None:
            id_pretrained_name = DEFAULT_ID_PRETRAINED
            long_pretrained_name = DEFAULT_LONG_PRETRAINED
        elif isinstance(pretrained_name, str):
            id_pretrained_name = pretrained_name
            long_pretrained_name = DEFAULT_LONG_PRETRAINED
        elif isinstance(pretrained_name, dict):
            id_pretrained_name = pretrained_name.get("id", DEFAULT_ID_PRETRAINED)
            long_pretrained_name = pretrained_name.get("long", DEFAULT_LONG_PRETRAINED)
        else:
            raise TypeError("pretrained_name 必须是 None / str / dict")

        print(f"正在抽取 {id_pretrained_name} 的底层词嵌入权重...")

        # ID 文本使用 ByT5 词嵌入/编码器
        pretrained = T5model if id_pretrained_name == DEFAULT_ID_PRETRAINED else T5ForConditionalGeneration.from_pretrained(id_pretrained_name)
        self.embedding_pretrained = pretrained.encoder.embed_tokens

        for param in self.embedding_pretrained.parameters():
            param.requires_grad = False

        # 长文本使用 BERT；ID 文本可切换到 ByT5 编码器
        self.bert_backbone = BertBackbone(long_pretrained_name)
        self.t5_backbone = T5Backbone(id_pretrained_name)

        # 注意力池化打分层: 输入维度从预训练 embedding 自动推断，避免硬编码
        self.id_hidden_dim = self.embedding_pretrained.embedding_dim
        self.attn_score = nn.Linear(self.id_hidden_dim, 1, bias=False)

    def forward(self, idx, mask_idx, type='id', method='mean'):
        # args:
        # type: 'id' 或 'sign'，分别对应两类文本；method: 'baseline', 'mean', 'attention', 'bert' 四种编码方式
        if type not in {'id', 'sign'}:
            raise ValueError(f"未知 type: {type}，可选值为 ['id', 'sign']")

        if method == 'bert' or type == 'sign':
            # sign/long 文本默认走 BERT 编码；dataset 已自动添加 special tokens
            return self.bert_backbone(idx, mask_idx)  # 输出: [Batch, 768]

        # 到这里默认是 ID 文本(ByT5)
        embeds = self.embedding_pretrained(idx)

        if method == 'baseline':
            # 直接对Embedding后的文本做均值池化
            mask_expanded = mask_idx.unsqueeze(-1).float()
            embeds = embeds * mask_expanded
            sum_embeds = embeds.sum(dim=1)
            lens = mask_idx.sum(dim=1, keepdim=True).clamp(min=1e-9).float()
            return sum_embeds / lens  # 输出: [Batch, EmbedDim]

        # mean/attention 使用 ByT5 encoder 后的上下文特征
        encoded = self.t5_backbone(idx, mask_idx)

        if method == 'mean':
            # 对Encoder后的文本做均值池化
            mask_expanded = mask_idx.unsqueeze(-1).float()
            encoded = encoded * mask_expanded
            sum_encoded = encoded.sum(dim=1)
            lens = mask_idx.sum(dim=1, keepdim=True).clamp(min=1e-9).float()
            return sum_encoded / lens

        if method == 'attention':
            # 对每个 token 打分，然后在有效 token 上做 softmax 得到注意力权重
            attn_logits = self.attn_score(encoded).squeeze(-1)  # [Batch, SeqLen]
            attn_logits = attn_logits.masked_fill(mask_idx <= 0, float('-inf'))

            # 避免整行无有效 token 时 softmax 产生 NaN
            no_valid_row = (mask_idx.sum(dim=1) <= 0)
            if no_valid_row.any():
                attn_logits = attn_logits.clone()
                attn_logits[no_valid_row] = 0.0

            attn_weights = F.softmax(attn_logits, dim=1).unsqueeze(-1)  # [Batch, SeqLen, 1]
            return (encoded * attn_weights).sum(dim=1)  # 输出: [Batch, EmbedDim]

        raise ValueError(f"未知 method: {method}，可选值为 ['baseline', 'mean', 'attention', 'bert']")


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



class UserModel(nn.Module):
    def __init__(self, CVModel, TextModel):
        super(UserModel, self).__init__()
        self.CVModel = CVModel
        self.TextModel = TextModel

        # 不同的投影头，赋予不同字段不同的语义空间
        self.name_head = nn.Linear(1472, 256)
        self.sign_head = nn.Linear(768, 256)
        self.name_ln = nn.LayerNorm(256)
        self.sign_ln = nn.LayerNorm(256)
        self.name_temp = nn.Parameter(torch.tensor(1.0))
        self.sign_temp = nn.Parameter(torch.tensor(1.0))
        self.avatar_head = nn.Linear(512, 256)
        self.bg_head = nn.Linear(512, 256)

        # 离散/数值特征处理:
        # [sex, following, follower, num_works_norm, has_name, has_sign, has_avatar, has_top_photo, top_photo_is_imputed]
        self.num_head = nn.Sequential(
            nn.Linear(9, 64),
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
        v_name = self.name_ln(self.name_head(self.TextModel(name_tokens['input_ids'], name_tokens['attention_mask'], 'id'))) * self.name_temp
        v_sign = self.sign_ln(self.sign_head(self.TextModel(sign_tokens['input_ids'], sign_tokens['attention_mask'], 'sign', method='bert'))) * self.sign_temp

        v_num = self.num_head(numerical_feats)

        # 沿着特征维度拼接
        concat_feat = torch.cat([v_avatar, v_bg, v_name, v_sign, v_num], dim=-1)
        return self.Aggregation(concat_feat)


class ManuModel(nn.Module):
    def __init__(self, CVModel, TextModel):
        super(ManuModel, self).__init__()
        self.CVModel = CVModel
        self.TextModel = TextModel

        self.title_head = nn.Linear(768, 256)
        self.title_ln = nn.LayerNorm(256)
        self.title_temp = nn.Parameter(torch.tensor(1.0))
        self.cover_head = nn.Linear(512, 256)
        # 修改为3维：点赞, 评论, 转发
        self.stats_head = nn.Linear(3, 64)

        self.item_fusion = nn.Linear(256 + 256 + 64, 256)
        self.attention_pooling = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)


    def forward(self, covers, titles_tokens, stats, work_valid_mask=None, cover_valid_mask=None):

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
        if cover_valid_mask is not None:
            v_cover = v_cover * cover_valid_mask.unsqueeze(-1).float()

        # 2. 处理文本特征：同理合并维度
        S = titles.shape[2]
        titles_flat = titles.view(B * N, S)

        # 👇【FIXED Bug 1】: 将 title_mask 也进行拍平操作，确保维度对齐
        title_mask_flat = title_mask.view(B * N, S)

        # 传入拍平后的 id 和 mask
        v_title_flat = self.title_ln(self.title_head(self.TextModel(titles_flat, title_mask_flat, 'sign', method='bert'))) * self.title_temp
        v_title = v_title_flat.view(B, N, -1)  # 恢复维度: [Batch, N, 256]
        if work_valid_mask is not None:
            v_title = v_title * work_valid_mask.unsqueeze(-1).float()

        # 3. 处理数值特征
        v_stats = self.stats_head(stats)  # Linear层自动作用于最后一个维度: [Batch, N, 64]
        if work_valid_mask is not None:
            v_stats = v_stats * work_valid_mask.unsqueeze(-1).float()

        # 4. 融合单条作品特征
        item_feats = F.relu(self.item_fusion(torch.cat([v_cover, v_title, v_stats], dim=-1)))  # [Batch, N, 256]

        # 5. 序列注意力池化 + mask
        key_padding_mask = None
        safe_valid_for_attn = None
        if work_valid_mask is not None:
            # MHA 不接受“整行全部被 mask”，否则会产生 NaN
            safe_valid_for_attn = work_valid_mask.float()
            no_valid_row = safe_valid_for_attn.sum(dim=1) <= 0
            if no_valid_row.any():
                safe_valid_for_attn = safe_valid_for_attn.clone()
                safe_valid_for_attn[no_valid_row, 0] = 1.0
            key_padding_mask = safe_valid_for_attn <= 0
        attn_out, _ = self.attention_pooling(item_feats, item_feats, item_feats, key_padding_mask=key_padding_mask)

        if work_valid_mask is None:
            works_feat = torch.mean(attn_out, dim=1)
        else:
            valid = work_valid_mask.unsqueeze(-1).float()
            denom = valid.sum(dim=1).clamp(min=1.0)
            works_feat = (attn_out * valid).sum(dim=1) / denom

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
