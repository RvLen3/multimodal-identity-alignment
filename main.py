import argparse
import csv
import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import CrossPlatformDataset, align_collate_fn, triplet_align_collate_fn
from models import MultiModalAlignment


def move_tokens_to_device(tokens_dict: Dict[str, torch.Tensor], dev: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(dev) for k, v in tokens_dict.items()}


def prepare_model_inputs(feat: Dict, dev: torch.device) -> Tuple[Tuple, Tuple]:
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
    )
    return user_inputs, manu_inputs


def score_similarity(emb_a: torch.Tensor, emb_b: torch.Tensor, temperature: float) -> torch.Tensor:
    return F.cosine_similarity(emb_a, emb_b) * temperature


def binary_auc(labels: List[float], scores: List[float]) -> float:
    if len(labels) == 0:
        return 0.0

    n_pos = sum(1 for x in labels if x > 0.5)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    rank_sum_pos = 0.0
    idx = 0
    while idx < len(pairs):
        j = idx
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[idx][0]:
            j += 1
        avg_rank = (idx + 1 + j + 1) / 2.0
        for k in range(idx, j + 1):
            if pairs[k][1] > 0.5:
                rank_sum_pos += avg_rank
        idx = j + 1

    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def save_metrics(history: List[Dict], method: str) -> None:
    csv_path = f"baseline_{method}_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f"[Info] metrics saved to {csv_path}")


def plot_metrics(history: List[Dict], method: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[Warn] matplotlib not available, skip plotting: {exc}")
        return

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    train_auc = [h["train_auc"] for h in history]
    val_auc = [h["val_auc"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(epochs, train_loss, marker="o", label="train")
    axes[0].plot(epochs, val_loss, marker="o", label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, marker="o", label="train")
    axes[1].plot(epochs, val_acc, marker="o", label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(epochs, train_auc, marker="o", label="train")
    axes[2].plot(epochs, val_auc, marker="o", label="val")
    axes[2].set_title("AUC")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    fig.tight_layout()
    fig_path = f"baseline_{method}.png"
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)
    print(f"[Info] figure saved to {fig_path}")


def train_one_epoch_pair(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    temperature: float,
    max_steps: int,
) -> Tuple[float, float, float]:
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0
    all_scores: List[float] = []
    all_labels: List[float] = []

    pbar = tqdm(dataloader, desc="train(pair)", leave=False)
    for step, (feat1, feat2, label) in enumerate(pbar, start=1):
        if max_steps > 0 and step > max_steps:
            break

        label = label.view(-1).float().to(device)
        emb1 = model(*prepare_model_inputs(feat1, device))
        emb2 = model(*prepare_model_inputs(feat2, device))

        logits = score_similarity(emb1, emb2, temperature)
        loss = criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += (pred == label).sum().item()
            total_count += label.numel()
            total_loss += loss.item() * label.numel()
            all_scores.extend(logits.detach().cpu().tolist())
            all_labels.extend(label.detach().cpu().tolist())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return (
        total_loss / max(total_count, 1),
        total_correct / max(total_count, 1),
        binary_auc(all_labels, all_scores),
    )


@torch.no_grad()
def evaluate_pair(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    temperature: float,
    max_steps: int,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    all_scores: List[float] = []
    all_labels: List[float] = []

    pbar = tqdm(dataloader, desc="valid(pair)", leave=False)
    for step, (feat1, feat2, label) in enumerate(pbar, start=1):
        if max_steps > 0 and step > max_steps:
            break

        label = label.view(-1).float().to(device)
        emb1 = model(*prepare_model_inputs(feat1, device))
        emb2 = model(*prepare_model_inputs(feat2, device))

        logits = score_similarity(emb1, emb2, temperature)
        loss = criterion(logits, label)

        pred = (torch.sigmoid(logits) >= 0.5).float()
        total_correct += (pred == label).sum().item()
        total_count += label.numel()
        total_loss += loss.item() * label.numel()
        all_scores.extend(logits.detach().cpu().tolist())
        all_labels.extend(label.detach().cpu().tolist())

    return (
        total_loss / max(total_count, 1),
        total_correct / max(total_count, 1),
        binary_auc(all_labels, all_scores),
    )


def train_one_epoch_triplet(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    temperature: float,
    max_steps: int,
) -> Tuple[float, float, float]:
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0
    all_scores: List[float] = []
    all_labels: List[float] = []

    pbar = tqdm(dataloader, desc="train(triplet)", leave=False)
    for step, (feat_a, feat_p, feat_n) in enumerate(pbar, start=1):
        if max_steps > 0 and step > max_steps:
            break

        emb_a = model(*prepare_model_inputs(feat_a, device))
        emb_p = model(*prepare_model_inputs(feat_p, device))
        emb_n = model(*prepare_model_inputs(feat_n, device))

        s_ap = score_similarity(emb_a, emb_p, temperature)
        s_an = score_similarity(emb_a, emb_n, temperature)

        logits = s_ap - s_an
        target = torch.ones_like(logits)
        loss = criterion(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = (logits >= 0).float()
            total_correct += pred.sum().item()
            total_count += pred.numel()
            total_loss += loss.item() * pred.numel()

            all_scores.extend(s_ap.detach().cpu().tolist())
            all_labels.extend([1.0] * s_ap.numel())
            all_scores.extend(s_an.detach().cpu().tolist())
            all_labels.extend([0.0] * s_an.numel())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return (
        total_loss / max(total_count, 1),
        total_correct / max(total_count, 1),
        binary_auc(all_labels, all_scores),
    )


@torch.no_grad()
def evaluate_triplet(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    temperature: float,
    max_steps: int,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    all_scores: List[float] = []
    all_labels: List[float] = []

    pbar = tqdm(dataloader, desc="valid(triplet)", leave=False)
    for step, (feat_a, feat_p, feat_n) in enumerate(pbar, start=1):
        if max_steps > 0 and step > max_steps:
            break

        emb_a = model(*prepare_model_inputs(feat_a, device))
        emb_p = model(*prepare_model_inputs(feat_p, device))
        emb_n = model(*prepare_model_inputs(feat_n, device))

        s_ap = score_similarity(emb_a, emb_p, temperature)
        s_an = score_similarity(emb_a, emb_n, temperature)

        logits = s_ap - s_an
        target = torch.ones_like(logits)
        loss = criterion(logits, target)

        pred = (logits >= 0).float()
        total_correct += pred.sum().item()
        total_count += pred.numel()
        total_loss += loss.item() * pred.numel()

        all_scores.extend(s_ap.detach().cpu().tolist())
        all_labels.extend([1.0] * s_ap.numel())
        all_scores.extend(s_an.detach().cpu().tolist())
        all_labels.extend([0.0] * s_an.numel())

    return (
        total_loss / max(total_count, 1),
        total_correct / max(total_count, 1),
        binary_auc(all_labels, all_scores),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Training demo for multimodal identity alignment")
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--save_path", default="./saved_models/baseline.pth", type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--epochs", default=2, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--max_videos", default=3, type=int)
    parser.add_argument("--temperature", default=10.0, type=float)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--max_train_steps", default=0, type=int, help="0 means use full epoch")
    parser.add_argument("--max_valid_steps", default=0, type=int, help="0 means use full epoch")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--method", default="pair", choices=["pair", "triplet"], help="training sample mode")
    parser.add_argument("--easy_neg_per_anchor", default=1, type=int, help="number of random easy negatives per identity row")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Info] device={device}")
    print(f"[Info] loading dataset method={args.method}...")

    dataset = CrossPlatformDataset(
        data_dir=args.data_dir,
        method=args.method,
        max_videos=args.max_videos,
        easy_neg_per_anchor=args.easy_neg_per_anchor,
        seed=args.seed,
    )
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = max(1, len(dataset) - val_size)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

    collate_fn = align_collate_fn if args.method == "pair" else triplet_align_collate_fn

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = MultiModalAlignment().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    best_path = args.save_path.replace(".pth", f"_{args.method}_best.pth")

    best_val_acc = -1.0
    history: List[Dict] = []

    for epoch in range(1, args.epochs + 1):
        if args.method == "pair":
            train_loss, train_acc, train_auc = train_one_epoch_pair(
                model, train_loader, optimizer, criterion, device, args.temperature, args.max_train_steps
            )
            val_loss, val_acc, val_auc = evaluate_pair(
                model, val_loader, criterion, device, args.temperature, args.max_valid_steps
            )
        else:
            train_loss, train_acc, train_auc = train_one_epoch_triplet(
                model, train_loader, optimizer, criterion, device, args.temperature, args.max_train_steps
            )
            val_loss, val_acc, val_auc = evaluate_triplet(
                model, val_loader, criterion, device, args.temperature, args.max_valid_steps
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_auc": train_auc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_auc": val_auc,
            }
        )

        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_auc={train_auc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f}"
        )

        torch.save(model.state_dict(), args.save_path)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"[Info] new best model saved to {best_path}")

    if history:
        save_metrics(history, args.method)
        plot_metrics(history, args.method)

    print(f"[Done] last model saved to {args.save_path}")


if __name__ == "__main__":
    main()
