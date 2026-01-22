"""
EHRXQA Binary Classification (End-to-End)

Trains PyHealth RETAIN as the patient encoder end-to-end on EHRXQA yes/no QA.
Question encoder (Bio_ClinicalBERT) is frozen; RETAIN + QA head are trainable.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

from pyhealth.datasets import SampleEHRDataset, get_dataloader
from pyhealth.models import RETAIN

from extract_patient_vectors import convert_json_entry_to_sample
from extract_question_embeddings import load_model_and_tokenizer


def _set_seed(seed: int) -> None:
    import random

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_entries(json_files: List[Path]) -> List[dict]:
    entries: List[dict] = []
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            entries.extend(data)
        else:
            print(f"Warning: {json_file} is not a list, skipping")
    return entries


def _extract_label(entry: dict) -> int:
    # EHRXQA processed format uses answer=[0] or answer=[1]
    ans = entry.get("answer", [0])
    if isinstance(ans, list) and len(ans) > 0:
        try:
            return int(ans[0])
        except Exception:
            return 0
    try:
        return int(ans)
    except Exception:
        return 0


def build_pyhealth_samples(entries: List[dict], max_visits: int = 0) -> List[dict]:
    """Build PyHealth SampleEHRDataset samples for RETAIN + QA."""
    empty_code = "__EMPTY__"
    samples: List[dict] = []
    for entry in entries:
        base = convert_json_entry_to_sample(entry)
        cond_hist = base.get("cond_hist", [])

        # Normalize to consistent nested list depth for PyHealth validation:
        # expected: List[List[str]] (visits -> codes). Empty histories must still be 2-level, e.g. `[[]]`.
        if not isinstance(cond_hist, list):
            cond_hist = [[]]
        elif len(cond_hist) == 0:
            cond_hist = [[]]
        else:
            # If a flat list of codes sneaks in (List[str]), wrap as a single visit.
            if isinstance(cond_hist[0], str):
                cond_hist = [cond_hist]

            fixed_visits: List[List[str]] = []
            for v in cond_hist:
                if isinstance(v, list):
                    fixed_visits.append([c for c in v if isinstance(c, str)])
                else:
                    fixed_visits.append([])
            cond_hist = fixed_visits
            if len(cond_hist) == 0:
                cond_hist = [[]]

        if max_visits and len(cond_hist) > max_visits:
            cond_hist = cond_hist[-max_visits:]

        # RETAIN (pyhealth 1.1.6) uses pack_padded_sequence; lengths must be > 0.
        # If the whole history is empty (e.g., missing visit_ids), inject a single placeholder code.
        if not any(len(v) > 0 for v in cond_hist):
            cond_hist = [[empty_code]]

        q = entry.get("question", "")
        if not q or not isinstance(q, str):
            q = ""

        label = _extract_label(entry)

        samples.append(
            {
                "patient_id": str(base.get("patient_id", entry.get("patient_id", "unknown"))),
                "visit_id": str(base.get("visit_id", entry.get("visit_id", "unknown_visit"))),
                # RETAIN expects (dim=3,type=str): visits -> codes
                "list_list_codes": cond_hist,
                "question": q,
                "label": int(label),
            }
        )
    return samples


def split_samples(
    samples: List[dict], train_ratio: float, val_ratio: float, seed: int
) -> Tuple[List[dict], List[dict], List[dict]]:
    n = len(samples)
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)

    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    def _take(idxs: np.ndarray) -> List[dict]:
        return [samples[i] for i in idxs.tolist()]

    return _take(train_idx), _take(val_idx), _take(test_idx)


@torch.no_grad()
def encode_questions_mean_pool(
    questions: List[str],
    tokenizer,
    question_model,
    device: torch.device,
) -> torch.Tensor:
    """Encode questions into (B, hidden) tensor on `device` (BERT is frozen)."""
    questions = [q.strip() if isinstance(q, str) else "" for q in questions]
    encoded = tokenizer(
        questions,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    outputs = question_model(input_ids=input_ids, attention_mask=attention_mask)
    hidden_states = outputs.last_hidden_state  # (B, T, H)

    mask_exp = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
    sum_emb = torch.sum(hidden_states * mask_exp, dim=1)  # (B, H)
    sum_mask = torch.clamp(mask_exp.sum(dim=1), min=1e-9)  # (B, H)
    return sum_emb / sum_mask


class RetainQAHead(nn.Module):
    """QA head: project question embedding, concat with patient embedding, then classify yes/no."""

    def __init__(self, patient_dim: int, question_dim: int, hidden_dim: int = 0):
        super().__init__()
        if hidden_dim <= 0:
            hidden_dim = 2 * patient_dim

        self.question_proj = nn.Linear(question_dim, patient_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * patient_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, patient_emb: torch.Tensor, question_emb: torch.Tensor) -> torch.Tensor:
        q_proj = self.question_proj(question_emb)  # (B, patient_dim)
        x = torch.cat([patient_emb, q_proj], dim=1)
        return self.mlp(x)  # (B, 2)


def _labels_to_tensor(labels, device: torch.device) -> torch.Tensor:
    # pyhealth dataloader yields `label` as a python list (e.g., [0,1,0,...])
    if isinstance(labels, list):
        return torch.tensor(labels, dtype=torch.long, device=device)
    if torch.is_tensor(labels):
        return labels.to(device=device, dtype=torch.long)
    return torch.tensor(list(labels), dtype=torch.long, device=device)


@torch.no_grad()
def evaluate_e2e(
    retain_model: RETAIN,
    head: RetainQAHead,
    data_loader,
    tokenizer,
    question_model,
    device: torch.device,
) -> Dict[str, float]:
    retain_model.eval()
    head.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    for batch in data_loader:
        # patient embedding from RETAIN (pyhealth 1.1.6 supports embed=True)
        out = retain_model(
            list_list_codes=batch["list_list_codes"],
            label=batch["label"],
            embed=True,
        )
        patient_emb = out["embed"]  # (B, D)

        # question embedding from frozen BERT
        q_emb = encode_questions_mean_pool(batch["question"], tokenizer, question_model, device)

        logits = head(patient_emb, q_emb)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
        labels = _labels_to_tensor(batch["label"], device).detach().cpu().numpy().tolist()

        all_preds.extend(preds)
        all_labels.extend(labels)

    return {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "f1": float(f1_score(all_labels, all_preds, average="binary")),
    }


def train_e2e(
    retain_model: RETAIN,
    head: RetainQAHead,
    train_loader,
    val_loader,
    test_loader,
    tokenizer,
    question_model,
    device: torch.device,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    min_delta: float,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        list(retain_model.parameters()) + list(head.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val_f1 = -float("inf")
    patience_counter = 0
    best_state: Dict[str, Dict[str, torch.Tensor]] = {}

    print("\nTrainable params:")
    trainable = sum(
        p.numel()
        for p in list(retain_model.parameters()) + list(head.parameters())
        if p.requires_grad
    )
    print(f"  RETAIN+head trainable parameters: {trainable}")

    last_val_metrics: Dict[str, float] = {"accuracy": 0.0, "f1": 0.0}

    for epoch in range(1, epochs + 1):
        retain_model.train()
        head.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            out = retain_model(
                list_list_codes=batch["list_list_codes"],
                label=batch["label"],
                embed=True,
            )
            patient_emb = out["embed"]

            q_emb = encode_questions_mean_pool(batch["question"], tokenizer, question_model, device)
            logits = head(patient_emb, q_emb)
            labels = _labels_to_tensor(batch["label"], device)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        val_metrics = evaluate_e2e(retain_model, head, val_loader, tokenizer, question_model, device)
        last_val_metrics = val_metrics
        val_f1 = val_metrics["f1"]

        print(
            f"Epoch {epoch:02d} | loss={avg_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | val_f1={val_metrics['f1']:.4f}"
        )

        if val_f1 > best_val_f1 + min_delta:
            best_val_f1 = val_f1
            patience_counter = 0
            best_state = {
                "retain": {k: v.detach().cpu().clone() for k, v in retain_model.state_dict().items()},
                "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
            }
            print(f"  → New best val_f1: {best_val_f1:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping: no improvement for {patience} epochs. Restoring best weights.")
            break

    if best_state:
        retain_model.load_state_dict(best_state["retain"])
        head.load_state_dict(best_state["head"])

    if test_loader is not None:
        test_metrics = evaluate_e2e(retain_model, head, test_loader, tokenizer, question_model, device)
        print(f"\n[TEST] acc={test_metrics['accuracy']:.4f} f1={test_metrics['f1']:.4f}")
        final_metrics = test_metrics
    else:
        final_metrics = last_val_metrics

    ckpt_state = {
        "retain_state_dict": retain_model.state_dict(),
        "head_state_dict": head.state_dict(),
    }
    return ckpt_state, final_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train end-to-end RETAIN for EHRXQA yes/no QA")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/data/yuyu/data/EHRXQA/ehrxqa/dataset",
        help="Directory containing *_processed.json files",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retain_embed_dim", type=int, default=128)
    parser.add_argument(
        "--max_visits",
        type=int,
        default=0,
        help="If >0, keep only the most recent N visits in cond_hist for RETAIN",
    )
    parser.add_argument(
        "--question_model_name",
        type=str,
        default="emilyalsentzer/Bio_ClinicalBERT",
    )

    args = parser.parse_args()
    _set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_dir = Path(args.dataset_dir)
    json_files = list(dataset_dir.glob("*_processed.json"))
    if not json_files:
        raise RuntimeError(f"No *_processed.json files found in {dataset_dir}")

    print(f"Found {len(json_files)} JSON files:")
    for f in json_files:
        print(f"  - {f.name}")

    print("\nLoading entries...")
    entries = _load_entries(json_files)
    print(f"Loaded {len(entries)} total entries")

    print("\nBuilding PyHealth samples...")
    samples = build_pyhealth_samples(entries, max_visits=args.max_visits)
    train_samples, val_samples, test_samples = split_samples(
        samples, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed
    )
    print(f"Split: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")

    # Build a vocab dataset for RETAIN tokenizers (use all samples for stability)
    vocab_dataset = SampleEHRDataset(samples=samples, dataset_name="ehrxqa_vocab")
    train_dataset = SampleEHRDataset(samples=train_samples, dataset_name="ehrxqa_train")
    val_dataset = SampleEHRDataset(samples=val_samples, dataset_name="ehrxqa_val")
    test_dataset = SampleEHRDataset(samples=test_samples, dataset_name="ehrxqa_test")

    train_loader = get_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # RETAIN patient encoder
    retain_model = RETAIN(
        dataset=vocab_dataset,
        feature_keys=["list_list_codes"],
        label_key="label",
        mode="binary",
        embedding_dim=args.retain_embed_dim,
    ).to(device)

    # Frozen question encoder
    print("\nLoading question encoder (frozen)...")
    question_tokenizer, question_model, _ = load_model_and_tokenizer(args.question_model_name, str(device))
    question_model = question_model.to(device)
    question_model.eval()
    for p in question_model.parameters():
        p.requires_grad = False

    # QA head (patient_dim equals embedding_dim for single feature)
    question_dim = int(question_model.config.hidden_size)
    patient_dim = int(args.retain_embed_dim)
    head = RetainQAHead(patient_dim=patient_dim, question_dim=question_dim).to(device)

    # One-time debug: ensure we can fetch embedding from RETAIN
    debug_batch = next(iter(val_loader))
    debug_out = retain_model(
        list_list_codes=debug_batch["list_list_codes"],
        label=debug_batch["label"],
        embed=True,
    )
    print(f"RETAIN output keys: {list(debug_out.keys())} | embed shape: {tuple(debug_out['embed'].shape)}")

    ckpt_state, final_metrics = train_e2e(
        retain_model=retain_model,
        head=head,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        tokenizer=question_tokenizer,
        question_model=question_model,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        min_delta=args.min_delta,
    )

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = checkpoint_dir / f"ehrxqa_retain_e2e_{timestamp}.pth"

    torch.save(
        {
            **ckpt_state,
            "pyhealth_version": "1.1.6",
            "retain_embed_dim": args.retain_embed_dim,
            "question_model_name": args.question_model_name,
            "final_metrics": final_metrics,
            "args": vars(args),
        },
        checkpoint_path,
    )
    print(f"\nSaved checkpoint to: {checkpoint_path}")
    print(f"Final metrics: {final_metrics}")


if __name__ == "__main__":
    main()
