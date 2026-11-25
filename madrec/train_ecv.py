"""Fine-tune an ECV checkpoint from annotated, item-disjoint splits."""

import json
import random
from pathlib import Path
from tempfile import TemporaryDirectory

from .ecv import LABEL_IDS, LABELS, encode_pair
from .io import read_jsonl
from .pipeline import string


def read_pairs(path, tokenizer):
    examples = []
    for row in read_jsonl(path):
        passage = string(row["evidence"])
        claim = string(row["claim"])
        if not passage or not claim:
            raise ValueError(f"{path}: evidence and claim must be nonempty")
        label = row["label"]
        if label not in LABEL_IDS:
            raise ValueError(f"{path}: invalid ECV label {label!r}")
        examples.append((encode_pair(tokenizer, passage, claim), LABEL_IDS[label]))
    if not examples:
        raise ValueError(f"no ECV examples in {path}")
    return examples


def verification_metrics(truth, predictions, support_probabilities):
    f1 = []
    for label in LABELS:
        tp = sum(actual == predicted == label for actual, predicted in zip(truth, predictions))
        predicted_count = predictions.count(label)
        actual_count = truth.count(label)
        denominator = predicted_count + actual_count
        f1.append(2 * tp / denominator if denominator else 0.0)
        if label == LABEL_IDS["NotSupported"]:
            precision = tp / predicted_count if predicted_count else 0.0
            recall = tp / actual_count if actual_count else 0.0
    return {
        "macro_f1": sum(f1) / len(f1),
        "not_supported_precision": precision,
        "not_supported_recall": recall,
        "brier": sum((prob - actual) ** 2 for prob, actual in zip(support_probabilities, truth)) / len(truth),
    }


def train(args):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    if args.output.exists():
        raise ValueError(f"choose a new output directory: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=2, id2label=LABELS, label2id=LABEL_IDS,
    ).to(device)

    def collate(examples):
        encoded, labels = zip(*examples)
        batch = tokenizer.pad(list(encoded), return_tensors="pt")
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch

    training = DataLoader(
        read_pairs(args.train, tokenizer), batch_size=args.batch_size, shuffle=True, collate_fn=collate,
    )
    validation = DataLoader(
        read_pairs(args.validation, tokenizer), batch_size=args.batch_size, collate_fn=collate,
    )
    test = None
    if args.test:
        test = DataLoader(
            read_pairs(args.test, tokenizer), batch_size=args.batch_size, collate_fn=collate,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    def evaluate(classifier, loader):
        classifier.eval()
        truth, predictions, support = [], [], []
        with torch.inference_mode():
            for batch in loader:
                truth.extend(batch.pop("labels").tolist())
                logits = classifier(**batch.to(device)).logits.float()
                predictions.extend(logits.argmax(dim=-1).tolist())
                support.extend(logits.softmax(dim=-1)[:, LABEL_IDS["Supported"]].tolist())
        return verification_metrics(truth, predictions, support)

    with TemporaryDirectory(dir=args.output.parent) as temporary:
        checkpoint = Path(temporary) / "checkpoint"
        best_f1 = -1.0
        best_epoch = None
        for epoch in range(1, args.epochs + 1):
            model.train()
            for batch in training:
                optimizer.zero_grad(set_to_none=True)
                loss = model(**batch.to(device)).loss
                loss.backward()
                optimizer.step()
            validation_metrics = evaluate(model, validation)
            print(json.dumps({"epoch": epoch, "validation": validation_metrics}))
            if validation_metrics["macro_f1"] > best_f1:
                best_f1 = validation_metrics["macro_f1"]
                best_epoch = epoch
                model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        report = {"best_epoch": best_epoch, "validation_macro_f1": best_f1}
        if test is not None:
            best = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
            report["test"] = evaluate(best, test)
        (checkpoint / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        checkpoint.rename(args.output)
    print(json.dumps(report))


def add_arguments(parser):
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new directory for the best checkpoint")
    parser.add_argument("--base-model", default="microsoft/deberta-v3-base")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", help="PyTorch device, e.g. cpu or cuda")
