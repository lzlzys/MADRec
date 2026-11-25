"""Frozen evidence–claim classifier used by EG-RAR."""

from .pipeline import probability

LABELS = {0: "NotSupported", 1: "Supported"}
LABEL_IDS = {label: index for index, label in LABELS.items()}


def encode_pair(tokenizer, passage: str, claim: str) -> dict:
    claim_tokens = tokenizer(claim, add_special_tokens=False, verbose=False)["input_ids"]
    if len(claim_tokens) + tokenizer.num_special_tokens_to_add(pair=True) >= 512:
        raise ValueError("ECV claim leaves no token budget for evidence (512 tokens total)")
    return tokenizer(passage, claim, truncation="only_first", max_length=512)


class ECV:
    def __init__(self, checkpoint: str, device: str | None = None, batch_size: int = 16):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        if self.model.config.num_labels != 2:
            raise ValueError("ECV requires a binary classification checkpoint")
        indexes = [
            index
            for index, label in self.model.config.id2label.items()
            if label.strip().lower() == "supported"
        ]
        if len(indexes) != 1:
            raise ValueError("ECV checkpoint config must identify a single Supported label")
        self.supported = int(indexes[0])
        self.model.to(self.device).eval()

    def __call__(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = []
        with self.torch.inference_mode():
            for start in range(0, len(pairs), self.batch_size):
                encoded = [
                    encode_pair(self.tokenizer, passage, claim)
                    for passage, claim in pairs[start:start + self.batch_size]
                ]
                inputs = self.tokenizer.pad(encoded, return_tensors="pt").to(self.device)
                logits = self.model(**inputs).logits
                scores.extend(logits.float().softmax(dim=-1)[:, self.supported].tolist())
        return [probability(value) for value in scores]
