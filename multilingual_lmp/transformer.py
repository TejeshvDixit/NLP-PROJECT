"""Fine-tune a multilingual pair regressor and compare held-out predictions."""
import argparse
import json
import math
from pathlib import Path

from .evaluate import decision, load_rows, report, scores

CHECKPOINT = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEED = 346
MAX_LENGTH = 128


def split(rows):
    train = [r for r in rows if r["domain"] == "tenancy" and r["group"] != "family-05"]
    test = [r for r in rows if r["domain"] == "employment" or r["group"] == "family-05"]
    if len(train) + len(test) != len(rows) or {r["group"] for r in train} & {r["group"] for r in test}:
        raise ValueError("Expected disjoint tenancy training and held-out evaluation groups")
    return train, test


TRAIN_TOPICS = {
    "en": ("The soup is warm.", "A dog chased a ball.", "The train crossed the bridge.",
           "Rain fell throughout the night."),
    "hi": ("सूप गरम है।", "एक कुत्ता गेंद के पीछे दौड़ा।", "रेलगाड़ी पुल पार कर गई।",
           "रात भर बारिश हुई।"),
}
TEST_TOPICS = {
    "en": ("Penguins dive.", "A blue butterfly landed on a flower.",
           "The telescope revealed a distant galaxy."),
    "hi": ("तारे चमकते हैं।", "एक नीली तितली फूल पर बैठी।",
           "दूरबीन से एक दूर की आकाशगंगा दिखाई दी।"),
}


def control_pairs(rows, topics, prefix):
    """Use each distinct original once, within its existing split."""
    originals = {}
    for row in rows:
        originals.setdefault((row["language"], row["original"]), row)
    language_counts = {"en": 0, "hi": 0}
    for index, ((lang, original), row) in enumerate(originals.items()):
        texts = topics[lang]
        unrelated = texts[language_counts[lang] % len(texts)]
        language_counts[lang] += 1
        for check, text, target in (("identical", original, 10), ("unrelated", unrelated, 1)):
            yield dict(language=lang, domain=row["domain"], group=row["group"],
                       check=check, original=original, simplified=text, reference=target,
                       pair_id=f"{prefix}-{index:02d}-{check}")


def encode(tokenizer, rows, device):
    return tokenizer([r["original"] for r in rows], [r["simplified"] for r in rows],
                     max_length=MAX_LENGTH, truncation=True, padding=True, return_tensors="pt").to(device)


def predict(model, tokenizer, rows, device, batch_size):
    import torch

    model.eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            inputs = encode(tokenizer, rows[start:start+batch_size], device)
            values.extend((1 + 9 * model(**inputs).logits.flatten().clamp(0, 1)).tolist())
    return values


def similarities(model, tokenizer, rows, device, batch_size):
    """Compare independent sentence embeddings and symmetric token alignment."""
    import torch
    from torch.nn.functional import normalize

    texts = list(dict.fromkeys(r[field] for r in rows for field in ("original", "simplified")))
    embeddings = {}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start+batch_size]
            inputs = tokenizer(batch, max_length=MAX_LENGTH, truncation=True, padding=True,
                               return_special_tokens_mask=True, return_tensors="pt").to(device)
            attention = inputs["attention_mask"].bool()
            content = attention & ~inputs.pop("special_tokens_mask").bool()
            vectors = model(**inputs).last_hidden_state
            for text, vectors_i, attention_i, content_i in zip(batch, vectors, attention, content):
                pooled = vectors_i[attention_i].mean(dim=0)
                embeddings[text] = (normalize(pooled, dim=0), normalize(vectors_i[content_i], dim=1))
        results = {}
        for row in rows:
            a, a_tokens = embeddings[row["original"]]
            b, b_tokens = embeddings[row["simplified"]]
            aligned = a_tokens @ b_tokens.T
            alignment = ((aligned.max(dim=0).values.mean() + aligned.max(dim=1).values.mean()) / 2
                         if aligned.numel() else torch.tensor(0.0))
            results[row["pair_id"]] = dict(
                minilm_cosine=1 + 9 * a.dot(b).clamp(0, 1).item(),
                minilm_token_alignment=1 + 9 * alignment.clamp(0, 1).item(),
            )
    return results


def train_and_evaluate(args):
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, set_seed

    rows = load_rows(args.annotations)
    train, test = split(rows)
    if not train or not test:
        raise ValueError("Training and test sets must be nonempty")
    set_seed(SEED)
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, revision=args.revision)
    encoder = AutoModel.from_pretrained(args.checkpoint, revision=args.revision).to(device)
    revision = encoder.config._commit_hash
    sanity = []
    for lang, unrelated in (("en", "Penguins dive."), ("hi", "तारे चमकते हैं।")):
        original = next(r["original"] for r in test if r["language"] == lang)
        for name, text in (("identical", original), ("unrelated", unrelated)):
            sanity.append(dict(language=lang, check=name, original=original, simplified=text,
                               pair_id=f"sanity-{lang}-{name}"))
    augmentation = list(control_pairs(train, TRAIN_TOPICS, "anchor")) if args.sanity_anchors else []
    if augmentation:
        sanity = list(control_pairs(test, TEST_TOPICS, "sanity"))
    training_rows = train + augmentation
    predictions = similarities(encoder, tokenizer, test + sanity, device, args.batch_size)
    del encoder
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint, revision=revision or args.revision, num_labels=1, problem_type="regression",
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        order = torch.randperm(len(training_rows)).tolist()
        for start in range(0, len(training_rows), args.batch_size):
            batch = [training_rows[i] for i in order[start:start+args.batch_size]]
            target = torch.tensor([(r["reference"] - 1) / 9 for r in batch], device=device)
            optimizer.zero_grad()
            output = model(**encode(tokenizer, batch, device), labels=target)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(output.loss.item())
        print(f"Epoch {epoch+1}/{args.epochs}: train MSE {sum(losses)/len(losses):.5f}", flush=True)
    for row, value in zip(test + sanity, predict(model, tokenizer, test + sanity, device, args.batch_size), strict=True):
        results = predictions[row["pair_id"]]
        results.update(scores(row["original"], row["simplified"], row["language"])[0])
        results["pair_regression"] = value

    result = dict(label_source="human" if args.annotations else "synthetic",
                  checkpoint=args.checkpoint, revision=revision or args.revision,
                  seed=SEED, epochs=args.epochs, batch_size=args.batch_size,
                  learning_rate=args.learning_rate, device=args.device, torch_version=torch.__version__,
                  train_groups=sorted({r["group"] for r in train}),
                  test_groups=sorted({r["group"] for r in test}),
                  training_pairs=len(train), training_controls=len(augmentation),
                  metrics=report(test, predictions),
                  sanity=[{**r, **predictions[r["pair_id"]]} for r in sanity],
                  predictions=[{**r, **predictions[r["pair_id"]],
                                "decision": decision(predictions[r["pair_id"]]["pair_regression"])}
                               for r in test])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--checkpoint", default=CHECKPOINT)
    train.add_argument("--revision")
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--threads", type=int, default=4)
    train.add_argument("--annotations", nargs="+")
    train.add_argument("--sanity-anchors", action="store_true",
                       help="add exact-copy and off-topic controls from training families")
    train.add_argument("--output", required=True, type=Path)
    train.add_argument("--save-model", type=Path)
    infer = sub.add_parser("score")
    infer.add_argument("model", type=Path)
    infer.add_argument("original")
    infer.add_argument("simplified")
    for command in (train, infer):
        command.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "train":
        if min(args.epochs, args.batch_size, args.threads) < 1 or not (
            math.isfinite(args.learning_rate) and args.learning_rate > 0
        ):
            parser.error("Epochs, batch size, threads and learning rate must be positive")
        train_and_evaluate(args)
    else:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model = AutoModelForSequenceClassification.from_pretrained(args.model).to(args.device)
        # Transformers 4.57 can mistake a locally saved BERT tokenizer for Mistral.
        tokenizer_options = {"fix_mistral_regex": False} if model.config.model_type == "bert" else {}
        tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_options)
        row = dict(original=args.original, simplified=args.simplified)
        value, = predict(model, tokenizer, [row], args.device, 1)
        print(json.dumps(dict(score=value, decision=decision(value))))


if __name__ == "__main__":
    main()
