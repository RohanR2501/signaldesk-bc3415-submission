"""One bounded CUDA DistilBERT experiment using exact Phase 2A memberships.

Run as python -m signaldesk_ml.transformer_train train|verify. Test/full-cohort files
are denied by an audit hook before any data IO. No prediction API is registered.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from signaldesk.config import PROJECT_ROOT
from signaldesk_data.source import file_hash, write_json
from signaldesk_ml.baselines import load_data, metrics, prepare_annotations
from signaldesk_ml.transformer_inference import (
    LABELS,
    SentimentTransformer,
    load_transformer,
    require_cuda,
)


def seal_test(event: str, args: tuple[Any, ...]) -> None:
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        name = os.fsdecode(args[0]).replace("\\", "/").rsplit("/", 1)[-1].lower()
        if name in {"test.parquet", "sentiment.parquet", "split_manifest.parquet"}:
            raise PermissionError("Frozen test and full-cohort access is forbidden")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def training_class_weights(targets: list[int]) -> dict[str, Any]:
    """Square-root inverse frequency, E_training[weight] = 1; canonical label order."""
    if not targets or any(target not in range(len(LABELS)) for target in targets):
        raise ValueError("Training targets must use the canonical three-class mapping")
    counts = np.bincount(targets, minlength=len(LABELS))
    if np.any(counts == 0):
        raise ValueError("All three training classes must be present")
    frequencies = counts / counts.sum()
    raw = 1 / np.sqrt(frequencies)
    weights = raw / np.sum(frequencies * raw)
    return {
        "label_order": LABELS,
        "counts": counts.tolist(),
        "frequencies": frequencies.tolist(),
        "weights": weights.tolist(),
        "cuda_float32_weights": weights.astype(np.float32).tolist(),
        "training_frequency_weighted_mean": float(np.sum(frequencies * weights)),
        "formula": "raw=1/sqrt(n_class/n_train); w=raw/sum(frequency*raw)",
        "source": "training labels only",
        "reduction": "weighted mean over each effective accumulation group",
    }


def classification_loss(
    model: Any, batch: dict[str, torch.Tensor], class_weights: torch.Tensor | None = None
) -> torch.Tensor:
    if class_weights is None:
        return model(**batch).loss  # type: ignore[no-any-return]
    logits = model(**{key: value for key, value in batch.items() if key != "labels"}).logits
    return torch.nn.functional.cross_entropy(logits.float(), batch["labels"], weight=class_weights)


def accumulation_fraction(
    targets: list[int], chosen: list[int], group: list[int], weights: list[float] | None
) -> float:
    """Make accumulated microbatch gradients equal the effective-batch mean loss."""
    if weights is None:
        return len(chosen) / len(group)
    return sum(weights[targets[i]] for i in chosen) / sum(weights[targets[i]] for i in group)


def controlled_reference(config: dict[str, Any]) -> dict[str, Any] | None:
    if config["loss"] == "unweighted_cross_entropy":
        return None
    if config["loss"] != "sqrt_inverse_frequency_cross_entropy":
        raise ValueError("Unsupported training loss")
    reference: dict[str, Any] = json.loads(
        (PROJECT_ROOT / config["reference_directory"] / "run.json").read_text(encoding="utf-8")
    )
    allowed = {
        "version",
        "artifact_directory",
        "maximum_training_seconds",
        "loss",
        "reference_directory",
    }
    differences = {
        key
        for key in set(config) | set(reference["configuration"])
        if config.get(key) != reference["configuration"].get(key)
    }
    if differences - allowed:
        raise ValueError("Controlled experiment changed a non-loss training condition")
    if config["maximum_training_seconds"] != 1500 or reference["max_length"] != 256:
        raise ValueError("Phase 2C requires a 25-minute budget and the original 256-token limit")
    if config["artifact_directory"] != "artifacts/sentiment/distilbert-weighted-v1":
        raise ValueError("Weighted experiment must use its separate approved directory")
    return reference


def approved_data(config: dict[str, Any]) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    baseline_dir = PROJECT_ROOT / config["baseline_directory"]
    baseline = json.loads((baseline_dir / "run.json").read_text(encoding="utf-8"))
    train, validation, dataset = load_data(baseline["configuration"])
    if dataset != baseline["dataset"]:
        raise ValueError("Dataset differs from Phase 2A")
    provenance = PROJECT_ROOT / baseline["configuration"]["annotation_directory"]
    if not (provenance / "internal/provenance.json").exists():
        raise ValueError("Existing blind reservation is required before Phase 2B")
    annotation = prepare_annotations(validation, baseline["configuration"], dataset)
    membership_path = baseline_dir / "membership.json"
    if file_hash(membership_path) != baseline["membership_sha256"]:
        raise ValueError("Phase 2A membership checksum mismatch")
    membership = json.loads(membership_path.read_text(encoding="utf-8"))
    reserved = {row["review_id"] for row in annotation["reserved"]}
    scoring = validation.filter(~pl.col("review_id").is_in(sorted(reserved)))
    validate_membership(
        train["review_id"].to_list(), scoring["review_id"].to_list(), reserved, membership
    )
    if (train.height, scoring.height, len(reserved)) != (32230, 6673, 240):
        raise ValueError("Approved Phase 2B counts changed")
    prediction_path = baseline_dir / "validation_predictions.parquet"
    if file_hash(prediction_path) != baseline["files"][prediction_path.name]:
        raise ValueError("Baseline predictions changed")
    reference = pl.read_parquet(prediction_path)
    for column in ["review_id", "sentiment_proxy"]:
        if not reference[column].equals(scoring[column]):
            raise ValueError("Baseline scoring IDs or labels differ")
    if (
        metrics(
            reference["sentiment_proxy"].to_list(),
            reference["logistic_regression_balanced"].to_list(),
        )
        != baseline["models"]["logistic_regression_balanced"]["metrics"]
    ):
        raise ValueError("Baseline metrics do not reproduce")
    return (
        train,
        scoring,
        {
            "dataset": dataset,
            "membership_sha256": file_hash(membership_path),
            "baseline_run_sha256": file_hash(baseline_dir / "run.json"),
            "annotation_provenance_sha256": file_hash(provenance / "internal/provenance.json"),
            "annotation_template_sha256": file_hash(provenance / "annotation_template.csv"),
        },
    )


def validate_membership(
    train: list[str], scoring: list[str], reserved: set[str], membership: dict[str, Any]
) -> None:
    if (
        train != membership["training_ids"]
        or scoring != membership["scoring_ids"]
        or reserved != set(membership["annotation_ids"])
    ):
        raise ValueError("Exact Phase 2A ordered memberships must be reused")
    if set(train) & (set(scoring) | reserved) or set(scoring) & reserved:
        raise ValueError("Annotation/validation contamination")


def length_profile(train: pl.DataFrame, tokenizer: Any, config: dict[str, Any]) -> dict[str, Any]:
    rows = sorted(
        train.select("review_id", "model_text").to_dicts(),
        key=lambda row: hashlib.sha256(
            f"{config['seed']}:length:{row['review_id']}".encode()
        ).digest(),
    )
    sample = rows[: config["length_sample_size"]]
    lengths = np.array(
        [
            len(ids)
            for ids in tokenizer(
                [row["model_text"] for row in sample],
                truncation=False,
                verbose=False,
            )["input_ids"]
        ]
    )
    # Predeclared training-only rule: 128 if >=95% fit, otherwise 256.
    maximum = 128 if float(np.mean(lengths > 128)) <= 0.05 else 256
    return {
        "sample_rows": len(sample),
        "sample_ids_sha256": hashlib.sha256(
            "\n".join(row["review_id"] for row in sample).encode()
        ).hexdigest(),
        "percentiles": {str(p): float(np.percentile(lengths, p)) for p in [50, 90, 95, 99]},
        "fraction_over_128": float(np.mean(lengths > 128)),
        "fraction_over_256": float(np.mean(lengths > 256)),
        "max_length": maximum,
        "rule": "128 if at least 95% of training sample fits, otherwise 256; right truncation",
    }


def training_batch(
    encoded: Any, targets: list[int], indices: list[int], tokenizer: Any
) -> dict[str, torch.Tensor]:
    features = [{key: encoded[key][i] for key in ["input_ids", "attention_mask"]} for i in indices]
    batch: dict[str, torch.Tensor] = dict(
        tokenizer.pad(features, padding=True, pad_to_multiple_of=8, return_tensors="pt")
    )
    batch["labels"] = torch.tensor([targets[i] for i in indices], dtype=torch.long)
    return {key: value.to("cuda:0") for key, value in batch.items()}


def preflight(
    model: Any,
    tokenizer: Any,
    encoded: Any,
    targets: list[int],
    config: dict[str, Any],
    max_length: int,
    class_weights: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Full optimizer memory probe; restore all weights/RNG before the sole real run."""
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    initial_hash = hashlib.sha256()
    for key, value in initial.items():
        initial_hash.update(key.encode())
        initial_hash.update(value.numpy().tobytes())
    chosen = config["batch_size"]
    for batch_size in [chosen, chosen // 2, chosen // 4]:
        try:
            model.train()
            torch.cuda.reset_peak_memory_stats()
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
            scaler = torch.amp.GradScaler("cuda")
            # Worst-length real training inputs padded to the selected maximum.
            indices = sorted(
                range(len(targets)), key=lambda i: len(encoded["input_ids"][i]), reverse=True
            )[:batch_size]
            batch = training_batch(encoded, targets, indices, tokenizer)
            torch.cuda.synchronize()
            started = perf_counter()
            for _ in range(6):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = classification_loss(model, batch, class_weights)
                torch.autograd.backward(scaler.scale(loss))
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            torch.cuda.synchronize()
            seconds = perf_counter() - started
            reserved = torch.cuda.max_memory_reserved()
            allocated = torch.cuda.max_memory_allocated()
            if reserved > 0.9 * torch.cuda.get_device_properties(0).total_memory:
                raise torch.cuda.OutOfMemoryError("Probe exceeds 90% of device VRAM")
            result = {
                "batch_size": batch_size,
                "steps": 6,
                "seconds": seconds,
                "tokens_per_sequence": int(batch["input_ids"].shape[1]),
                "peak_allocated_bytes": allocated,
                "peak_reserved_bytes": reserved,
                "estimated_two_epoch_seconds": seconds
                / 6
                * math.ceil(len(targets) / batch_size)
                * config["epochs"],
                "weights_and_rng_restored": True,
                "max_length": max_length,
                "initialization_sha256": initial_hash.hexdigest(),
            }
            del optimizer, scaler, batch, loss
            model.zero_grad(set_to_none=True)
            model.load_state_dict(initial)
            torch.cuda.empty_cache()
            seed_everything(config["seed"])
            return result
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            model.load_state_dict(initial)
            torch.cuda.empty_cache()
    raise RuntimeError("CUDA preflight cannot fit safely; no CPU fallback")


def verify(config: dict[str, Any]) -> dict[str, Any]:
    require_cuda()
    training, scoring, integrity = approved_data(config)
    directory = PROJECT_ROOT / config["artifact_directory"]
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if integrity != run["integrity"] or config != run["configuration"]:
        raise ValueError("Run inputs/configuration changed")
    if controlled_reference(config) is not None:
        expected = training_class_weights(
            [LABELS.index(label) for label in training["sentiment_proxy"]]
        )
        if expected != run["class_weighting"]:
            raise ValueError("Saved class weights differ from training-only frequencies")
    for name, checksum in run["output_files"].items():
        if file_hash(directory / name) != checksum:
            raise ValueError("Evaluation artifact changed")
    model = load_transformer(directory)
    predictions = model.predict_texts(scoring["model_text"].to_list())
    stored = pl.read_parquet(directory / "validation_predictions.parquet")
    if (
        predictions != stored["prediction"].to_list()
        or scoring["review_id"].to_list() != stored["review_id"].to_list()
        or scoring["sentiment_proxy"].to_list() != stored["sentiment_proxy"].to_list()
        or metrics(scoring["sentiment_proxy"].to_list(), predictions) != run["metrics"]
    ):
        raise ValueError("Saved CUDA checkpoint predictions/metrics do not reproduce")
    result = {
        "status": "passed",
        "cuda_used": True,
        "scored_validation_rows": 6673,
        "training_rows": 32230,
        "annotation_rows_excluded": 240,
        "exact_membership_verified": True,
        "saved_predictions_reproduced": True,
        "test_accessed": False,
        "retraining_performed": False,
    }
    write_json(directory / "verification.json", result)
    return result


def train(config: dict[str, Any]) -> dict[str, Any]:
    require_cuda()
    directory = PROJECT_ROOT / config["artifact_directory"]
    if (directory / "run.json").exists():
        return verify(config)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Run exists or is partial; refusing overwrite/retraining")
    train_frame, scoring, integrity = approved_data(config)
    reference = controlled_reference(config)
    if reference is not None and integrity != reference["integrity"]:
        raise ValueError("Phase 2B/2C integrity references differ")
    seed_everything(config["seed"])
    torch.set_num_threads(4)
    pretrained = json.loads((PROJECT_ROOT / ".runtime/pretrained.json").read_text())
    if pretrained["revision"] != config["revision"] or pretrained["model"] != config["model"]:
        raise ValueError("Pretrained revision mismatch")
    pretrained_files = {
        path.name: file_hash(path) for path in Path(pretrained["path"]).iterdir() if path.is_file()
    }
    if reference is not None and pretrained_files != reference["pretrained_files"]:
        raise ValueError("Original pretrained weights/tokenizer changed")
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained["path"], local_files_only=True, trust_remote_code=False
    )
    profile = length_profile(train_frame, tokenizer, config)
    maximum = profile["max_length"]
    if reference is not None and (maximum != 256 or profile != reference["length_profile"]):
        raise ValueError("Original training-only token-length profile changed")
    model: Any = AutoModelForSequenceClassification.from_pretrained(
        pretrained["path"],
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        num_labels=3,
        id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
        attn_implementation="sdpa",
    ).to("cuda:0")
    encoded = tokenizer(
        train_frame["model_text"].to_list(), truncation=True, max_length=maximum, padding=False
    )
    targets = [LABELS.index(label) for label in train_frame["sentiment_proxy"]]
    weighting = training_class_weights(targets) if reference is not None else None
    class_weights = (
        torch.tensor(weighting["weights"], dtype=torch.float32, device="cuda:0")
        if weighting is not None
        else None
    )
    probe = preflight(model, tokenizer, encoded, targets, config, maximum, class_weights)
    estimate = probe["estimated_two_epoch_seconds"]
    if reference is not None:
        if probe["batch_size"] != 16:
            raise RuntimeError("Cannot preserve batch size 16 within GPU memory; stopped")
        # The worst-length probe overestimated Phase 2B by ~3x. Anchor the estimate
        # to its measured dynamic-padding training time, scaled by relative probe time.
        estimate = (
            reference["training_seconds"] * probe["seconds"] / reference["preflight"]["seconds"]
        )
        probe["reference_adjusted_estimated_seconds"] = estimate
    print(
        json.dumps(
            {"length_profile": profile, "gpu_preflight": probe, "class_weighting": weighting}
        ),
        flush=True,
    )
    if estimate > config["maximum_training_seconds"]:
        raise RuntimeError("GPU estimate exceeds the configured time budget; training not started")
    directory.mkdir(parents=True)
    write_json(
        directory / "started.json",
        {
            "started_at": datetime.now(UTC).isoformat(),
            "configuration": config,
            "integrity": integrity,
            "profile": profile,
            "preflight": probe,
            "class_weighting": weighting,
        },
    )
    batch_size = probe["batch_size"]
    accumulation = config["gradient_accumulation"] * (config["batch_size"] // batch_size)
    batches = math.ceil(len(targets) / batch_size)
    total_updates = math.ceil(batches / accumulation) * config["epochs"]
    warmup = max(1, int(total_updates * config["warmup_fraction"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            step / warmup
            if step < warmup
            else max(0, (total_updates - step) / (total_updates - warmup))
        ),
    )
    scaler = torch.amp.GradScaler("cuda")
    predictor = SentimentTransformer(model, tokenizer, maximum)
    history = []
    best_f1 = -1.0
    selected_epoch = 0
    training_seconds = 0.0
    torch.cuda.reset_peak_memory_stats()
    wall_start = perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        indices = torch.randperm(
            len(targets), generator=torch.Generator().manual_seed(config["seed"] + epoch)
        ).tolist()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        epoch_start = perf_counter()
        summed_loss = 0.0
        for index in range(batches):
            if perf_counter() - wall_start > config["maximum_training_seconds"]:
                raise RuntimeError("Configured training wall budget reached; stopped")
            chosen = indices[index * batch_size : (index + 1) * batch_size]
            batch = training_batch(encoded, targets, chosen, tokenizer)
            # Weight by actual examples per accumulation group, including the final remainder.
            group_start = (index // accumulation) * accumulation * batch_size
            group_count = min(accumulation * batch_size, len(targets) - group_start)
            group = indices[group_start : group_start + group_count]
            with torch.autocast("cuda", dtype=torch.float16):
                loss = classification_loss(model, batch, class_weights)
                scaled_loss = loss * accumulation_fraction(
                    targets, chosen, group, weighting["cuda_float32_weights"] if weighting else None
                )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite training loss")
            torch.autograd.backward(scaler.scale(scaled_loss))
            batch_mass = (
                sum(weighting["cuda_float32_weights"][targets[i]] for i in chosen)
                if weighting
                else len(chosen)
            )
            summed_loss += float(loss.detach()) * batch_mass
            if (index + 1) % accumulation == 0 or index + 1 == batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (index + 1) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "batch": index + 1,
                            "batches": batches,
                            "elapsed_seconds": round(perf_counter() - epoch_start, 1),
                        }
                    ),
                    flush=True,
                )
        torch.cuda.synchronize()
        epoch_seconds = perf_counter() - epoch_start
        training_seconds += epoch_seconds
        started = perf_counter()
        predictions = predictor.predict_texts(
            scoring["model_text"].to_list(), config["evaluation_batch_size"]
        )
        torch.cuda.synchronize()
        evaluation_seconds = perf_counter() - started
        score = metrics(scoring["sentiment_proxy"].to_list(), predictions)
        record = {
            "epoch": epoch,
            "training_seconds": epoch_seconds,
            "mean_training_loss": summed_loss
            / (
                sum(weighting["cuda_float32_weights"][target] for target in targets)
                if weighting
                else len(targets)
            ),
            "metrics": score,
            "validation_seconds": evaluation_seconds,
        }
        history.append(record)
        write_json(directory / "epoch_history.json", history)
        print(json.dumps(record), flush=True)
        if score["macro_f1"] > best_f1:
            best_f1 = score["macro_f1"]
            selected_epoch = epoch
            model.save_pretrained(directory / "checkpoint", safe_serialization=True)
            tokenizer.save_pretrained(directory / "checkpoint")
            write_json(
                directory / "checkpoint/class_mapping.json",
                {str(i): label for i, label in enumerate(LABELS)},
            )
            scoring.select("review_id", "sentiment_proxy").with_columns(
                pl.Series("prediction", predictions)
            ).write_parquet(directory / "validation_predictions.parquet")
    run = {
        "configuration": config,
        "class_weighting": weighting,
        "initialization": (
            "Pinned original pretrained encoder and seeded new head; not fine-tuned weights"
        ),
        "reference_run_sha256": (
            file_hash(PROJECT_ROOT / config["reference_directory"] / "run.json")
            if reference is not None
            else None
        ),
        "integrity": integrity,
        "max_length": maximum,
        "length_profile": profile,
        "preflight": probe,
        "history": history,
        "selected_epoch": selected_epoch,
        "selection_rule": "Highest validation Macro-F1; earliest tie",
        "metrics": history[selected_epoch - 1]["metrics"],
        "training_seconds": training_seconds,
        "training_and_evaluation_wall_seconds": perf_counter() - wall_start,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "gpu": torch.cuda.get_device_name(0),
        "cuda_device": str(next(model.parameters()).device),
        "cuda_runtime": torch.version.cuda,
        "precision": "CUDA autocast fp16 + GradScaler; fp32 weights",
        "batch_size": batch_size,
        "gradient_accumulation": accumulation,
        "class_counts_training": {label: targets.count(i) for i, label in enumerate(LABELS)},
        "label_source": "rating_proxy_v1",
        "features": ["model_text"],
        "test_accessed": False,
        "data_access_guard": "Python open audit hook denies sealed parquets",
        "uv_lock_sha256": file_hash(PROJECT_ROOT / "uv.lock"),
        "code_sha256": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): file_hash(path)
            for path in sorted((PROJECT_ROOT / "ml/signaldesk_ml").glob("*.py"))
        },
        "environment": {
            name: importlib.metadata.version(name)
            for name in ["torch", "transformers", "accelerate", "numpy", "polars", "scikit-learn"]
        },
        "pretrained_files": pretrained_files,
        "checkpoint_files": {
            str(path.relative_to(directory)).replace("\\", "/"): file_hash(path)
            for path in (directory / "checkpoint").iterdir()
            if path.is_file()
        },
        "output_files": {
            name: file_hash(directory / name)
            for name in ["validation_predictions.parquet", "epoch_history.json"]
        },
        "finished_at": datetime.now(UTC).isoformat(),
    }
    run["checkpoint_bytes"] = sum(
        path.stat().st_size for path in (directory / "checkpoint").iterdir() if path.is_file()
    )
    write_json(directory / "run.json", run)
    del model, predictor, optimizer, scaler
    torch.cuda.empty_cache()
    return verify(config)


def main() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.addaudithook(seal_test)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "verify"])
    parser.add_argument("--config", default="config/sentiment-transformer.json")
    arguments = parser.parse_args()
    config = json.loads((PROJECT_ROOT / arguments.config).read_text())
    print(json.dumps(train(config) if arguments.command == "train" else verify(config)), flush=True)


if __name__ == "__main__":
    main()
