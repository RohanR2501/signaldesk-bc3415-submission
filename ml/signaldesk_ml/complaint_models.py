"""One bounded, training-only weak-supervision experiment. No human-label reader."""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from signaldesk.config import PROJECT_ROOT
from signaldesk_data.source import file_hash, write_json
from signaldesk_ml.complaint_inference import (
    LABELS,
    ComplaintModel,
    ComplaintNLI,
    ScoreArray,
    load_complaint_model,
)
from signaldesk_ml.complaint_preparation import DISCOVERY, ranked, read_allowed
from signaldesk_ml.complaint_rules import RULE_VERSION, TAXONOMY_VERSION, detect_complaints

CONFIG = PROJECT_ROOT / "config/complaint-models.json"
CONTEXT = re.compile(
    r"\b(?:no|not|never|without|old|previous|replaced|resolved|fixed|refund|repaired)\b", re.I
)


def deny_development_files(event: str, args: tuple[Any, ...]) -> None:
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.fsdecode(args[0]).replace("\\", "/").lower()
        if path.rsplit("/", 1)[-1] in {
            "test.parquet",
            "sentiment.parquet",
            "split_manifest.parquet",
            "investigation.parquet",
        } or ("/annotations/" in path and path.endswith(".csv")):
            raise PermissionError("Protected data cannot be opened for complaint development")


def configuration() -> dict[str, Any]:
    return dict(json.loads(CONFIG.read_text()))


def select_rows(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    excluded_ids: set[str],
    excluded_groups: set[str],
) -> list[dict[str, Any]]:
    """No labels/metadata features: deterministic diversity, enrichment then broad top-up."""
    eligible = [
        row
        for row in rows
        if row["review_id"] not in excluded_ids and row["duplicate_group_id"] not in excluded_groups
    ]
    eligible.sort(key=lambda row: ranked(row["review_id"], "complaint-model-selection"))
    selected: dict[str, dict[str, Any]] = {}
    families: set[str] = set()

    def add(candidates: list[dict[str, Any]], size: int, origin: str) -> None:
        count = 0
        for row in candidates:
            if len(selected) >= config["sample_size"] or count >= size:
                break
            identifier = row["review_id"]
            if identifier in selected or row["duplicate_group_id"] in families:
                continue
            selected[identifier] = {**row, "sampling_origin": origin}
            families.add(row["duplicate_group_id"])
            count += 1

    def hits(row: dict[str, Any], label: str) -> bool:
        return re.search(DISCOVERY[label], row["model_text"], re.I) is not None

    add(eligible, config["broad_size"], "broad")
    for label in LABELS:
        add(
            [row for row in eligible if hits(row, label)],
            config["per_category_cap"],
            f"keyword:{label}",
        )
    add(
        [row for row in eligible if not any(hits(row, label) for label in LABELS)],
        config["no_keyword_size"],
        "no_keyword",
    )
    add(
        [
            row
            for row in eligible
            if CONTEXT.search(row["model_text"]) and any(hits(row, label) for label in LABELS)
        ],
        config["context_size"],
        "context",
    )
    add(eligible, config["sample_size"], "broad_topup")
    if len(selected) != config["sample_size"]:
        raise ValueError("Insufficient eligible original reviews for configured sample")
    result = sorted(
        selected.values(), key=lambda row: ranked(row["review_id"], "complaint-display")
    )
    groups = sorted(families, key=lambda group: ranked(group, "complaint-holdout"))
    holdout = set(groups[: round(len(groups) * config["holdout_fraction"])])
    for row in result:
        row["development_split"] = "holdout" if row["duplicate_group_id"] in holdout else "fit"
        row["source_partition"] = "train"
    return result


def prepare(directory: Path, config: dict[str, Any]) -> dict[str, Any]:
    if (directory / "selection.json").exists():
        saved: dict[str, Any] = json.loads((directory / "selection.json").read_text())
        if saved["configuration"] != config:
            raise ValueError("Frozen configuration changed; existing run cannot be retuned")
        if file_hash(directory / "development.parquet") != saved["development_sha256"]:
            raise ValueError("Frozen development sample changed")
        return saved
    if directory.exists():
        raise ValueError("Partial selection exists; do not overwrite")
    base = PROJECT_ROOT / "artifacts/sentiment/baselines-v1"
    baseline = json.loads((base / "run.json").read_text())
    if file_hash(base / "membership.json") != baseline["membership_sha256"]:
        raise ValueError("Protected sentiment membership changed")
    members = json.loads((base / "membership.json").read_text())
    provenance_path = (
        PROJECT_ROOT / "data/annotations/sentiment-v1/complaints-v1/internal/provenance.json"
    )
    provenance = json.loads(provenance_path.read_text())
    if set(provenance["primary_ids"]) != set(members["annotation_ids"]):
        raise ValueError("Annotation reservations disagree")
    validation = read_allowed("validation", ["review_id", "duplicate_group_id"])
    primary = validation.filter(pl.col("review_id").is_in(members["annotation_ids"]))
    if len(primary) != 240 or len(provenance["optional_reservation"]) != 20:
        raise ValueError("Unexpected annotation reservation size")
    excluded_ids = set(members["annotation_ids"]) | {
        row["review_id"] for row in provenance["optional_reservation"]
    }
    excluded_groups = set(primary["duplicate_group_id"]) | {
        row["duplicate_group_id"] for row in provenance["optional_reservation"]
    }
    train = read_allowed("train", ["review_id", "duplicate_group_id"])
    if train["review_id"].to_list() != members["training_ids"]:
        raise ValueError("Frozen training membership changed")
    pool = [
        row
        for row in train.to_dicts()
        if row["review_id"] not in excluded_ids and row["duplicate_group_id"] not in excluded_groups
    ]
    pool.sort(key=lambda row: ranked(row["review_id"], "complaint-model-pool"))
    ids = [row["review_id"] for row in pool[: config["candidate_pool"]]]
    texts = read_allowed("train", ["review_id", "duplicate_group_id", "text", "model_text"], ids)
    if not texts["text"].equals(texts["model_text"]):
        raise ValueError("Only original review body may enter the classifier")
    rows = select_rows(texts.to_dicts(), config, excluded_ids, excluded_groups)
    directory.mkdir(parents=True)
    pl.DataFrame(rows).write_parquet(directory / "development.parquet")
    info = {
        "configuration": config,
        "development_sha256": file_hash(directory / "development.parquet"),
        "membership_sha256": file_hash(base / "membership.json"),
        "reservation_provenance_sha256": file_hash(provenance_path),
        "rules_sha256": file_hash(PROJECT_ROOT / "ml/signaldesk_ml/complaint_rules.py"),
        "taxonomy_sha256": file_hash(PROJECT_ROOT / "docs/COMPLAINT_TAXONOMY.md"),
        "manifest_sha256": file_hash(PROJECT_ROOT / "data/processed/appliances-v1/manifest.json"),
        "data_files": {
            name: file_hash(PROJECT_ROOT / f"data/processed/appliances-v1/{name}.parquet")
            for name in ("train", "validation")
        },
        "data_access": (
            "train metadata/text; validation ID/family metadata only; "
            "internal reservation JSON only"
        ),
        "annotation_csv_opened": False,
        "test_opened": False,
        "investigation_opened": False,
        "excluded_ids": sorted(excluded_ids),
        "excluded_families": sorted(excluded_groups),
        "candidate_pool_rows": len(texts),
        "rows": len(rows),
        "sampling_counts": dict(Counter(row["sampling_origin"] for row in rows)),
        "split_counts": dict(Counter(row["development_split"] for row in rows)),
        "fit_ids": [row["review_id"] for row in rows if row["development_split"] == "fit"],
        "holdout_ids": [row["review_id"] for row in rows if row["development_split"] == "holdout"],
    }
    write_json(directory / "selection.json", info)
    return info


def pseudo_label(
    score: float, rule: bool, suppressed: bool, config: dict[str, Any]
) -> dict[str, Any]:
    """Three states; missing rule evidence alone NEVER creates negative supervision."""
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid teacher score")
    disagreement = (rule and score <= config["nli_negative"]) or (
        not rule and score >= config["nli_positive"]
    )
    state, origin = "uncertain", "intermediate_or_disagreement"
    if rule and score >= config["weak_rule_positive"]:
        state, origin = "positive", "nli_rule_agreement"
    elif not rule and not suppressed and score >= config["weak_nli_only_positive"]:
        state, origin = "positive", "nli_only_extreme"
    elif not rule and score <= config["weak_negative"]:
        state, origin = "negative", "nli_low_no_affirmative_rule"
    return {
        "state": state,
        "origin": origin,
        "score": score,
        "rule_selected": rule,
        "rule_suppressed": suppressed,
        "disagreement": disagreement,
    }


def weak_rows(
    rows: list[dict[str, Any]], scores: ScoreArray, config: dict[str, Any]
) -> list[dict[str, Any]]:
    if scores.shape != (len(rows), len(LABELS)):
        raise ValueError("Teacher shape does not match review/category order")
    result = []
    for row, values in zip(rows, scores, strict=True):
        detection = detect_complaints(row["model_text"])
        for category, score in zip(LABELS, values, strict=True):
            evidence = [asdict(e) for e in detection.evidence if e.category == category]
            suppressed = [asdict(e) for e in detection.suppressed if e.category == category]
            result.append(
                {
                    "review_id": row["review_id"],
                    "duplicate_group_id": row["duplicate_group_id"],
                    "development_split": row["development_split"],
                    "category": category,
                    **pseudo_label(float(score), bool(evidence), bool(suppressed), config),
                    "rule_evidence_json": json.dumps(evidence, ensure_ascii=False, sort_keys=True),
                    "suppressed_evidence_json": json.dumps(
                        suppressed, ensure_ascii=False, sort_keys=True
                    ),
                }
            )
    return result


def fit_category(
    rows: list[dict[str, Any]], weak: list[dict[str, Any]], category: str, config: dict[str, Any]
) -> tuple[Any | None, dict[str, Any]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from threadpoolctl import threadpool_limits

    usable = [
        row
        for row in weak
        if row["category"] == category
        and row["development_split"] == "fit"
        and row["state"] != "uncertain"
    ]
    counts = Counter(row["state"] for row in usable)
    info: dict[str, Any] = {
        "fit_positive": counts["positive"],
        "fit_negative": counts["negative"],
        "fit_ids": [row["review_id"] for row in usable],
    }
    if (
        counts["positive"] < config["minimum_fit_positive"]
        or counts["negative"] < config["minimum_fit_negative"]
    ):
        return None, {
            **info,
            "available": False,
            "reason": "insufficient_provisional_label_support",
        }
    by_id = {row["review_id"]: row["model_text"] for row in rows}
    texts = [by_id[row["review_id"]] for row in usable]
    targets = [int(row["state"] == "positive") for row in usable]
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=config["tfidf_min_df"],
                    sublinear_tf=True,
                    max_features=config["tfidf_max_features"],
                    token_pattern=r"(?u)\b\w+(?:['’]\w+)*\b|[!?]+",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=config["logistic_C"],
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=config["logistic_max_iter"],
                    random_state=config["seed"],
                ),
            ),
        ]
    )
    started = time.perf_counter()
    with threadpool_limits(limits=config["cpu_threads"]):
        pipeline.fit(texts, targets)
    return pipeline, {
        **info,
        "available": True,
        "fit_seconds": time.perf_counter() - started,
        "features": len(pipeline.named_steps["tfidf"].vocabulary_),
    }


def train() -> dict[str, Any]:
    config = configuration()
    directory = PROJECT_ROOT / config["artifact_directory"]
    if (directory / "run.json").exists():
        return verify()
    selection = prepare(directory, config)
    rows = pl.read_parquet(directory / "development.parquet").to_dicts()
    texts = [row["model_text"] for row in rows]
    if not (directory / "teacher.json").exists():
        cache = json.loads((PROJECT_ROOT / ".runtime/complaint-pretrained.json").read_text())
        if (cache["model"], cache["revision"]) != (config["model"], config["revision"]):
            raise ValueError("Pinned checkpoint cache does not match configuration")
        import torch

        nli = ComplaintNLI(Path(cache["path"]), config)
        torch.cuda.reset_peak_memory_stats()
        # Longest 16 bodies give a conservative padding/truncation throughput probe.
        probe = sorted(texts, key=len, reverse=True)[:16]
        nli.score_texts(probe[:2])
        torch.cuda.synchronize()
        start = time.perf_counter()
        nli.score_texts(probe)
        torch.cuda.synchronize()
        probe_seconds = time.perf_counter() - start
        estimate = probe_seconds / len(probe) * len(rows)
        check = {
            "probe_reviews": len(probe),
            "probe_seconds": probe_seconds,
            "estimated_full_seconds_conservative": estimate,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "dtype": "float16",
            "attention": "sdpa",
            "pair_batch_size": config["pair_batch_size"],
        }
        write_json(directory / "probe.json", check)
        print(json.dumps({"throughput_check": check}), flush=True)
        if estimate > config["processing_cap_seconds"]:
            raise RuntimeError(
                "Conservative throughput estimate exceeds the initial 20-minute budget"
            )
        last_update = time.perf_counter()

        def progress(done: int, total: int) -> None:
            nonlocal last_update
            if time.perf_counter() - last_update >= 30 or done == total:
                print(json.dumps({"nli_pairs_done": done, "nli_pairs_total": total}), flush=True)
                last_update = time.perf_counter()

        start = time.perf_counter()
        scores = nli.score_texts(
            texts, max_seconds=config["processing_cap_seconds"], progress=progress
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        nli.model.save_pretrained(directory / "nli_checkpoint", safe_serialization=True)
        nli.tokenizer.save_pretrained(directory / "nli_checkpoint")
        pl.DataFrame(
            {
                "review_id": [row["review_id"] for row in rows],
                **{label: scores[:, j] for j, label in enumerate(LABELS)},
            }
        ).write_parquet(directory / "teacher.parquet")
        write_json(
            directory / "teacher.json",
            {
                "processing_seconds": elapsed,
                "review_hypothesis_pairs": len(rows) * len(LABELS),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "teacher_sha256": file_hash(directory / "teacher.parquet"),
                "configuration": config,
                "development_sha256": selection["development_sha256"],
            },
        )
        del nli
        torch.cuda.empty_cache()
    teacher_info = json.loads((directory / "teacher.json").read_text())
    if (
        teacher_info["configuration"] != config
        or teacher_info["development_sha256"] != selection["development_sha256"]
    ):
        raise ValueError("Teacher cache configuration/sample changed")
    if file_hash(directory / "teacher.parquet") != teacher_info["teacher_sha256"]:
        raise ValueError("Cached teacher changed")
    teacher = pl.read_parquet(directory / "teacher.parquet")
    if teacher["review_id"].to_list() != [row["review_id"] for row in rows]:
        raise ValueError("Teacher ID order changed")
    scores = teacher.select(LABELS).to_numpy()
    weak = weak_rows(rows, scores, config)
    pl.DataFrame(weak).write_parquet(directory / "weak_labels.parquet")
    import joblib

    models, linear = {}, {}
    for category in LABELS:
        path = directory / f"linear/{category}.joblib"
        if path.exists():
            raise ValueError("Partial fitting run exists; refusing to retrain/overwrite")
        model, info = fit_category(rows, weak, category, config)
        if model is not None:
            models[category] = model
            path.parent.mkdir(exist_ok=True)
            joblib.dump(model, path, compress=3)
            info["path"] = path.relative_to(directory).as_posix()
        linear[category] = info
    predictions = ComplaintModel(config, "linear", pipelines=models).predict_texts(texts)
    linear_scores = {
        "review_id": [row["review_id"] for row in rows],
        **{label: [p.predictions[j].score for p in predictions] for j, label in enumerate(LABELS)},
    }
    pl.DataFrame(linear_scores).write_parquet(directory / "linear_scores.parquet")
    counts = {
        category: {
            split: dict(
                Counter(
                    w["state"]
                    for w in weak
                    if w["category"] == category
                    and (split == "all" or w["development_split"] == split)
                )
            )
            for split in ("all", "fit", "holdout")
        }
        for category in LABELS
    }
    run = {
        "configuration": config,
        "taxonomy_version": TAXONOMY_VERSION,
        "rule_version": RULE_VERSION,
        "independent_validation": "not_performed",
        "linear": linear,
        "weak_counts": counts,
        "teacher": teacher_info,
        "source_selection_sha256": file_hash(directory / "selection.json"),
        "python": sys.version,
        "lock_sha256": file_hash(PROJECT_ROOT / "uv.lock"),
        "files": {
            p.relative_to(directory).as_posix(): file_hash(p)
            for p in directory.rglob("*")
            if p.is_file()
        },
    }
    write_json(directory / "run.json", run)
    return verify()


def verify() -> dict[str, Any]:
    config = configuration()
    directory = PROJECT_ROOT / config["artifact_directory"]
    run = json.loads((directory / "run.json").read_text())
    if run["configuration"] != config:
        raise ValueError("Saved configuration changed")
    model = load_complaint_model(directory, mode="linear", trusted=True)
    selection = json.loads((directory / "selection.json").read_text())
    for name, checksum in selection["data_files"].items():
        if file_hash(PROJECT_ROOT / f"data/processed/appliances-v1/{name}.parquet") != checksum:
            raise ValueError("Frozen permitted source changed")
    for path, key in [
        ("ml/signaldesk_ml/complaint_rules.py", "rules_sha256"),
        ("docs/COMPLAINT_TAXONOMY.md", "taxonomy_sha256"),
        ("data/processed/appliances-v1/manifest.json", "manifest_sha256"),
        ("artifacts/sentiment/baselines-v1/membership.json", "membership_sha256"),
        (
            "data/annotations/sentiment-v1/complaints-v1/internal/provenance.json",
            "reservation_provenance_sha256",
        ),
    ]:
        if file_hash(PROJECT_ROOT / path) != selection[key]:
            raise ValueError("Frozen taxonomy or rule reference changed")
    rows = pl.read_parquet(directory / "development.parquet").to_dicts()
    assert all(row["source_partition"] == "train" for row in rows)
    assert not {row["review_id"] for row in rows} & set(selection["excluded_ids"])
    assert not {row["duplicate_group_id"] for row in rows} & set(selection["excluded_families"])
    assert not {row["duplicate_group_id"] for row in rows if row["development_split"] == "fit"} & {
        row["duplicate_group_id"] for row in rows if row["development_split"] == "holdout"
    }
    teacher = pl.read_parquet(directory / "teacher.parquet").select(LABELS).to_numpy()
    weak = weak_rows(rows, teacher, config)
    assert pl.DataFrame(weak).equals(pl.read_parquet(directory / "weak_labels.parquet"))
    predicted = model.predict_texts([row["model_text"] for row in rows])
    saved = pl.read_parquet(directory / "linear_scores.parquet")
    for j, category in enumerate(LABELS):
        if run["linear"][category]["available"]:
            actual = np.asarray([p.predictions[j].score for p in predicted], dtype=np.float64)
            np.testing.assert_allclose(actual, saved[category].to_list(), atol=1e-12)
            fit_ids = set(run["linear"][category]["fit_ids"])
            assert fit_ids <= set(selection["fit_ids"])
            assert not fit_ids & set(selection["holdout_ids"])
    nli = load_complaint_model(directory, mode="nli")
    indices = list(range(8))
    assert nli.nli is not None
    reloaded = nli.nli.score_texts([rows[i]["model_text"] for i in indices])
    np.testing.assert_allclose(reloaded, teacher[indices], atol=0.003, rtol=0.003)
    result = {
        "status": "passed",
        "rows": len(rows),
        "annotation_csv_opened": False,
        "test_opened": False,
        "investigation_opened": False,
        "pseudo_labels_exactly_reproduced": True,
        "linear_reload_scores_reproduced": True,
        "nli_reload_probe_rows": len(indices),
        "nli_reload_tolerance": "atol=rtol=0.003 (fp16)",
        "reserved_id_family_overlap": 0,
        "fit_holdout_family_overlap": 0,
    }
    write_json(directory / "verification.json", result)
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "train", "verify"])
    args = parser.parse_args()
    sys.addaudithook(deny_development_files)
    if args.action == "prepare":
        config = configuration()
        result = prepare(PROJECT_ROOT / config["artifact_directory"], config)
        print(json.dumps({key: result[key] for key in ("rows", "sampling_counts", "split_counts")}))
    elif args.action == "train":
        train()
    else:
        verify()


if __name__ == "__main__":
    main()
