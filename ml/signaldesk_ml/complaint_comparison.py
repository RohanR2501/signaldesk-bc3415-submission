"""Descriptive training-derived comparison; generated labels never become ground truth."""

import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from typing import Any

import numpy as np
import polars as pl

from signaldesk.config import PROJECT_ROOT
from signaldesk_data.source import file_hash, write_json
from signaldesk_ml.complaint_inference import LABELS, Mode, load_complaint_model
from signaldesk_ml.complaint_models import configuration, deny_development_files
from signaldesk_ml.complaint_rules import detect_complaints


def compare() -> dict[str, Any]:
    config = configuration()
    directory = PROJECT_ROOT / config["artifact_directory"]
    if (directory / "comparison.json").exists():
        saved: dict[str, Any] = json.loads((directory / "comparison.json").read_text())
        if saved["run_sha256"] != file_hash(directory / "run.json"):
            raise ValueError("Comparison belongs to a different run")
        return saved
    run = json.loads((directory / "run.json").read_text())
    rows = pl.read_parquet(directory / "development.parquet").to_dicts()
    weak = pl.read_parquet(directory / "weak_labels.parquet").to_dicts()
    teacher = pl.read_parquet(directory / "teacher.parquet").select(LABELS).to_numpy()
    linear = pl.read_parquet(directory / "linear_scores.parquet")
    rules = np.array(
        [
            [label in detect_complaints(row["model_text"]).categories for label in LABELS]
            for row in rows
        ]
    )
    nli_selected = teacher >= config["nli_positive"]
    coverage = {}
    agreement = {}
    for j, category in enumerate(LABELS):
        info: dict[str, Any] = {
            "rules_selected": int(rules[:, j].sum()),
            "nli_selected": int(nli_selected[:, j].sum()),
            "nli_abstain": int(
                (
                    (teacher[:, j] > config["nli_negative"])
                    & (teacher[:, j] < config["nli_positive"])
                ).sum()
            ),
            "rules_nli_selected_disagreements": int((rules[:, j] != nli_selected[:, j]).sum()),
            "rules_only_selected": int((rules[:, j] & ~nli_selected[:, j]).sum()),
            "nli_only_selected": int((~rules[:, j] & nli_selected[:, j]).sum()),
        }
        if run["linear"][category]["available"]:
            scores = linear[category].to_numpy()
            selected = scores >= config["linear_positive"]
            info.update(
                {
                    "linear_selected": int(selected.sum()),
                    "linear_abstain": int(
                        (
                            (scores > config["linear_negative"])
                            & (scores < config["linear_positive"])
                        ).sum()
                    ),
                    "linear_nli_selected_disagreements": int(
                        (selected != nli_selected[:, j]).sum()
                    ),
                    "linear_rules_selected_disagreements": int((selected != rules[:, j]).sum()),
                }
            )
            by_id = {row["review_id"]: i for i, row in enumerate(rows)}
            known = [
                w
                for w in weak
                if w["category"] == category
                and w["development_split"] == "holdout"
                and w["state"] != "uncertain"
            ]
            matched = sum(
                (scores[by_id[w["review_id"]]] >= 0.5) == (w["state"] == "positive") for w in known
            )
            agreement[category] = {
                "pseudo_label_agreement_fraction": matched / len(known) if known else None,
                "matching_generated_labels": int(matched),
                "known_generated_labels": len(known),
                "generated_positive": sum(w["state"] == "positive" for w in known),
                "binary_inspection_cutoff": 0.5,
                "independent_validation": False,
            }
        else:
            info["linear_selected"] = None
            agreement[category] = {"unavailable": True, "independent_validation": False}
        coverage[category] = info
    # Select qualitative inspection slots deterministically; these remain developer observations.
    examples = []
    for j, category in enumerate(LABELS):
        slots = {
            "rule_nli_selected": [
                i for i in range(len(rows)) if rules[i, j] and nli_selected[i, j]
            ],
            "nli_only_selected": [
                i for i in range(len(rows)) if not rules[i, j] and nli_selected[i, j]
            ],
            "rule_only_selected": [
                i for i in range(len(rows)) if rules[i, j] and not nli_selected[i, j]
            ],
            "suppressed_context": [
                i
                for i in range(len(rows))
                if any(
                    e.category == category
                    for e in detect_complaints(rows[i]["model_text"]).suppressed
                )
            ],
        }
        for kind, candidates in slots.items():
            if not candidates:
                continue
            # Shortest complete record facilitates inspectable scope; no snippets used as features.
            i = min(candidates, key=lambda i: (len(rows[i]["model_text"]), rows[i]["review_id"]))
            examples.append(
                {
                    "review_id": rows[i]["review_id"],
                    "category": category,
                    "inspection_slot": kind,
                    "text": rows[i]["model_text"],
                    "development_split": rows[i]["development_split"],
                    "nli_score": float(teacher[i, j]),
                    "linear_score": linear[category][i],
                    "rules": asdict(detect_complaints(rows[i]["model_text"])),
                    "independent_gold_label": None,
                }
            )
    write_json(directory / "qualitative_examples.json", examples)
    texts = [row["model_text"] for row in rows if row["development_split"] == "holdout"][:64]
    latency = {}
    for mode in ("rules", "linear", "nli"):
        source: Mode = mode
        started = time.perf_counter()
        model = load_complaint_model(directory, mode=source, trusted=source == "linear")
        load_seconds = time.perf_counter() - started
        model.predict_texts(texts[:8])
        measured = []
        for text in texts:
            started = time.perf_counter()
            model.predict_texts([text])
            measured.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        model.predict_texts(texts)
        batch_seconds = time.perf_counter() - started
        latency[mode] = {
            "reviews": len(texts),
            "load_seconds_including_hash_verification": load_seconds,
            "single_p50_ms": float(np.percentile(measured, 50)),
            "single_p95_ms": float(np.percentile(measured, 95)),
            "batch64_seconds": batch_seconds,
            "batch_amortized_ms": batch_seconds / len(texts) * 1000,
        }
        del model
    selected_counts = {
        "rules": dict(Counter(rules.sum(axis=1).tolist())),
        "nli": dict(Counter(nli_selected.sum(axis=1).tolist())),
    }
    result = {
        "run_sha256": file_hash(directory / "run.json"),
        "rows": len(rows),
        "denominator": (
            "2000 training-derived enriched development reviews; not population prevalence"
        ),
        "coverage": coverage,
        "holdout_pseudo_label_agreement": agreement,
        "selected_category_counts_per_review": selected_counts,
        "latency": latency,
        "qualitative_examples_sha256": file_hash(directory / "qualitative_examples.json"),
        "independent_validation": "not_performed",
    }
    write_json(directory / "comparison.json", result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    sys.addaudithook(deny_development_files)
    compare()
