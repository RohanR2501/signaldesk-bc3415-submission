"""Deterministic duplicate families, with an explicit bounded near-duplicate definition."""

import hashlib
import re
import unicodedata
from collections import defaultdict
from typing import Any


def normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def signature(shingles: set[str]) -> int:
    hashes = [int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest()) for s in shingles]
    return sum(
        (1 << bit) for bit in range(64) if sum(1 if h & (1 << bit) else -1 for h in hashes) >= 0
    )


def duplicate_groups(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[dict[str, str], dict[str, int]]:
    """Group normalized exact text and verified 3-word-shingle near duplicates.

    Four 16-bit bands guarantee candidate recall for <=3 differing SimHash bits.
    This is not a guarantee of detecting semantic paraphrases or all high-Jaccard pairs.
    """
    if config["near_duplicate_hamming_distance"] != 3:
        raise ValueError("Four-band candidate scheme requires Hamming radius exactly three")
    parents = list(range(len(rows)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = root(left), root(right)
        if left != right:
            parents[max(left, right)] = min(left, right)

    exact: dict[str, int] = {}
    signatures: dict[int, int] = {}
    all_shingles: dict[int, set[str]] = {}
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    comparisons = near_edges = 0
    for index, row in enumerate(rows):
        normalized = normalized_text(row["text"])
        if normalized in exact:
            union(index, exact[normalized])
            continue
        exact[normalized] = index
        tokens = re.findall(r"\w+", normalized)
        if len(tokens) < config["near_duplicate_min_tokens"]:
            continue
        shingles = {" ".join(tokens[offset : offset + 3]) for offset in range(len(tokens) - 2)}
        fingerprint = signature(shingles)
        keys = [(band, (fingerprint >> (16 * band)) & 65535) for band in range(4)]
        candidates = {candidate for key in keys for candidate in buckets[key]}
        for candidate in sorted(candidates):
            comparisons += 1
            if comparisons > config["max_duplicate_comparisons"]:
                raise ValueError(
                    "Near-duplicate candidate bound exceeded; no unchecked output released"
                )
            if (fingerprint ^ signatures[candidate]).bit_count() > 3:
                continue
            other = all_shingles[candidate]
            if len(shingles & other) / len(shingles | other) >= config["near_duplicate_jaccard"]:
                union(index, candidate)
                near_edges += 1
        signatures[index] = fingerprint
        all_shingles[index] = shingles
        for key in keys:
            buckets[key].append(index)

    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        members[root(index)].append(index)
    result = {}
    conflicts = 0
    for indices in members.values():
        group = hashlib.sha256(min(rows[i]["review_id"] for i in indices).encode()).hexdigest()
        conflicts += len({rows[i]["sentiment_proxy"] for i in indices}) > 1
        for index in indices:
            result[rows[index]["review_id"]] = group
    return result, {
        "union_records": len(rows),
        "normalized_exact_redundant_rows": len(rows) - len(exact),
        "near_duplicate_edges": near_edges,
        "candidate_comparisons": comparisons,
        "duplicate_families": sum(len(indices) > 1 for indices in members.values()),
        "proxy_conflicting_families": conflicts,
    }
