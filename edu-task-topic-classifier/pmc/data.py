"""Immutable source ingestion and explicit feature boundary (stdlib only)."""
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

FEATURE_FIELDS = ("task",)
SPLITS = ("train", "validation", "test")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def feature_text(row):
    # No generic concatenation, dataframe drop list, or metadata fallback.
    text = row["task"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty task text")
    return text


def assert_disjoint(rows):
    groups, exact, ids = {}, {}, set()
    for row in rows:
        split = row["split"]
        if split not in SPLITS or not row["semantic_group_id"]:
            raise ValueError("Invalid split/group")
        if row["id"] in ids:
            raise ValueError("Duplicate task ID")
        ids.add(row["id"])
        key = re.sub(r"\s+", " ", unicodedata.normalize("NFC", feature_text(row))).strip()
        # Stronger than source check: identical text even across subjects.
        for mapping, value in ((groups, row["semantic_group_id"]), (exact, key)):
            if value in mapping and mapping[value] != split:
                raise ValueError("Leakage: group or exact task crosses split")
            mapping[value] = split


def prepare(source, output, include_review=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents:
        raise ValueError("Prepared artifacts must be outside immutable source directory")
    if output.exists():
        raise FileExistsError("Use a new output directory; artifacts are immutable")
    files = ["tasks_normalized_v2.csv", "task_topic_links_v2.csv", "manual_review_queue_v2.csv"]
    hashes = {name: sha256(source / name) for name in files}
    with open(source / files[0], encoding="utf-8-sig", newline="") as f:
        raw = list(csv.DictReader(f))
    assert_disjoint(raw)
    with open(source / files[2], encoding="utf-8-sig", newline="") as f:
        review_rows = list(csv.DictReader(f))
    # Queue field names are asserted, never silently ignored.
    id_field = "task_id" if review_rows and "task_id" in review_rows[0] else "id"
    review_ids = {r[id_field] for r in review_rows}
    review_groups = {r["semantic_group_id"] for r in review_rows}
    review_groups.update(r["semantic_group_id"] for r in raw if r["id"] in review_ids)
    nodes = {}
    with open(source / files[1], encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            path = json.loads(r["topic_path_ids"])
            names = json.loads(r["topic_path_names"])
            if not path or len(path) != len(names):
                raise ValueError("Invalid topic path")
            for i, (node, name) in enumerate(zip(path, names)):
                value = {"name": name, "path": path[:i+1], "subject": path[0]}
                if node in nodes and nodes[node] != value:
                    raise ValueError("Conflicting taxonomy paths")
                nodes[node] = value
    labels, subjects, rows = set(), set(), []
    for r in raw:
        target = json.loads(r["target_topic_ids"])
        subject = json.loads(r["subject_ids"])
        if len(subject) != 1 or not target or json.loads(r["unknown_topic_ids"]):
            raise ValueError("Only known topics and single subject supported in v1")
        for t in target:
            if t not in nodes or nodes[t]["subject"] != subject[0]:
                raise ValueError("Unknown/cross-subject topic")
        labels.update(t for t in nodes if len(nodes[t]["path"]) > 1)
        subjects.update(subject)
        if include_review or r["semantic_group_id"] not in review_groups:
            rows.append({"task": feature_text(r), "id": r["id"], "group": r["semantic_group_id"],
                         "split": r["split"], "subject": subject[0], "targets": target})
    schema = {"version": 1, "feature_fields": list(FEATURE_FIELDS),
              "labels": sorted(labels, key=int), "subjects": sorted(subjects, key=int), "nodes": nodes}
    # Inventory is fixed metadata, not target-frequency-based selection from held-out data.
    schema["train_support"] = dict(Counter(t for r in rows if r["split"] == "train" for t in {a for target in r["targets"] for a in nodes[target]["path"][1:]}))
    if hashes != {name: sha256(source / name) for name in files}:
        raise RuntimeError("Source changed during preparation")
    output.mkdir(parents=True)
    write_json(output / "schema.json", schema)
    for split in SPLITS:
        with open(output / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                if r["split"] == split:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {"source": str(source), "source_sha256": hashes,
                "feature_fields": list(FEATURE_FIELDS), "review_excluded": not include_review,
                "excluded_rows": len(raw)-len(rows), "split_counts": dict(Counter(r["split"] for r in rows)),
                "labels": len(labels), "group_overlap": 0,
                "artifact_sha256": {p.name: sha256(p) for p in output.iterdir() if p.is_file()}}
    write_json(output / "manifest.json", manifest)
    return manifest


def verify_artifacts(directory):
    directory = Path(directory)
    m = read_json(directory / "manifest.json")
    for name, digest in m["artifact_sha256"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Artifact changed: {name}")
    if m["feature_fields"] != ["task"]:
        raise ValueError("Illegal features")
    return m


def load_rows(directory, split, limit=None, seed=42):
    with open(Path(directory) / f"{split}.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if limit and len(rows) > limit:
        # A deterministic sample within the original split, never re-split.
        random.Random(seed).shuffle(rows)
        rows = rows[:limit]
    return rows


def validate_config(config):
    if config.get("feature_fields") != ["task"]:
        raise ValueError("Only feature_fields=['task'] is allowed")
    if config["kind"] not in {"baseline", "bert"}:
        raise ValueError("Unknown model kind")
    for key in ('batch_size', 'gradient_accumulation', 'max_length', 'epochs', 'patience',
                'log_interval', 'mps_empty_cache_interval',
                'train_limit', 'validation_limit', 'word_features', 'char_features', 'max_iter'):
        if key in ('train_limit', 'validation_limit') and config.get(key) is None:
            continue
        if key in config and (not isinstance(config[key], int) or config[key] < 1):
            raise ValueError(f'{key} must be a positive integer')
    fraction = config.get('mps_memory_fraction', 1.0)
    if not isinstance(fraction, (int, float)) or not 0 < fraction <= 1:
        raise ValueError('mps_memory_fraction must be in (0, 1]')
    for key in ('fixed_padding', 'optimizer_foreach'):
        if key in config and not isinstance(config[key], bool):
            raise ValueError(f'{key} must be boolean')
    if not isinstance(config.get('seed'), int):
        raise ValueError('seed must be an integer')
    if config.get("mode") == "full" and any(config.get(k) is not None for k in ("train_limit", "validation_limit")):
        raise ValueError("Full mode cannot limit train/validation")
    if config["kind"] == "bert" and "bert" in config:
        validate_config(dict(config["bert"], kind="bert", feature_fields=["task"], seed=config["seed"]))
    return config
