"""Frozen English query evaluation on all existing DESED test audio; no tuning."""

import argparse
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from .audio import read_wav_mono
from .event_review import overlap
from .retrieval import ClapSearch
from .sed_data import read_soundscape_manifest
from .software_validation import write_json
from .workbench import sha256

QUERIES = {
    "alarm_bell_ringing": "the sound of an alarm bell ringing",
    "blender": "the sound of a blender",
    "cat": "the sound of a cat meowing",
    "dishes": "the sound of dishes clattering",
    "dog": "the sound of a dog barking",
    "electric_shaver_toothbrush": "the sound of an electric shaver or toothbrush",
    "frying": "the sound of food frying",
    "running_water": "the sound of running water",
    "speech": "the sound of a person speaking",
    "vacuum_cleaner": "the sound of a vacuum cleaner",
}


def relevant(span, references, label):
    return any(
        r["label"] == label and overlap(span, r) >= min(0.2, (r["offset_s"] - r["onset_s"]) / 2)
        for r in references
    )


def ranking_metrics(hits, spans, references, label):
    positives = sum(relevant({"onset_s": a, "offset_s": b}, references, label) for a, b in spans)
    relevance = [relevant(hit, references, label) for hit in hits]
    return {
        "positive_windows": positives,
        "total_windows": len(spans),
        "precision_at_5": float(np.mean(relevance[:5])) if relevance else 0.0,
        "recall_at_10": sum(relevance[:10]) / positives if positives else None,
        "reciprocal_rank_at_10": next((1 / (i + 1) for i, v in enumerate(relevance[:10]) if v), 0),
        "random_expected_precision": positives / len(spans),
        "absent_class": positives == 0,
    }


def run(root):
    manifest = root / "data/desed-real-strong-only/manifest.jsonl"
    records = sorted(
        [r for r in read_soundscape_manifest(manifest) if r.split == "test"],
        key=lambda r: r.scene_id,
    )
    output = root / "reports/generated/workbench"
    protocol = {
        "manifest_sha256": sha256(manifest),
        "clips": len(records),
        "split": "test",
        "queries": QUERIES,
        "clips_per_sequence": 20,
        "window_seconds": 10,
        "hop_seconds": 5,
        "tuning": False,
        "relevance": "reference overlap >= min(0.2 s, half reference event duration)",
        "caveat": "10-class retrieval benchmark; not unrestricted semantic understanding",
    }
    write_json(output / "retrieval_protocol.json", protocol)
    searcher = ClapSearch(root / "data/pretrained/clap-htsat-unfused-8fa0f1c")
    rows = []
    with tempfile.TemporaryDirectory(prefix="audioscope-retrieval-eval-") as directory:
        path = Path(directory) / "sequence.wav"
        for start in range(0, len(records), 20):
            parts, references, offset = [], [], 0.0
            selected = records[start : start + 20]
            for record in selected:
                audio, _ = read_wav_mono(manifest.parent / record.path, 16000)
                parts.append(audio)
                references.extend(
                    {
                        "label": e.label,
                        "onset_s": e.onset_s + offset,
                        "offset_s": e.offset_s + offset,
                    }
                    for e in record.events
                )
                offset += len(audio) / 16000
            wavfile.write(path, 16000, np.concatenate(parts))
            for label, query in QUERIES.items():
                hits = searcher.search(path, query, 10)
                rows.append(
                    {
                        "sequence": start // 20,
                        "label": label,
                        "metrics": ranking_metrics(hits, searcher.spans, references, label),
                        "top_hits": hits,
                    }
                )
            print(
                f"CLAP retrieval: {min(start + 20, len(records))}/{len(records)} clips", flush=True
            )
    applicable = [r["metrics"] for r in rows if not r["metrics"]["absent_class"]]
    write_json(
        output / "retrieval_evaluation.json",
        {
            "status": "completed",
            "protocol": protocol,
            "model": searcher.provenance,
            "evaluated_queries": len(rows),
            "queries_with_reference_positives": len(applicable),
            "absent_class_queries": len(rows) - len(applicable),
            "aggregate": {
                key: float(np.mean([r[key] for r in applicable]))
                for key in [
                    "precision_at_5",
                    "recall_at_10",
                    "reciprocal_rank_at_10",
                    "random_expected_precision",
                ]
            },
            "rows": rows,
            "abstention_supported": False,
            "notice": "CLAP always ranks candidates, including when the requested sound is absent.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    run(parser.parse_args().root.resolve())
