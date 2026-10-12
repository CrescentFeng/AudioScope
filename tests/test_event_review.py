import pytest

from audioscope.event_review import compare_jobs, compare_reference, error_triage


def event(label="speech", start=0, end=1):
    return {"label": label, "onset_s": start, "offset_s": end}


def job(events):
    return {
        "id": "test",
        "status": "completed",
        "payload": {
            "duration_s": 5,
            "audio_sha256": "same",
            "events": events,
            "threshold_profile": "overall",
            "model": {"hash": "test"},
        },
    }


def test_matching_is_one_to_one_and_reports_duplicates():
    result = error_triage([event()], [event(), event(start=0.1)])
    assert result["tp"] == 1 and result["fp"] == 1 and result["fn"] == 0
    assert result["extra_prediction"][0]["category"] == "duplicate_or_boundary"


def test_triage_missing_fragmented_boundary_and_wrong_class():
    refs = [event("speech", 0, 1), event("dog", 2, 3), event("cat", 4, 5)]
    predictions = [event("speech", 0, 0.3), event("speech", 0.5, 0.7), event("cat", 2, 3)]
    result = error_triage(refs, predictions)
    assert [e["category"] for e in result["missed_reference"]] == [
        "fragmented_or_boundary",
        "possible_class_confusion",
        "missed",
    ]
    assert result["fp"] == 3 and result["fn"] == 3


def test_reference_requires_hash_complete_and_truth_source():
    current = job([event(), event("unknown", 2, 3)])
    document = {
        "audio_sha256": "same",
        "source": "human_reference",
        "complete": True,
        "events": [event()],
    }
    result = compare_reference(current, document)
    assert result["metrics"]["event_based"]["micro"]["f1"] == 1
    for key, value in [("audio_sha256", "wrong"), ("source", "prediction"), ("complete", False)]:
        with pytest.raises(ValueError):
            compare_reference(current, {**document, key: value})


def test_job_comparison_does_not_claim_accuracy():
    left, right = job([event()]), job([event(start=0.1)])
    result = compare_jobs(left, right)
    assert result["exact_same_events"] == 0 and result["left_only_events"] == 1
    right["payload"]["audio_sha256"] = "different"
    with pytest.raises(ValueError):
        compare_jobs(left, right)
