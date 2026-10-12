"""Progressive disclosure and callback wiring, without a model or network."""
import numpy as np
import pytest
from scipy.io import wavfile

from audioscope import workbench_ui as ui


class FakeDeployment:
    sample_rate = 16000
    provenance = {"checkpoint_sha256": "fixture"}

    def __init__(self, *args):
        pass

    def predict_samples(self, samples, threshold_profile):
        return {"events": [{"label": "dog", "onset_s": 0., "offset_s": .5, "confidence": .7}],
                "unknown_events": [], "backend": "test", "stats": {}}


@pytest.fixture
def page(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIOSCOPE_LOCAL_HISTORY", "1")
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    monkeypatch.setattr(ui, "SEDDeployment", FakeDeployment)
    return ui.build_workbench(tmp_path / "unused.pt", None, tmp_path)


def callback(page, name):
    return next(f for f in page.fns.values() if f.fn.__name__ == name)


def test_first_open_hides_advanced_controls_and_empty_results(page):
    components = page.config["components"]
    for c in components:
        if c["type"] == "accordion":
            assert c["props"]["open"] is False
    results = next(c for c in components if c["props"].get("elem_id") == "scope-results")
    assert results["props"]["visible"] is False
    audio = next(c for c in components if c["props"].get("elem_id") == "scope-audio")
    assert audio["props"].get("value") is None
    models = [c for c in components if c["props"].get("label") == "分析模型"]
    assert models[0]["props"]["value"] == "current"


def test_analysis_reveals_chinese_results_and_reset_hides_old_recording(page, tmp_path):
    audio = tmp_path / "sample.wav"
    wavfile.write(audio, 16000, np.zeros(16000, dtype=np.int16))
    analyse = callback(page, "analyse")
    result = analyse.fn(str(audio), "overall", False, "current")
    assert len(result) == len(analyse.outputs) == 27
    assert "检测到：狗叫" in result[1]
    assert "狗叫" in result[2]
    assert result[24] == [["狗叫", 0., .5]]
    assert result[25]["visible"] is True
    assert result[26] == str(audio)
    reset = callback(page, "clear_analysis")
    empty = reset.fn()
    assert len(empty) == len(reset.outputs) == 26
    assert empty[0] == {} and empty[3] == [] and empty[4] == []
    assert empty[25]["visible"] is False


def test_chinese_table_selection_resolves_exact_event():
    event = {"label": "dog", "onset_s": 1., "offset_s": 2.}
    assert ui.selected_event([event], ["狗叫", 1., 2.]) == event
    with pytest.raises(ValueError):
        ui.selected_event([event], ["猫叫", 1., 2.])


def test_empty_overview_does_not_claim_silence_or_accuracy():
    text = ui.overview_zh({"payload": {"events": [], "duration_s": 2.}})
    assert "暂未检测到" in text
    assert "静音" not in text and "准确率" not in text
