from xml.etree import ElementTree

from audioscope.timeline import timeline


def test_label_color_does_not_depend_on_other_events():
    speech = {"label": "speech", "onset_s": 0, "offset_s": 1}
    dog = {"label": "dog", "onset_s": 0, "offset_s": 1}
    a = ElementTree.fromstring(timeline([speech], 5))
    b = ElementTree.fromstring(timeline([speech, dog], 5))
    speech_rect = next(rect for rect in b.findall("rect") if "speech" in rect.find("title").text)
    assert speech_rect.attrib["fill"] == a.find("rect").attrib["fill"]
    assert b.findall("text")[5].attrib["text-anchor"] == "end"


def test_comparison_rows_and_escaping():
    labels = ["dog", "speech"]
    e = {"label": "speech", "onset_s": 0, "offset_s": 1}
    rendered = ElementTree.fromstring(timeline([e], 5, labels))
    assert [node.text for node in rendered.findall("text")][6:] == labels
    escaped = timeline([{**e, "label": "<script>"}], 5)
    assert "<script>" not in escaped and "&lt;script&gt;" in escaped


def test_display_labels_preserve_internal_classes_and_escape_text():
    event = {"label": "dog", "onset_s": 0, "offset_s": 1}
    output = timeline([event], 2, display_labels={"dog": "狗声 <test>"})
    assert "狗声 &lt;test&gt;" in output
    assert event["label"] == "dog"
