"""Escaped, labelled event timeline shared by local workbench views."""

import html

COLORS = {
    "alarm_bell_ringing": "#2563eb",
    "blender": "#16a34a",
    "cat": "#9333ea",
    "dishes": "#ea580c",
    "dog": "#b45309",
    "electric_shaver_toothbrush": "#be185d",
    "frying": "#4d7c0f",
    "running_water": "#0891b2",
    "speech": "#4338ca",
    "vacuum_cleaner": "#7e22ce",
    "unknown": "#64748b",
}


def timeline(events: list[dict], duration_s: float, label_order=None, display_labels=None) -> str:
    if duration_s <= 0 or not events:
        return "<p>没有可显示的事件。</p>"
    labels = label_order or sorted({str(e["label"]) for e in events})
    width, left, plot = 1000, 190, 790
    height = 36 + len(labels) * 34
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="声音事件时间轴" style="width:100%;background:#f8fafc">'
    ]
    for index in range(6):
        x = left + plot * index / 5
        anchor = "end" if index == 5 else "start"
        parts.append(
            f'<text x="{x}" y="16" text-anchor="{anchor}" font-size="12" fill="#334155">'
            f"{duration_s * index / 5:.1f}s</text>"
        )
    for index, label in enumerate(labels):
        display_label = (display_labels or {}).get(label, label)
        y = 32 + index * 34
        parts.append(
            f'<text x="6" y="{y + 14}" font-size="12" fill="#172554">{html.escape(display_label)}</text>'
        )
        for event in (e for e in events if e["label"] == label):
            x = left + plot * event["onset_s"] / duration_s
            length = max(2, plot * (event["offset_s"] - event["onset_s"]) / duration_s)
            parts.append(
                f'<rect x="{x:.2f}" y="{y}" width="{length:.2f}" height="18" '
                f'fill="{COLORS.get(label, "#64748b")}" rx="3"><title>'
                f"{html.escape(display_label)} {event['onset_s']:.2f}–"
                f"{event['offset_s']:.2f}s</title></rect>"
            )
    return "".join(parts) + "</svg>"
