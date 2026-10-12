from __future__ import annotations

import argparse
import html
from pathlib import Path

import gradio as gr

from audioscope.engineering import SEDDeployment, stream_file


def _timeline(events: list[dict], duration_s: float) -> str:
    if duration_s <= 0:
        return "<p>没有可显示的音频。</p>"
    labels = sorted({str(event["label"]) for event in events})
    if not labels:
        return "<p>当前阈值下没有检测到事件。</p>"
    width = 900
    row_height = 32
    height = 30 + row_height * len(labels)
    colours = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c"]
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        'style="background:#fafafa;border:1px solid #ddd;border-radius:8px">'
    ]
    plot_left = 150
    plot_width = width - plot_left - 20
    for index, label in enumerate(labels):
        y = 20 + index * row_height
        parts.append(f'<text x="8" y="{y + 15}" font-size="12">{html.escape(label)}</text>')
        parts.append(
            f'<line x1="{plot_left}" y1="{y + 10}" x2="{plot_left + plot_width}" '
            'y2="{y + 10}" stroke="#ddd"/>'
        )
        for event in (item for item in events if item["label"] == label):
            x = plot_left + plot_width * float(event["onset_s"]) / duration_s
            event_width = max(
                2.0,
                plot_width * (float(event["offset_s"]) - float(event["onset_s"])) / duration_s,
            )
            parts.append(
                f'<rect x="{x:.2f}" y="{y + 2}" width="{event_width:.2f}" height="16" '
                f'fill="{colours[index % len(colours)]}" rx="3"><title>'
                f"{html.escape(label)} {event['onset_s']:.2f}-{event['offset_s']:.2f}s"
                "</title></rect>"
            )
    parts.append("</svg>")
    return "".join(parts)


def build_app(model_path: Path, onnx_path: Path | None = None) -> gr.Blocks:
    deployment = SEDDeployment(model_path, onnx_path)

    def analyse(audio_path: str, profile: str, mode: str):
        if not audio_path:
            return [], "<p>请先上传或录制音频。</p>", {}
        if mode == "流式模拟（1 秒块）":
            streamed = stream_file(
                Path(audio_path),
                model_path,
                threshold_profile=profile,
                chunk_seconds=1.0,
                onnx_path=onnx_path,
            )
            events = streamed["events"]
            duration_s = float(streamed["processed_seconds"])
            diagnostics = streamed
        else:
            result = deployment.predict_file(Path(audio_path), threshold_profile=profile)
            events = [*result["events"], *result["unknown_events"]]
            duration_s = float(result["stats"]["audio_seconds"])
            diagnostics = {
                "backend": result["backend"],
                "threshold_profile": result["threshold_profile"],
                "stats": result["stats"],
                "known_events": len(result["events"]),
                "unknown_events": len(result["unknown_events"]),
            }
        table = [
            [
                event["label"],
                round(float(event["onset_s"]), 3),
                round(float(event["offset_s"]), 3),
                event.get("confidence", ""),
            ]
            for event in events
        ]
        return table, _timeline(events, duration_s), diagnostics

    with gr.Blocks(title="AudioScope L4 工程验证台") as demo:
        gr.Markdown(
            "# AudioScope L4 工程验证台\n"
            "上传任意时长音频或直接录音；支持长音频滑窗、跨窗融合、流式模拟、"
            "开放集拒识和 ONNX Runtime。"
        )
        with gr.Row():
            audio = gr.Audio(
                sources=["upload", "microphone"], type="filepath", format="wav", label="音频"
            )
            with gr.Column():
                profile = gr.Radio(
                    ["overall", "short_event", "class_balanced"],
                    value="overall",
                    label="阈值配置",
                )
                mode = gr.Radio(
                    ["长音频滑窗", "流式模拟（1 秒块）"],
                    value="长音频滑窗",
                    label="推理模式",
                )
                run = gr.Button("开始分析", variant="primary")
        timeline = gr.HTML(label="事件时间轴")
        events = gr.Dataframe(
            headers=["类别", "开始（秒）", "结束（秒）", "置信度"],
            datatype=["str", "number", "number", "str"],
            interactive=False,
            label="检测结果",
        )
        diagnostics = gr.JSON(label="工程诊断")
        run.click(analyse, inputs=[audio, profile, mode], outputs=[events, timeline, diagnostics])
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()
    build_app(args.model.resolve(), args.onnx.resolve() if args.onnx else None).launch(
        server_name=args.server_name, server_port=args.server_port
    )
