from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from audioscope.experiment import load_artifact, predict_file


def build_app(model_path: Path) -> gr.Blocks:
    artifact = load_artifact(model_path)

    def classify(audio_path: str):
        if not audio_path:
            return {}
        return {
            item["label"]: item["probability"] for item in predict_file(Path(audio_path), artifact)
        }

    with gr.Blocks(title="AudioScope") as demo:
        gr.Markdown("# AudioScope\n环境声音分类验证台（当前为工程 smoke 模型，不代表 ESC-50 指标）")
        with gr.Row():
            audio = gr.Audio(type="filepath", label="上传或录制 WAV")
            labels = gr.Label(num_top_classes=3, label="Top-3")
        run = gr.Button("分析")
        run.click(classify, inputs=audio, outputs=labels)
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Reserved for the next model registry step")
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    build_app(args.model).launch()
