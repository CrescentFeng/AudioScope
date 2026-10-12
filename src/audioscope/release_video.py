"""Local Pillow / FFmpeg evidence video with macOS Chinese speech."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .release_audit import _sha256
from .software_validation import OUT, write_json

FONT = "/System/Library/Fonts/STHeiti Medium.ttc"
DISCLAIMER = "软件验证完成；未进行自有物理设备/房间认证。"


def run(command):
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def probe(path):
    return json.loads(
        run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)])
    )


def subtitle_time(seconds):
    ms = round(seconds * 1000)
    return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"


def wrap(text, width=40):
    return "\n".join(text[i : i + width] for i in range(0, len(text), width))


def build(root, smoke=False, font_path=None):
    output = root / ("release/video-smoke" if smoke else "release/video")
    output.mkdir(parents=True, exist_ok=True)
    work = output / "build"
    work.mkdir(exist_ok=True)
    font_path = font_path or FONT
    large, medium, small = [ImageFont.truetype(font_path, n) for n in (62, 38, 30)]
    files = [
        OUT / "robustness.json",
        OUT / "stream_soak.json",
        OUT / "catalog_snapshot.json",
        Path("reports/generated/l4-engineering/long_audio_prediction.json"),
    ]
    if smoke:
        robust = {"conditions": [{"condition": "synthetic fixture", "event_f1": 0.25}]}
        soak = {
            "processed_seconds": 5,
            "rtf": 0.01,
            "compute_p95_ms": 5,
            "rss_growth_bytes_after_warmup": 0,
        }
        counts = {"audio_files": 0, "annotations": 0, "model_artifacts": 0}
        timeline = {"events": []}
    else:
        robust, soak, database, timeline = [json.loads((root / p).read_text()) for p in files]
        if robust["status"] != "passed" or soak["status"] != "passed":
            raise ValueError("Validation must pass before video build")
        counts = database["verification"]["counts"]
    sections = [
        (
            "01 / 从分类到事件检测",
            "AudioScope v1.0",
            [
                "MFCC / SVM → LogMel CNN → AST / CLAP",
                "CRNN / Conformer → ONNX CPU deployment",
                "Software-Validated Release",
            ],
            "AudioScope 是一个环境声音理解项目。从传统分类、深度学习和迁移学习开始，扩展到真实强标注声音事件检测。本版本把重点放在可重复的软件实验、长音频推理和工程交付。所有展示数字均来自实际生成的评测报告。",
            "AudioScope progresses from classification and transfer learning to strongly labelled sound event detection. This software release presents reproducible experiments and deployment evidence.",
        ),
        (
            "02 / 数据血缘与部署",
            "可追溯的数据目录",
            [
                f"音频记录  {counts['audio_files']:,}",
                f"时间 / 类别标注  {counts['annotations']:,}",
                f"模型产物  {counts['model_artifacts']}",
                "PostgreSQL · Alembic · API · 文件 URI + SHA-256",
            ],
            "数据库关联音频元数据、标注、切分、模型哈希和实验指标。音频本体保留在文件系统，通过路径与校验值追踪。已有独立数据库容器与目录接口。正式 Python 镜像构建曾受网络限制，兼容容器已验证；这一限制仍在交付文档中保留。",
            "PostgreSQL tracks audio metadata, annotations, splits, model hashes and metrics. Files stay outside SQL. The compatibility container was verified; the primary Python image build has a documented registry-network limitation.",
        ),
        (
            "03 / 长音频推理",
            "帧概率 → 事件时间轴",
            [
                "10 秒窗口 / 5 秒重叠",
                "先融合帧级概率，再解码事件",
                "图中条带来自已保存的真实推理输出",
            ],
            "长音频通过十秒窗口与五秒重叠处理。系统先融合重叠部分的帧概率，再转换为事件时间轴。图中的事件来自实际推理输出。不同类别能够同时发生；置信度拒识作为开放集基线保留，不能把它解释为可靠的未知声音识别。",
            "Long audio uses ten-second windows and five-second overlap. Frame probabilities are merged before event decoding. The timeline uses recorded inference output. Open-set rejection remains a confidence baseline.",
        ),
        (
            "04 / 可控软件域偏移",
            "七种条件 · 全量测试",
            [
                "221 条 test 录音 × 7 条件 × 2 次推理",
                "固定随机种子 2027 / 冻结验证集阈值",
                "软件模拟，不构成设备或房间认证",
            ],
            "我们在全部二百二十一条测试录音上，加入模拟混响、电话带宽、设备频响和粉红噪声，共七种条件。每种条件重复推理，检查结果一致。图表直接显示事件检测分数和性能下降。这些实验衡量软件扰动下的表现，不代表真实设备或房间验收。",
            "Seven deterministic conditions cover simulated reverberation, bandwidth, device response and pink noise on all 221 test clips. Repeated inference must match. Scores and degradation are reported without a quality pass threshold.",
        ),
        (
            "05 / 有界流式压力测试",
            "1,800 秒音频 · 连续处理",
            [
                f"计算 P95  {soak['compute_p95_ms']:.2f} ms",
                f"端到端 RTF  {soak['rtf']:.5f}",
                f"预热后 RSS 增长  {soak['rss_growth_bytes_after_warmup'] / 1024**2:.2f} MiB",
                "10 秒音频历史 / 固定概率窗口 / 每类事件状态",
            ],
            "流式处理采用增量事件状态机，限制音频历史、平滑窗口和统计缓存大小。压力测试将现有音频循环输入同一个处理器，累计处理一千八百秒。它以加速方式运行，记录计算时间、内存增长和事件一致性；这不是三十分钟实体麦克风录音。",
            "The streaming state machine uses bounded audio, smoothing and timing buffers. One processor handles 1,800 seconds of audio in an accelerated file test, checking latency, memory and event consistency. No physical microphone is involved.",
        ),
        (
            "06 / 发布边界",
            "开发完成，通过软件验证",
            [
                DISCLAIMER,
                "现场采集工具保留为未来扩展",
                "模型局限：噪声退化 / 开放集误报 / 类别覆盖有限",
            ],
            "AudioScope 软件验证版包含可复现评测、数据库、部署接口、测试、报告和自动演示视频。软件验证通过后可以封板。真实设备和房间采集保留为未来扩展。最终结论是：软件验证完成；未进行自有物理设备、房间认证。",
            "AudioScope includes reproducible evaluation, a database catalog, deployment interfaces, tests, reports and this generated video. Software validation is complete; physical device and room certification has not been performed.",
        ),
    ]
    if smoke:
        sections = sections[:1]
    encoders = run(["ffmpeg", "-hide_banner", "-encoders"])
    encoder = "libx264" if "libx264" in encoders else "libopenh264"
    subtitles = {"zh-CN": [], "en": []}
    offset = 0.0
    parts = []
    for index, (eyebrow, title, lines, chinese, english) in enumerate(sections):
        audio_path = work / f"voice-{index}.aiff"
        subtitle_chunks = []
        for phrase in re.split(r"(?<=[。；，])", chinese):
            subtitle_chunks.extend(phrase[i : i + 32] for i in range(0, len(phrase), 32))
        timings = []
        if smoke:
            run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:duration=2",
                    str(audio_path),
                ]
            )
            subtitle_chunks = [chinese[:32]]
            timings = [(0, 2)]
        else:
            voices = []
            cursor = 0.0
            for ci, caption in enumerate(subtitle_chunks):
                text_path = work / f"voice-{index}-{ci}.txt"
                text_path.write_text(caption, encoding="utf-8")
                clip = work / f"voice-{index}-{ci}.aiff"
                run(["say", "-v", "Tingting", "-r", "220", "-f", str(text_path), "-o", str(clip)])
                length = float(probe(clip)["format"].get("duration", 0))
                if length <= 0:
                    raise RuntimeError("Empty speech output: allow macOS speech service access")
                timings.append((cursor, cursor + length))
                cursor += length
                voices.append(clip)
            voice_list = work / f"voices-{index}.txt"
            voice_list.write_text("".join(f"file '{p.name}'\n" for p in voices))
            run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(voice_list),
                    "-c",
                    "copy",
                    str(audio_path),
                ]
            )
        duration = max(2 if smoke else 24, float(probe(audio_path)["format"]["duration"]) + 1)
        duration = math.ceil(duration * 8) / 8
        for ci, caption in enumerate(subtitle_chunks):
            start, end = offset + timings[ci][0], offset + timings[ci][1]
            subtitles["zh-CN"].append((start, end, caption))
        subtitles["en"].append((offset, offset + duration, textwrap.fill(english, 80)))
        base = Image.new("RGB", (1920, 1080), "#101a2c")
        draw = ImageDraw.Draw(base)
        draw.text((100, 65), eyebrow, font=small, fill="#61d5cb")
        draw.text((100, 125), title, font=large, fill="white")
        for li, line in enumerate(lines):
            draw.text((100, 235 + li * 65), line, font=medium, fill="#dbe4f4")
        draw.text(
            (100, 975),
            "AudioScope  /  Software-Validated Release  /  Synthetic narration",
            font=small,
            fill="#8c9db6",
        )
        part = work / f"part-{index}.mp4"
        command = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "1920x1080",
            "-r",
            "8",
            "-i",
            "pipe:0",
            "-i",
            str(audio_path),
            "-af",
            "apad",
            "-t",
            str(duration),
            "-c:v",
            encoder,
            "-b:v",
            "2500k",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "24",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(part),
        ]
        with (work / f"encode-{index}.log").open("wb") as log:
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
            try:
                for frame in range(round(duration * 8)):
                    canvas = base.copy()
                    paint = ImageDraw.Draw(canvas)
                    progress = (frame + 1) / (duration * 8)
                    if index == 3:
                        for ri, row in enumerate(robust["conditions"]):
                            y = 460 + ri * 49
                            paint.text((100, y), row["condition"], font=small, fill="white")
                            paint.rectangle(
                                (
                                    500,
                                    y + 5,
                                    500 + max(1, 2200 * row["event_f1"] * min(1, progress * 3)),
                                    y + 32,
                                ),
                                fill="#61d5cb",
                            )
                            paint.text(
                                (1200, y), f"{row['event_f1']:.3f}", font=small, fill="white"
                            )
                    if index == 2:
                        labels = sorted({e["label"] for e in timeline["events"]})
                        for li, label in enumerate(labels[:10]):
                            y = 460 + li * 35
                            paint.text((100, y), label, font=small, fill="#c5d3e9")
                            for event in timeline["events"]:
                                if event["label"] == label:
                                    x = 600 + event["onset_s"] * 9
                                    paint.rectangle(
                                        (x, y + 4, 600 + event["offset_s"] * 9, y + 24),
                                        fill="#61d5cb",
                                    )
                        paint.line(
                            (600 + progress * 1080, 455, 600 + progress * 1080, 810),
                            fill="white",
                            width=3,
                        )
                    caption_index = next(
                        (i for i, (_, end) in enumerate(timings) if frame / 8 < end),
                        len(timings) - 1,
                    )
                    caption = subtitle_chunks[caption_index]
                    paint.rounded_rectangle((75, 855, 1845, 950), radius=18, fill="#1e304c")
                    paint.text((110, 877), caption, font=medium, fill="white")
                    paint.rectangle((100, 1040, 100 + 1720 * progress, 1046), fill="#61d5cb")
                    if frame == round(duration * 4):
                        canvas.save(work / f"preview-{index}.png")
                    proc.stdin.write(canvas.tobytes())
                proc.stdin.close()
                if proc.wait() != 0:
                    raise RuntimeError(f"FFmpeg failed; see encode-{index}.log")
            except BaseException:
                proc.kill()
                proc.wait()
                raise
        parts.append(part)
        offset += duration
        print(f"Rendered section {index + 1}, duration {duration:.1f}s", flush=True)
    concat = work / "concat.txt"
    concat.write_text("".join(f"file '{p.name}'\n" for p in parts), encoding="utf-8")
    video = output / "AudioScope_v1.0_zh-CN.mp4"
    run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(video),
        ]
    )
    for language, entries in subtitles.items():
        (output / f"{language}.srt").write_text(
            "\n\n".join(
                f"{i + 1}\n{subtitle_time(a)} --> {subtitle_time(b)}\n{text}"
                for i, (a, b, text) in enumerate(entries)
            )
            + "\n",
            encoding="utf-8",
        )
    metadata = probe(video)
    visual = next(s for s in metadata["streams"] if s["codec_type"] == "video")
    audio = next(s for s in metadata["streams"] if s["codec_type"] == "audio")
    actual_duration = float(metadata["format"]["duration"])
    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vn", "-f", "s16le", "-ac", "1", "pipe:1"],
        check=True,
        capture_output=True,
    ).stdout
    pcm_samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    rms = float(np.sqrt(np.mean(pcm_samples**2))) if len(pcm_samples) else 0.0
    run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"])
    checks = {
        "resolution": (visual["width"], visual["height"]) == (1920, 1080),
        "video_codec": visual["codec_name"] == "h264",
        "pixel_format": visual["pix_fmt"] == "yuv420p",
        "audio_codec": audio["codec_name"] == "aac",
        "duration": (1 <= actual_duration <= 10) if smoke else (120 <= actual_duration <= 210),
        "full_decode": True,
        "audible_audio": rms > 50,
        "subtitle_files": all((output / f"{lang}.srt").stat().st_size > 0 for lang in subtitles),
    }
    result = {
        "status": "passed" if all(checks.values()) else "failed",
        "smoke": smoke,
        "physical_validation_performed": False,
        "voice": "synthetic-tone" if smoke else "Tingting",
        "duration_s": actual_duration,
        "checks": checks,
        "video_uri": str(video.relative_to(root)),
        "video_sha256": _sha256(video),
        "disclaimer": DISCLAIMER,
        "source_sha256": {str(p): _sha256(root / p) for p in files} if not smoke else {},
        "asset_sha256": {str(p.relative_to(root)): _sha256(p) for p in output.glob("*.srt")},
    }
    write_json(output / "build_manifest.json", result)
    if not smoke:
        write_json(root / OUT / "video_validation.json", result)
    if result["status"] != "passed":
        raise ValueError(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--font")
    args = parser.parse_args()
    print(
        json.dumps(build(args.root.resolve(), args.smoke, args.font), ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
