"""Collect public licensed media; source weak labels are NEVER event ground truth.

No model inference, training, threshold selection, API upload or release gate update.
Run from the project root: .venv/bin/python scripts/collect_external_pilot.py
"""
from __future__ import annotations

import array
import csv
import hashlib
import html
import json
import math
import re
import subprocess
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/external-real-pilot-v1"
LABELS = {"alarm_bell_ringing", "blender", "cat", "dishes", "dog",
          "electric_shaver_toothbrush", "frying", "running_water", "speech", "vacuum_cleaner"}
HOSTS = {"freesound.org", "cdn.freesound.org", "commons.wikimedia.org", "upload.wikimedia.org"}


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.attrs = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.attrs.append((tag, dict(attrs)))


def resolve_media(item, page):
    """Use only a URL explicitly published on the public source page."""
    attrs = Page(page).attrs
    expected = item["license_url"].split(":", 1)[1].rstrip("/")
    assert any(a.get("href", "").split(":", 1)[-1].rstrip("/") == expected
               for _, a in attrs), "Expected license link absent; manual investigation required"
    if item["platform"] == "freesound":
        links = [a["data-static-file-url"] for _, a in attrs
                 if a.get("data-static-file-url", "").endswith("-hq.mp3")]
        assert len(set(links)) == 1, "No unambiguous publicly linked HQ preview"
        result = links[0]
        assert f'/{item["id"][3:]}_' in result, "Preview ID mismatch"
        return result, "public_hq_mp3_preview_not_original"
    result = item["media_url"]
    assert any(a.get("href", "").split("?", 1)[0] == result for _, a in attrs), "Original URL absent"
    return result, "original_public_wav"


def download(url, target, max_bytes=50 * 1024 * 1024):
    parsed = urlparse(url)
    assert parsed.scheme == "https" and parsed.hostname in HOSTS
    if target.exists():
        return
    temporary = target.with_suffix(target.suffix + ".part")
    # No login, cookies, API key or authenticated download endpoint used.
    subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location",
                    "--proto", "=https", "--proto-redir", "=https", "--max-time", "45",
                    "--max-filesize", str(max_bytes), "--output", str(temporary), url], check=True)
    assert temporary.stat().st_size <= max_bytes
    temporary.replace(target)


def collect(item):
    assert re.fullmatch(r"[a-z0-9-]+", item["id"])
    assert item["label"] in LABELS
    page_path = OUT / "source-pages" / (item["id"] + ".html")
    download(item["source_url"], page_path, 2 * 1024 * 1024)
    media_url, variant = resolve_media(item, page_path.read_text(encoding="utf-8"))
    source = OUT / "downloads" / (item["id"] + (".mp3" if "preview" in variant else ".wav"))
    download(media_url, source)
    wav = OUT / "audio-16k" / (item["id"] + ".wav")
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_format",
                       "-show_streams", "-of", "json", str(source)]))
    pcm = array.array("f", subprocess.check_output(["ffmpeg", "-nostdin", "-v", "error",
                      "-i", str(source), "-map", "0:a:0", "-f", "f32le", "-acodec", "pcm_f32le", "-"]))
    assert pcm and all(math.isfinite(v) for v in pcm), "Empty/nonfinite decoded audio"
    converted = array.array("f", subprocess.check_output(["ffmpeg", "-nostdin", "-v", "error",
                            "-i", str(source), "-map", "0:a:0", "-ac", "1", "-ar", "16000",
                            "-f", "f32le", "-acodec", "pcm_f32le", "-"]))
    assert converted and all(math.isfinite(v) for v in converted)
    peak = max(abs(v) for v in converted)
    # Some MP3 previews overshoot full scale. Preserve downloads byte-for-byte;
    # attenuate only such derived WAVs with one constant gain, never a limiter.
    gain = 0.98 / peak if peak >= 0.999 else 1.0
    samples = array.array("h", (round(v * gain * 32768) for v in converted))
    with wave.open(str(wav), "wb") as f:
        f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        f.writeframes(samples.tobytes())
    with wave.open(str(wav), "rb") as f:
        assert (f.getnchannels(), f.getsampwidth(), f.getframerate()) == (1, 2, 16000)
        duration = f.getnframes() / 16000
        samples = array.array("h", f.readframes(f.getnframes()))
    # MP3 header duration may include encoder delay/padding. Keep this visible.
    original_duration = float(probe["format"]["duration"])
    assert abs(duration - original_duration) < 0.2
    record = {**item, "download_url": media_url, "download_variant": variant,
              "download_sha256": sha(source), "audio_sha256": sha(wav),
              "source_page_sha256": sha(page_path), "audio_path": wav.relative_to(OUT).as_posix(),
              "download_path": source.relative_to(OUT).as_posix(), "duration_s": duration,
              "sample_rate": 16000, "channels": 1, "source_probe": probe,
              "quality": {"finite": True, "source_decode_peak": max(abs(v) for v in pcm),
                          "pcm_peak": max(abs(v) for v in samples) / 32768,
                          "pcm_full_scale_samples": sum(v in (-32768, 32767) for v in samples),
                          "pre_pcm_resampled_peak": peak,
                          "constant_gain_db": 20 * math.log10(gain),
                          "container_vs_pcm_duration_delta_s": original_duration - duration},
              "split": "external_pilot_quarantine", "training_eligible": False,
              "threshold_tuning_eligible": False, "formal_evaluation_eligible": False,
              "label_source": "publisher_description_weak", "human_listened": False,
              "labels_complete": False, "physical_validation_performed": False}
    draft = {"audio_sha256": record["audio_sha256"], "asset_id": item["id"],
             "source": "publisher_description_weak", "complete": False,
             "events": None, "source_reported_present_labels": [item["label"]],
             "unlisted_labels": "unknown_not_negative", "reviewer": None,
             "human_listened": False, "status": "needs_independent_listening_and_timing",
             "formal_evaluation_eligible": False,
             "note": "网站描述只支持来源弱标签；events=null 表示未标注，不是没有事件。未核实起止和伴随事件。不可改 complete=true 冒充完成。"}
    dump(OUT / "annotation-drafts" / (item["id"] + ".json"), draft)
    print(f'{item["id"]}: {duration:.3f}s; {variant}; weak label only', flush=True)
    return record


def duplicate_audit(records):
    esc = [json.loads(s) for s in (ROOT / "data/manifests/esc50.jsonl").read_text().splitlines()]
    with (ROOT / "data/raw/ESC-50-master/meta/esc50.csv").open() as f:
        esc_ids = {r["src_file"] for r in csv.DictReader(f)}
    known = {r["sha256"]: r["path"] for r in esc}
    base = ROOT / "data/desed-real-strong-only"
    desed = [json.loads(s) for s in (base / "manifest.jsonl").read_text().splitlines()]
    for r in desed:
        known[sha(base / r["path"])] = r["path"]
    checks = []
    for r in records:
        id_collision = r["platform"] == "freesound" and r["id"][3:] in esc_ids
        matches = sorted({known[h] for h in [r["download_sha256"], r["audio_sha256"]] if h in known})
        checks.append({"id": r["id"], "esc50_source_id_overlap": id_collision,
                       "exact_file_sha256_matches": matches})
    return {"esc50_records_checked": len(esc), "desed_records_checked": len(desed),
            "checks": checks, "within_batch_exact_duplicates": len({r["download_sha256"] for r in records}) != len(records),
            "scope": "已登记 ESC-50 source ID 和 ESC-50/DESED 文件 SHA256；转码、裁剪、跨站转载等近重复未排除。",
            "independence_proven": False, "near_duplicate_checked": False,
            "pretraining_overlap": "unknown", "note": "没有命中不等于证明没有数据泄漏；预训练语料不可完整追溯。"}


def main():
    for path in (OUT / "annotation-drafts").glob("*.json"):
        draft = json.loads(path.read_text())
        if (draft.get("source") != "publisher_description_weak" or draft.get("complete")
                or draft.get("human_listened") or draft.get("events") is not None):
            raise RuntimeError(f"Refusing to overwrite a modified annotation: {path}")
    for folder in ["source-pages", "downloads", "audio-16k", "annotation-drafts"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    selection = json.loads((OUT / "selection.json").read_text())
    records, failures = [], []
    def safe(item):
        try:
            return collect(item), None
        except Exception as exc:
            return None, {"id": item["id"], "error": str(exc)}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for record, error in pool.map(safe, selection["items"]):
            if record:
                records.append(record)
            else:
                failures.append(error)
    (OUT / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    audit = duplicate_audit(records)
    dump(OUT / "duplicate-audit.json", audit)
    summary = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "selection_sha256": sha(OUT / "selection.json"),
               "downloaded": len(records), "planned": len(selection["items"]),
               "duration_s": sum(r["duration_s"] for r in records),
               "weak_label_classes": sorted({r["label"] for r in records}),
               "strongly_annotated_files": 0, "model_inference_run": False,
               "formal_accuracy_computed": False, "physical_validation_performed": False,
               "status": "COLLECTED_WEAK_LABELS_ONLY_STRONG_ANNOTATION_BLOCKED",
               "failures": failures}
    dump(OUT / "collection-summary.json", summary)
    credits = ["AudioScope 外部素材署名与变更记录", "", "以下许可仅对应逐条素材；不将素材统一改授项目 MIT 许可。", ""]
    rows = []
    for r in records:
        credits.extend([f'{r["id"]} — {r["title"]} — {r["author"]}', r["source_url"],
                        f'{r["license"]} {r["license_url"]}',
                        f'取得版本：{r["download_variant"]}；变更：FFmpeg 解码、混合为单声道、重采样 16 kHz、PCM16；不裁切、不降噪。防转换削波的固定增益：{r["quality"]["constant_gain_db"]:.6f} dB（0 表示没有调整）。原下载文件未改动。', ""])
        e = html.escape
        rows.append(f'<tr><td>{e(r["id"])}</td><td>{e(r["label_zh"])}<br>来源弱标签，未回听</td>'
                    f'<td><audio controls preload="none" src="{e(r["audio_path"])}"></audio><br>{r["duration_s"]:.3f} 秒</td>'
                    f'<td><a href="{e(r["source_url"])}">{e(r["title"])}</a><br>{e(r["author"])} · {e(r["license"])}</td>'
                    f'<td>{e(r["description_zh"])}<br>{e(r["processing_zh"])}</td></tr>')
    (OUT / "ATTRIBUTION.txt").write_text("\n".join(credits), encoding="utf-8")
    (OUT / "试听索引.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<title>AudioScope 外部素材试收集</title><style>body{font:16px/1.6 system-ui;margin:32px;color:#172332}td,th{padding:12px;border:1px solid #ccd5df}table{border-collapse:collapse}audio{width:240px}.notice{background:#fff3d4;padding:16px}</style>'
        '<h1>外部录音素材 · 首批 10 类试收集</h1><p class="notice">仅完成来源弱标签整理；尚未完成逐事件强标注。未回听；不能计算正式 F1，也不能声称通过现场认证。空白事件标注不是负样本。部分素材经过作者剪辑；Freesound 文件为有损 HQ 预览版。</p>'
        '<p>未运行 AudioScope 预测，不按模型表现挑选。来源叙述不等于独立听辨确认。所有文件隔离保存，不加入训练或调参。</p>'
        '<table><thead><tr><th>编号</th><th>来源所述类别</th><th>音频</th><th>署名与来源</th><th>说明／限制</th></tr></thead><tbody>'
        + ''.join(rows) + '</tbody></table></html>', encoding="utf-8")
    paths = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS" and not p.name.endswith(".part"))
    (OUT / "SHA256SUMS").write_text("".join(f'{sha(p)}  {p.relative_to(OUT).as_posix()}\n' for p in paths))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
