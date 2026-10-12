"""Human-review workbench. Predictions are immutable; review saves are explicit."""

from __future__ import annotations

import os
from functools import wraps
from pathlib import Path

import gradio as gr

from .engineering import SEDDeployment
from .event_review import compare_jobs, demo_reference, load_reference
from .model_registry import model_registry, resolve_model
from .retrieval import ClapSearch, label_description, query_events
from .timeline import timeline as _timeline
from .workbench import LABELS_ZH, AnalysisService, WorkbenchStore, playback, sha256

UI_CSS = """
.gradio-container {max-width: 980px !important; margin: auto !important;}
#scope-header {padding: 20px 0 12px;}
#scope-header h1 {font-size: 32px; letter-spacing: -0.8px; margin-bottom: 8px;}
#scope-header p {font-size: 16px; color: var(--body-text-color-subdued);}
#scope-actions button {min-height: 46px;}
#scope-results {margin-top: 16px; padding: 20px; border: 1px solid var(--border-color-primary); border-radius: 14px;}
#scope-results h2 {margin-top: 0;}
#scope-professional {margin-top: 16px;}
#scope-footer {font-size: 12px; color: var(--body-text-color-subdued); padding: 8px 0;}
@media (max-width: 640px) {
  #scope-header h1 {font-size: 26px;}
  #scope-results {padding: 12px;}
}
"""


def overview_zh(job):
    """Short, approachable overview; never present model scores as accuracy."""
    payload = job["payload"]
    known = [e for e in payload["events"] if e["label"] != "unknown"]
    labels = list(dict.fromkeys(LABELS_ZH.get(e["label"], e["label"]) for e in known))
    headline = "检测到：" + "、".join(labels) if labels else "暂未检测到已知类别的声音"
    unknown = len(payload["events"]) - len(known)
    return (
        f"## {headline}\n\n"
        f"音频 {payload['duration_s']:.1f} 秒 · {len(known)} 段已知声音"
        + (f" · {unknown} 段待确认声音" if unknown else "")
        + "\n\n这是模型预测，可能误报或漏检。点击下方片段可回听确认。"
    )


def _friendly(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (ValueError, OSError) as error:
            raise gr.Error(str(error)) from error

    return wrapped


def selected_event(items, row):
    """Resolve displayed values, not row index: the browser may sort the table."""
    if row is None or len(row) < 3:
        raise ValueError("无法定位所选片段，请重新选择事件行。")
    for event in items:
        if (
            row[0] in (event["label"], LABELS_ZH.get(event["label"]))
            and float(event["onset_s"]) == float(row[1])
            and float(event["offset_s"]) == float(row[2])
        ):
            return event
    raise ValueError("所选片段不属于当前结果，请刷新后重新选择。")


def history_database_url(root):
    if os.getenv("AUDIOSCOPE_LOCAL_HISTORY"):
        return None
    database_url = os.getenv("AUDIOSCOPE_WORKBENCH_DB_URL") or os.getenv("AUDIOSCOPE_DATABASE_URL")
    if not database_url and (root / ".env").is_file():
        from .release_catalog import sessions

        database_url = sessions(root).kw["bind"].url
    return database_url


def build_workbench(model_path: Path, onnx_path: Path | None, root: Path):
    deployment = SEDDeployment(model_path, onnx_path)
    database_url = history_database_url(root)
    store = WorkbenchStore(root / "runtime/workbench", database_url)
    services = {"current": AnalysisService(deployment, store)}
    searcher = ClapSearch(root / "data/pretrained/clap-htsat-unfused-8fa0f1c")
    store.cleanup_exports()

    def choices():
        return [
            (f"{j['created_at'][:19]} · {j['status']} · {j['id'][:8]}", j["id"])
            for j in store.history()
        ]

    def show(job, audio_path):
        payload = job["payload"]
        events = payload["events"]
        reviewed = (job["review"] or {}).get("events", events)
        table = [
            [
                e["label"],
                e["onset_s"],
                e["offset_s"],
                e.get("confidence", ""),
                e.get("mean_confidence", ""),
            ]
            for e in events
        ]
        edit_rows = [[e["label"], e["onset_s"], e["offset_s"]] for e in reviewed]
        return (
            {"job_id": job["id"], "revision": job["revision"], "audio_path": audio_path},
            overview_zh(job),
            _timeline(events, payload["duration_s"], display_labels=LABELS_ZH),
            table,
            edit_rows,
            {
                "model": payload["model"],
                "stats": payload["stats"],
                "threshold_profile": payload["threshold_profile"],
                "backend": payload["backend"],
                "job_id": job["id"],
                "review_revision": job["revision"],
                "original_audio_retained": payload["retain_audio"],
            },
            gr.update(choices=choices(), value=job["id"]),
            _timeline(reviewed, payload["duration_s"]) if job["review"] else "尚未保存人工复核。",
            False,
            None,
            None,
            {},
            [],
            None,
            {},
            "",
            gr.update(choices=choices(), value=None),
            {},
            "",
            "点击检测表中的一行回听。",
            "",
            "",
            "",
            _timeline(events, payload["duration_s"]),
            [[LABELS_ZH.get(e["label"], e["label"]), e["onset_s"], e["offset_s"]] for e in events],
            gr.update(visible=True),
            audio_path,
        )

    @_friendly
    def analyse(audio_path, profile, retain_audio, model_key):
        if not audio_path:
            raise ValueError("请先上传音频。")
        if model_key not in services:
            model, onnx = resolve_model(root, model_key)
            services[model_key] = AnalysisService(SEDDeployment(model, onnx), store)
        job = services[model_key].analyse(Path(audio_path), profile, retain_audio)
        return show(job, audio_path)

    @_friendly
    def reopen(job_id):
        if not job_id:
            raise ValueError("请选择历史任务。")
        job = store.get(job_id)
        if job["status"] != "completed":
            raise ValueError("这个任务没有完整结果，请重新分析原音频。")
        path = store.retained_audio(job_id)
        return show(job, str(path) if path else None)

    def require_state(state):
        if not state or not state.get("job_id"):
            raise ValueError("请先分析音频或打开历史任务。")
        return store.get(state["job_id"])

    def source_path(state):
        job = require_state(state)
        path = Path(state["audio_path"]) if state.get("audio_path") else None
        if path is None or not path.is_file():
            raise ValueError("原音频未保留或已过期；请在‘重新关联’处上传同一个文件。")
        if sha256(path) != job["payload"]["audio_sha256"]:
            raise ValueError("音频与这个任务的哈希不一致，请重新关联原始文件。")
        return path

    @_friendly
    def relink(state, audio_path):
        job = require_state(state)
        if not audio_path or sha256(Path(audio_path)) != job["payload"]["audio_sha256"]:
            raise ValueError("请上传与任务音频 SHA-256 完全相同的文件。")
        return {**state, "audio_path": audio_path}, "原音频已关联，仅用于本次会话。"

    @_friendly
    def select_event(state, selection: gr.SelectData):
        job = require_state(state)
        event = selected_event(job["payload"]["events"], selection.row_value)
        return playback(source_path(state), event), (
            f"{LABELS_ZH.get(event['label'], event['label'])} · {event['onset_s']:.2f}–{event['offset_s']:.2f} 秒"
            "（回听包含前后各 0.5 秒上下文）"
        )

    @_friendly
    def save(state, rows, confirmed):
        require_state(state)
        rows = rows.tolist() if hasattr(rows, "tolist") else rows
        events = [
            {"label": r[0], "onset_s": r[1], "offset_s": r[2]}
            for r in rows
            if any(v not in (None, "") for v in r)
        ]
        revision = store.save_review(state["job_id"], events, state["revision"], confirmed)
        job = store.get(state["job_id"])
        return (
            {**state, "revision": revision},
            f"已保存人工复核 v{revision}；原始预测未改变；不会自动用于训练。",
            _timeline(job["review"]["events"], job["payload"]["duration_s"]),
            False,
            overview_zh(job),
        )

    @_friendly
    def export(state):
        require_state(state)
        store.cleanup_exports()
        return store.export(state["job_id"])

    @_friendly
    def search(state, query, kind, overlap):
        job = require_state(state)
        if kind == "已知类别（中文/英文）":
            hits = query_events(job["payload"]["events"], query, overlap)
        else:
            hits = searcher.search(source_path(state), query)
        rows = [
            [h["label"], h["onset_s"], h["offset_s"], h.get("similarity", ""), h["query_source"]]
            for h in hits
        ]
        return (
            {"job_id": job["id"], "items": hits},
            rows,
            f"找到 {len(hits)} 个候选；点击行可回听。检索不是新的人工真值。",
        )

    @_friendly
    def select_hit(state, hits, selection: gr.SelectData):
        if not hits or hits["job_id"] != state.get("job_id"):
            raise ValueError("任务已切换，请重新检索当前任务。")
        return playback(source_path(state), selected_event(hits["items"], selection.row_value))

    @_friendly
    def reference_compare(state, reference_path, builtin):
        job = require_state(state)
        if not builtin and not reference_path:
            raise ValueError("请上传完整参考 JSON，或选择内置示例参考。")
        result = demo_reference(root, job) if builtin else load_reference(Path(reference_path), job)
        predicted = job["payload"]["events"]
        reference = result["reference_events"]
        labels = sorted({event["label"] for event in predicted + reference})
        duration = job["payload"]["duration_s"]
        return (
            result,
            _timeline(reference, duration, labels),
            _timeline(predicted, duration, labels),
        )

    @_friendly
    def compare(state, other_id):
        job = require_state(state)
        if not other_id:
            raise ValueError("请选择一个对照任务。")
        other = store.get(other_id)
        return compare_jobs(job, other), _timeline(
            other["payload"]["events"], other["payload"]["duration_s"]
        )

    with gr.Blocks(
        title="AudioScope · 听见声音里的细节", css=UI_CSS,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate", font=["system-ui", "sans-serif"]),
        delete_cache=(3600, 86400),
    ) as demo:
        state = gr.State({})
        hits = gr.State([])
        gr.Markdown(
            "# AudioScope\n听见声音里的细节。上传一段音频，看看什么声音在何时出现。",
            elem_id="scope-header",
        )
        audio = gr.Audio(
            sources=["upload", "microphone"], type="filepath", format="wav",
            label="1 · 添加音频", elem_id="scope-audio",
        )
        gr.Markdown("支持 WAV · 最长 30 分钟 / 100 MiB。没有文件？先试试示例。")
        with gr.Row(elem_id="scope-actions"):
            run = gr.Button("2 · 开始分析", variant="primary", scale=2)
            demo_button = gr.Button("试用示例音频", scale=1)
        with gr.Accordion("高级设置 · 通常无需修改", open=False, elem_id="scope-settings"):
            registry = model_registry(root)
            model_choice = gr.Dropdown(
                choices=[("推荐：当前默认模型", "current")]
                + [(entry["label"], key) for key, entry in registry["models"].items()],
                value="current", label="分析模型",
            )
            profile = gr.Radio(
                [("通用（推荐）", "overall"), ("短声音优先", "short_event"), ("类别均衡", "class_balanced")],
                value="overall", label="检测偏好（不同取舍，不保证更准确）",
            )
            retain = gr.Checkbox(False, label="在本机保留原音频，方便以后回听")
            gr.Markdown(
                "默认仅保存结果与音频哈希。上传/回听临时缓存最多保留约 24 小时，"
                "每小时清理；勾选后原音频会长期保存。"
            )
        with gr.Column(visible=False, elem_id="scope-results") as results_panel:
            summary = gr.Markdown()
            timeline = gr.HTML()
            simple_events = gr.Dataframe(
                headers=["声音", "开始（秒）", "结束（秒）"], type="array", interactive=False,
                datatype=["str", "number", "number"], label="3 · 点击一行，回听这段声音", wrap=True,
            )
            selection_label = gr.Markdown("点击上方任意事件行，播放器将载入对应片段。")
            clip = gr.Audio(label="片段回听 · 含前后各 0.5 秒", interactive=False)
        with gr.Accordion("历史与导出 · 找回记录、下载结果", open=False):
            history = gr.Dropdown(choices=choices(), label="本机任务记录")
            with gr.Row():
                refresh = gr.Button("刷新任务列表")
                open_button = gr.Button("打开所选任务")
            relink_audio = gr.File(label="未保留音频时，重新关联同一 WAV 文件", file_types=[".wav"])
            relink_button = gr.Button("仅为本次会话关联音频")
            relink_status = gr.Markdown()
            export_button = gr.Button("导出原始预测及人工复核（JSON / CSV）")
            exports = gr.File(label="下载分析结果", file_count="multiple")
        with gr.Accordion("专业工作台 · 复核、检索与评测", open=False, elem_id="scope-professional"):
            gr.Markdown("先在上方分析音频或打开历史任务，再使用这些工具。首页始终展示原始模型预测；人工复核单独保存。")
            with gr.Tab("检测明细"):
                gr.Markdown("置信分数不是准确率。峰值表示事件内最高分，平均分反映整个片段。")
                events = gr.Dataframe(
                    headers=["类别", "开始（秒）", "结束（秒）", "最高分（非正确率）", "平均分"],
                    type="array", interactive=False,
                )
                gr.Markdown("点击明细行也可回听，播放器在上方的分析结果区。")
                with gr.Accordion("工程诊断", open=False):
                    diagnostics = gr.JSON(label="任务与运行信息（耗时不含上传和页面渲染）")
                    gr.Markdown(f"历史存储：{store.engine.dialect.name}；人工复核不自动进入训练集。")
            with gr.Tab("人工复核"):
                gr.Markdown(
                    "修改类别与时间，新增行可补漏检，删除行可去掉误报。"
                    "预测保持不变，每次保存生成独立版本。\n\n支持类别：" + label_description()
                )
                edits = gr.Dataframe(
                    headers=["类别", "开始（秒）", "结束（秒）"], type="array",
                    datatype=["str", "number", "number"], interactive=True,
                )
                confirm = gr.Checkbox(False, label="我已核对上述标签；不授权自动加入训练集")
                save_button = gr.Button("保存人工复核版本")
                review_notice = gr.Markdown()
                review_timeline = gr.HTML()
            with gr.Tab("文本检索"):
                gr.Markdown(
                    "类别查询示例：**说话并伴随流水**。CLAP 英文示例：**metal clanging**。"
                    "CLAP 返回 10 秒候选窗口，相似度不是正确率，不提供精确事件边界；"
                    "首次建立索引较慢，每次最多 5 分钟音频。"
                )
                query = gr.Textbox(label="声音描述")
                kind = gr.Radio(
                    ["已知类别（中文/英文）", "CLAP 开放描述（英文）"],
                    value="已知类别（中文/英文）", label="检索方式",
                )
                overlap = gr.Checkbox(False, label="类别必须同时出现（仅用于类别查询）")
                search_button = gr.Button("检索当前任务音频")
                search_status = gr.Markdown()
                results = gr.Dataframe(
                    headers=["类别/候选", "开始", "结束", "CLAP 余弦相似度", "来源"],
                    type="array", interactive=False,
                )
                search_audio = gr.Audio(label="检索片段回听", interactive=False)
            with gr.Tab("参考与版本对照"):
                gr.Markdown(
                    "参考标注用于测量预测误差，不会用于训练或调阈值。仅凭模型之间的差异"
                    "不能判断谁更准确。内置示例参考来自已有数据集，不是新制作的人工标签。"
                )
                reference_file = gr.File(
                    label="完整参考 JSON（audio_sha256 / source / complete / events）", file_types=[".json"],
                )
                builtin_reference = gr.Checkbox(False, label="使用内置示例的完整数据集参考")
                reference_button = gr.Button("对照参考标签")
                gr.Markdown("### 当前预测时间轴")
                reference_prediction = gr.HTML()
                gr.Markdown("### 参考标注时间轴")
                reference_timeline = gr.HTML()
                with gr.Accordion("查看当前音频指标与误差明细", open=False):
                    reference_result = gr.JSON(label="参考对照（不等于模型整体准确率）")
                compare_target = gr.Dropdown(choices=choices(), label="选择同一音频的另一次分析任务")
                compare_button = gr.Button("比较模型 / 阈值工作点")
                comparison_timeline = gr.HTML()
                with gr.Accordion("查看模型 / 工作点差异摘要", open=False):
                    comparison = gr.JSON(label="差异摘要（不是精度结论）")
        gr.Markdown(
            "在本机处理 · 默认不长期保留原音频 · 模型预测仅供参考，未进行自有设备 / 房间认证。",
            elem_id="scope-footer",
        )
        outputs = [
            state,
            summary,
            timeline,
            events,
            edits,
            diagnostics,
            history,
            review_timeline,
            confirm,
            clip,
            search_audio,
            hits,
            results,
            exports,
            reference_result,
            reference_timeline,
            compare_target,
            comparison,
            comparison_timeline,
            selection_label,
            review_notice,
            search_status,
            relink_status,
            reference_prediction,
            simple_events,
            results_panel,
            audio,
        ]
        def clear_analysis():
            # New input must not leave the previous recording's results/review active.
            return ({}, "", "", [], [], {}, gr.update(value=None), "", False, None,
                    None, {}, [], None, {}, "", gr.update(value=None), {}, "", "", "", "", "", "",
                    [], gr.update(visible=False))

        run.click(analyse, [audio, profile, retain, model_choice], outputs, concurrency_limit=1)
        demo_button.click(
            lambda: (*clear_analysis(), str(root / "data/l4-engineering/l4_long_demo.wav")), outputs=outputs,
        )
        audio.input(clear_analysis, outputs=outputs[:-1])
        open_button.click(reopen, [history], outputs)
        refresh.click(
            lambda: (gr.update(choices=choices()), gr.update(choices=choices())),
            outputs=[history, compare_target],
        )
        events.select(select_event, [state], [clip, selection_label])
        simple_events.select(select_event, [state], [clip, selection_label])
        save_button.click(
            save, [state, edits, confirm], [state, review_notice, review_timeline, confirm, summary]
        )
        export_button.click(export, [state], [exports])
        relink_button.click(relink, [state, relink_audio], [state, relink_status])
        search_button.click(
            search,
            [state, query, kind, overlap],
            [hits, results, search_status],
            concurrency_limit=1,
        )
        results.select(select_hit, [state, hits], [search_audio])
        reference_button.click(
            reference_compare,
            [state, reference_file, builtin_reference],
            [reference_result, reference_timeline, reference_prediction],
        )
        compare_button.click(compare, [state, compare_target], [comparison, comparison_timeline])
    return demo
