"""Generate Chinese hand-off and a checksum inventory from actual validation evidence."""
import json

from validate_external_desed import DATA, OUT, ROOT, SOURCE, digest, write

NAMES = {"alarm_bell_ringing": "警报/铃声", "blender": "搅拌机", "cat": "猫叫",
         "dishes": "餐具", "dog": "狗声", "electric_shaver_toothbrush": "电动剃须刀/牙刷",
         "frying": "煎炸", "running_water": "流水", "speech": "人声", "vacuum_cleaner": "吸尘器"}


def main():
    result = json.loads((OUT / "evaluation.json").read_text())
    audit = json.loads((OUT / "data-audit.json").read_text())
    metrics = result["metrics"]
    event, segment = metrics["event_based"], metrics["segment_based"]
    lines = [
        "AudioScope 外部真实录音验证——官方人工强标注版",
        "",
        "一、完成了什么",
        "使用 DESED public evaluation 的真实 YouTube 录音及其官方人工强时间标注。",
        f"官方包共 {audit['official_recordings']} 条；同源视频排除 {len(audit['source_id_exclusions'])} 条；",
        f"后续完整性/波形/逐录音许可筛查隔离 {len(audit['additional_exclusions'])} 条。",
        "后续隔离还包括逐录音许可/署名缺失项；筛选在模型推理前固定，不按成绩筛选。",
        f"最终评测 {result['clips']} 条、{result['reference_events']} 个参考事件，音频总计 {result['audio_seconds']/60:.2f} 分钟。",
        f"模型：{result['model']}（当前默认版），模型权重、阈值和后处理保持冻结。",
        "官方标注来自数据集，不是模型自行生成的标签，也没有冒充助手亲耳回听。",
        "",
        "二、实际结果（F1 不是‘正确录音占比’，不能直接叫准确率）",
        f"事件级 micro F1：{event['micro']['f1']:.2%}",
        f"事件级 macro F1：{event['macro_f1']:.2%}",
        f"事件级 precision：{event['micro']['precision']:.2%}；recall：{event['micro']['recall']:.2%}",
        f"事件 TP / FP / FN：{event['micro']['tp']} / {event['micro']['fp']} / {event['micro']['fn']}",
        f"1 秒片段级 micro F1：{segment['micro']['f1']:.2%}",
        f"模型处理 RTF：{result['rtf']:.4f}（不含下载、去重和文件解码）",
        "事件判对要求：类别一致，起点误差≤0.2秒，终点误差≤max(0.2秒,参考事件时长×20%)。",
        "micro 汇总所有事件；macro 对十类等权平均，避免人声等多样本类别掩盖弱类别。",
        "片段级只判断每一秒内是否存在某类声音，比事件起止边界匹配更宽松。",
        "",
        "三、每类结果：参考数 | TP | FP | FN | precision | recall | event-F1 | segment-F1",
    ]
    for label, score in event["per_class"].items():
        lines.append(f"{NAMES.get(label,label)}：{score['tp']+score['fn']} | {score['tp']} | {score['fp']} | {score['fn']} | "
                     f"{score['precision']:.2%} | {score['recall']:.2%} | {score['f1']:.2%} | {segment['per_class'][label]['f1']:.2%}")
    lines += ["", "四、错误证据", "逐条参考事件和预测事件见 event-comparisons.json；",
              "按 FP+FN 排序的 30 个失败示例见 failure-examples.json。",
              "边界错误可能同时造成一个 FP 和一个 FN，二者相加不能当成不同声音的个数。",
              "错误类型由时间重叠规则辅助分类，不代表已听审确认的物理原因。",
              "", "五、去重与局限", 
              f"历史范围：项目 data/ 下 {audit['history_paths_checked']} 个已有 WAV 路径，{audit['history_unique_files']} 个不同文件。",
              "先排除同源 YouTube 视频，再查文件哈希和单声道16k PCM哈希；另做3秒窗口的近重复筛查。",
              "近重复用频谱候选检索+波形相关性检查，不能证明排除了所有剪辑、混音、变速和短于3秒的重复。",
              "这是同一 DESED 数据集家族中新增外部测试子集，不等于完全跨数据集泛化证据。",
              "预训练阶段是否见过相关素材仍无法完全追溯。官方人工标注也可能存在误差。",
              "complete=true 仅指该数据集十个目标类的标注，不表示对世界上所有声音类别穷尽标注。",
              "未进行自有物理设备/房间认证；旧 field-final 标准和审核状态不变。",
              "本轮没有训练、调阈值或换默认模型。若以后利用这些结果优化，新模型仍需新的未触碰测试集。",
              "筛掉授权资料不完整的录音改变了样本构成，结果仅适用于这580条合格子集，不冒充官方全量692条榜单成绩。",
              "", "六、来源与授权记录", 
              "数据集作者：Nicolas Turpault, Romain Serizel, Ankit Shah, Justin Salamon。",
              "官方音频记录：https://zenodo.org/records/4560759；DOI：10.5281/zenodo.4560759。",
              "官方记录声明 CC-BY-4.0；保留官方来源与署名，不将第三方音频改为项目 MIT 许可。",
              f"逐录音授权表只有 {audit['license_table_rows']} 条，其中 {audit['license_eligible_official_rows']} 条授权和署名齐全；其余不进入本轮正式评测。",
              "数据集级许可与逐录音授权字段分别保留；不据此宣称法律认证。每条参考JSON保留原上传者及ID。",
              "本次仅下载和本地验证，没有上传或公开重新分发录音。",
              "标注固定版本：0c57c266c19079358d2a18b524f1f74f878fc6fc。",
              "标注：https://github.com/turpaultn/DESED/blob/0c57c266c19079358d2a18b524f1f74f878fc6fc/data/real/metadata/eval/public.tsv",
              "", "七、复现入口（在 AudioScope 项目根目录）", 
              ".venv/bin/python scripts/validate_external_desed.py history",
              "OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python scripts/validate_external_desed.py prepare",
              ".venv/bin/python scripts/validate_external_desed.py evaluate",
              "冻结协议防止无声改变模型/测试数据；协议发生变化会拒绝覆盖，需另立新实验。",
              "原始数据、清单和每条录音的参考JSON：" + str(DATA),
              "报告目录：" + str(OUT)]
    catalog_path = OUT / "catalog-import.json"
    if catalog_path.exists():
        catalog = json.loads(catalog_path.read_text())
        lines += ["", "八、数据库登记", f"登记状态：{catalog['status']}；新增数据版本包含 {catalog['counts']['audio_files']} 条音频元数据、"
                  f"{catalog['counts']['annotations']} 条人工事件标注、{catalog['counts']['benchmark_results']} 条指标记录。",
                  "音频文件保存在磁盘，PostgreSQL 保存路径、哈希、来源、标签和评测结果；不把大型 WAV 塞入数据库。",
                  "旧发布审核记录未修改，现场认证状态仍为未完成。"]
    (OUT / "结果解读.txt").write_text("\n".join(lines) + "\n")
    paths = [p for p in OUT.rglob("*") if p.is_file() and p.name != "checksums.json"]
    paths += [DATA / "manifest.jsonl", DATA / "history-index.json", DATA / "history-fingerprints.npz",
              DATA / "official/license_public_eval.tsv", DATA / "official/metadata/eval/public.tsv",
              SOURCE / "desed-public-eval.official.tsv", SOURCE / "desed-public-eval.zenodo.json"]
    paths += list((DATA / "references").glob("*.json"))
    paths += [ROOT / "scripts" / name for name in ("validate_external_desed.py", "catalog_external_desed.py", "summarize_external_desed.py", "finalize_external_desed.py")]
    write(OUT / "checksums.json", {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)})
    print(str(OUT / "结果解读.txt"))


if __name__ == "__main__":
    main()
