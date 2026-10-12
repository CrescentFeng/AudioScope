"""Create a searchable, illustrated PDF manual with an embedded CJK font."""

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "reports/generated/user-manual"
OUT = ROOT / "output/pdf/AudioScope 使用手册.pdf"
FONT = "/Library/Fonts/Arial Unicode.ttf"
NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#486FE8")
INK = colors.HexColor("#233244")
MUTED = colors.HexColor("#617286")
PALE = colors.HexColor("#F2F6FA")
LINE = colors.HexColor("#D9E1E8")


def styles():
    pdfmetrics.registerFont(TTFont("ManualCJK", FONT))
    base = getSampleStyleSheet()
    base.add(ParagraphStyle(
        name="Kicker", fontName="ManualCJK", fontSize=9, leading=13,
        textColor=BLUE, spaceAfter=4, uppercase=True,
    ))
    base.add(ParagraphStyle(
        name="ManualTitle", fontName="ManualCJK", fontSize=22, leading=29,
        textColor=colors.black, spaceAfter=8, keepWithNext=True,
    ))
    base.add(ParagraphStyle(
        name="ManualH2", fontName="ManualCJK", fontSize=13, leading=18,
        textColor=colors.black, spaceBefore=8, spaceAfter=4, keepWithNext=True,
    ))
    base.add(ParagraphStyle(
        name="ManualBody", fontName="ManualCJK", fontSize=10, leading=15,
        textColor=INK, spaceAfter=5, alignment=TA_LEFT,
    ))
    base.add(ParagraphStyle(
        name="ManualSmall", fontName="ManualCJK", fontSize=9, leading=13,
        textColor=MUTED, spaceAfter=5,
    ))
    base.add(ParagraphStyle(
        name="ManualBullet", fontName="ManualCJK", fontSize=9.7, leading=14,
        textColor=INK, leftIndent=15, firstLineIndent=-11, spaceAfter=4,
    ))
    base.add(ParagraphStyle(
        name="ManualCover", fontName="ManualCJK", fontSize=32, leading=43,
        textColor=colors.black, spaceAfter=12,
    ))
    base.add(ParagraphStyle(
        name="ManualSub", fontName="ManualCJK", fontSize=14, leading=21,
        textColor=MUTED, spaceAfter=15,
    ))
    base.add(ParagraphStyle(
        name="ManualCaption", fontName="ManualCJK", fontSize=8, leading=11,
        alignment=TA_CENTER, textColor=MUTED, spaceBefore=3, spaceAfter=7,
    ))
    base.add(ParagraphStyle(
        name="ManualNumber", fontName="ManualCJK", fontSize=10, leading=15,
        leftIndent=18, firstLineIndent=-18, textColor=INK, spaceAfter=6,
    ))
    return base


class ManualDoc(BaseDocTemplate):
    def __init__(self, path, **kwargs):
        super().__init__(path, pagesize=letter, **kwargs)
        frame = Frame(0.76 * inch, 0.70 * inch, 6.98 * inch, 9.55 * inch, id="normal")
        self.addPageTemplates([PageTemplate(id="manual", frames=[frame], onPage=self.decorate)])

    def decorate(self, canvas, doc):
        canvas.saveState()
        if doc.page > 1:
            canvas.setFillColor(MUTED)
            canvas.setFont("ManualCJK", 8)
            canvas.drawRightString(7.73 * inch, 10.42 * inch, "AudioScope  /  使用手册  /  v1.0 软件验证版")
            canvas.setStrokeColor(LINE)
            canvas.setLineWidth(0.5)
            canvas.line(0.76 * inch, 0.57 * inch, 7.74 * inch, 0.57 * inch)
            canvas.drawString(0.76 * inch, 0.38 * inch, "本机声音分析工作台")
            canvas.drawRightString(7.74 * inch, 0.38 * inch, str(doc.page - 1))
        canvas.restoreState()


def P(value, style):
    return Paragraph(value, style)


def body(value, s):
    return P(escape(value), s["ManualBody"])


def title(kicker, name, intro, s):
    return [P(escape(kicker), s["Kicker"]), P(escape(name), s["ManualTitle"]), P(escape(intro), s["ManualSmall"])]


def h2(value, s):
    return P(escape(value), s["ManualH2"])


def step(i, value, s):
    return P(f"<font color='{BLUE.hexval()}'><b>{i:02d}</b></font>　{escape(value)}", s["ManualNumber"])


def bullet(value, s):
    return P(f"<font color='{BLUE.hexval()}'><b>•</b></font>　{escape(value)}", s["ManualBullet"])


def img(name, width, caption, s):
    path = ART / name
    flow = Image(str(path), width=width * inch, height=width * inch * ImageReaderAspect(path))
    flow.hAlign = "CENTER"
    return [flow, P(escape(caption), s["ManualCaption"])]


def ImageReaderAspect(path):
    from reportlab.lib.utils import ImageReader

    w, h = ImageReader(str(path)).getSize()
    return h / w


def make_pdf():
    if not Path(FONT).is_file():
        raise FileNotFoundError("Expected installed Arial Unicode font for a searchable CJK PDF")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    s = styles()
    doc = ManualDoc(str(OUT), title="AudioScope 声音事件分析使用手册", author="AudioScope 项目")
    story = []

    # Cover page
    story += [Spacer(1, 0.65 * inch), P("AUDIOSCOPE", s["Kicker"]),
              P("声音事件分析<br/>使用手册", s["ManualCover"]),
              P("从启动工作台到查看、回听与管理分析结果", s["ManualSub"])]
    story += img("workflow.png", 6.5, "图 1  一次分析的主要流程", s)
    story += [Spacer(1, 0.12 * inch), body("适用版本：AudioScope v1.0 软件验证版及当前简洁工作台", s),
              body("适用读者：第一次使用 AudioScope 的本机用户。按照本手册，可以完成一次示例或自有 WAV 音频分析，并理解结果、保存和导出方式。", s),
              P("版本边界：软件验证已完成；尚未进行自有物理设备与房间认证。模型结果是待核对的预测，不是人工真值。", s["ManualSmall"]),
              PageBreak()]

    # 1 quick start
    story += title("01  快速开始", "第一次打开 AudioScope", "日常入口位于桌面 AudioScope 文件夹。启动器会先检查页面、数据库和模型，检查通过后再打开浏览器。", s)
    story += img("workflow.png", 6.45, "图 2  一次分析的主要流程", s)
    story += [step(1, "在桌面打开 AudioScope 文件夹，双击“打开AudioScope.command”。", s),
              step(2, "等待启动检查完成。浏览器打开 http://127.0.0.1:7862/ 后，AudioScope 首页即已就绪。", s),
              step(3, "首次使用 Docker Desktop 时，系统可能要求你本人完成首次设置或权限提示。", s),
              h2("看到错误时先做什么", s), body("如果启动器提示端口占用、模型不可用或数据库尚未就绪，按提示处理后再次双击启动器。不要通过反复刷新一个打不开的收藏网址来启动服务。电脑重启、Docker 未运行时，也优先使用桌面启动器。", s),
              body("工作台只在本机开放。当前地址不是可供外网访问的公开网站。", s), PageBreak()]

    # 2 home
    story += title("02  首页", "认识首页上的几个入口", "首页只保留完成一次分析所需的控件。首次使用无需进入专业功能，也通常无需更改高级设置。", s)
    story += img("home-schematic.png", 6.15, "图 3  当前简洁首页示意图；插图用于说明控件位置", s)
    story += [body("1 · 添加音频：点击上传区域选择 WAV，或将文件拖入区域。也可以按页面提示录制麦克风音频。浏览器首次录音时会询问麦克风权限。", s),
              body("2 · 开始分析：音频添加完成后点击。分析期间请等待页面返回结果，不必重复提交。", s),
              body("试用示例音频：快速体验完整流程，无需先准备自己的录音。", s),
              body("高级设置：模型、检测偏好和是否在本机长期保留原音频。默认推荐设置通常适合先体验。", s),
              body("历史与导出：查找过去的任务、重新关联音频或导出结果。专业工作台用于人工复核、检索和参考标注对照。", s), PageBreak()]

    # 3 analyze
    story += title("03  分析音频", "从音频文件得到事件时间轴", "先用示例音频熟悉操作，再分析自己的录音。模型会返回多个声音类别及其出现时间。", s)
    story += [h2("支持的音频", s), body("工作台接受 WAV 音频，最长 30 分钟、最大 100 MiB。上传内容应为有效 PCM 或浮点 WAV；损坏文件和包含无效数值的音频会被拒绝。", s), h2("操作步骤", s),
              step(1, "点击“试用示例音频”，或点击上传区域选择 WAV 文件。", s),
              step(2, "确认页面已显示所选音频后，点击“2 · 开始分析”。", s),
              step(3, "等待摘要、时间轴和事件表出现。较长音频需要更久；请等待当前任务完成。", s),
              step(4, "选择事件表中的一行试听对应片段。播放器会保留事件前后各约 0.5 秒上下文，遇到文件首尾时自动裁切。", s),
              h2("使用麦克风", s), body("选择页面音频控件中的麦克风来源并录制。浏览器或系统询问权限时，允许当前本机页面使用麦克风；如果误选拒绝，到浏览器网站权限中为 127.0.0.1 开启麦克风后重试。", s),
              body("上传和麦克风录音都送入同一分析工作流。麦克风表现受设备与环境影响；本版本没有进行自有麦克风和房间认证。", s), PageBreak()]

    # 4 results
    story += title("04  查看结果", "怎样读懂预测结果", "摘要给出声音事件概览，时间轴显示预测起止位置，事件表提供可点击的逐条回听入口。", s)
    story += img("results-schematic.png", 6.45, "图 4  结果区域示意；事件类别和时间为插图演示数据", s)
    story += [body("同一时段出现多条不同类别事件是允许的，表示模型认为这些声音可能同时发生。事件起止是模型估计边界，应通过回听核对。", s),
              body("点击事件表任意一行，可试听该事件片段。播放器会附带前后各约 0.5 秒上下文。", s), PageBreak()]

    # 5 interpreting fields
    story += title("05  理解分析结果", "置信分数不是准确率", "模型给出的是预测。需要与同一录音上的完整人工参考标注对照，才能评价这段录音上的识别表现。", s)
    story += img("result-terms.png", 6.35, "图 5  常见字段的含义", s)
    story += [body("峰值分数是该事件片段中最高的帧置信分数。平均分数是该片段的平均置信分数。两者都不能读成“有这么大概率正确”。", s),
              body("unknown 表示系统无法稳定归入已知类别的待确认片段；它既不表示静音，也不代表系统已命名出一种新声音。", s),
              body("不同类别在时间上重叠是正常的。模型认为多个声音可能同时发生；应结合原音频核对是否属实。", s), PageBreak()]

    # 6 settings and privacy
    story += title("06  设置与文件留存", "什么时候需要改高级设置", "建议首次分析保留推荐值。只有当你明确想比较检测取舍或需要长期保留录音时，再展开设置。", s)
    story += [h2("分析模型与检测偏好", s), body("高级设置里的推荐模型适合常规使用。检测偏好包含通用、短声音优先和类别均衡等工作点，它们代表不同取舍，不保证某一种对所有录音都更准确。", s),
              h2("音频是否保留", s), body("默认数据库保存分析结果和音频哈希，不永久保存原始上传音频。上传与回听临时缓存最多保留约 24 小时并按计划清理。", s),
              body("勾选“在本机保留原音频，方便以后回听”后，源 WAV 会存入本机项目数据目录，成为长期本地文件；如不再需要，应由用户自行管理和删除该文件。", s),
              h2("重新关联音频", s), body("未长期保留的历史任务仍可查看结果。如果要再次回听或使用 CLAP 检索，需要重新上传与原文件字节完全相同的 WAV。转码、裁切或另存造成哈希不同的文件无法关联旧任务。", s),
              P("请仅上传你有权处理的音频。工作台设计为单用户本机使用，当前没有公网多用户认证；不应直接暴露到公网。", s["ManualSmall"]), PageBreak()]

    # 7 history
    story += title("07  历史与导出", "找回任务并保存分析结果", "历史记录用于浏览本机已有分析。JSON 和 CSV 适合存档或进一步处理，但两种格式保留的信息不同。", s)
    story += [step(1, "展开“历史与导出 · 找回记录、下载结果”。", s),
              step(2, "如列表没有刚完成的任务，点“刷新任务列表”。", s),
              step(3, "在“本机任务记录”选择任务，点“打开所选任务”。", s),
              step(4, "需要文件时点“导出原始预测及人工复核（JSON / CSV）”，然后下载生成的文件。", s),
              h2("导出文件怎么选", s), body("JSON 适合完整留档和程序读取，保留模型、阈值、音频哈希与复核版本等结构化信息。CSV 适合表格浏览和筛选，会标明事件来源。", s),
              h2("预测与人工复核分开保存", s), body("人工复核可改类别与时间、补充漏检或删除误报。先核对复核表并勾选确认，再点保存。每次保存都会追加独立版本，原始模型预测仍保留；复核内容不会自动进入训练数据。", s),
              body("示例或自动化验收产生的记录不等于人工标注。需要评估准确性时，必须使用与同一音频对应、完整且可靠的参考标签。", s), PageBreak()]

    # 8 professional
    story += title("08  专业工作台", "复核、检索与标注对照", "高级功能默认收起。请先分析音频或打开一条历史任务，再展开“专业工作台 · 复核、检索与评测”。", s)
    story += [h2("检测明细与人工复核", s), body("检测明细查看类别、起止时间和峰值/平均分数。人工复核表允许增、删、改事件，保存后形成新的复核版本；它不会覆写原预测。", s),
              h2("文本检索", s), body("类别查询支持项目已知类别的中英文名称，可用“说话并伴随流水”表达同时出现。当前不支持否定句。", s),
              body("CLAP 开放描述使用英文短语，例如 metal clanging 或 a person talking。它返回 10 秒候选窗口，单次最多处理 5 分钟音频；候选相似度不是正确率，也不等于精确事件边界。", s),
              h2("参考标签与版本比较", s), body("要测量某段录音上的误差，需提供与该文件 SHA-256 完全对应的完整参考 JSON，或使用明确适用于示例音频的内置参考。部分标注不能代表完整真值。", s),
              body("模型/阈值版本对比只能显示输出差异。仅凭某个版本分数更高，不能得出它在这段音频上更准确；判断准确度仍要有完整参考标签。", s), PageBreak()]

    # 9 troubleshooting
    story += title("09  常见问题", "遇到问题时的处理方式", "先区分启动问题、音频格式问题和结果解释问题。大多数情况下，不需要重装项目或重建数据库。", s)
    story += [h2("网页显示拒绝连接", s), body("关闭网页后用桌面“打开AudioScope.command”重新检查并启动。收藏的网址本身不会唤起 Docker。若启动器报告 7862 端口占用，按提示释放占用程序后再试。", s),
              h2("上传时报错", s), body("确认文件是有效 WAV，长度不超过 30 分钟、大小不超过 100 MiB。文件损坏或编码不受支持时，可用可信工具另存为 PCM WAV 后重新分析；另存文件会被视作新的音频。", s),
              h2("结果与耳朵听到的不同", s), body("模型可能误报、漏报或给出不够准确的边界。点击事件行回听，必要时在人工复核中记录修正。预测仅供参考，不能将未经对照的结果当成人工标签。", s),
              h2("模型或数据库检查失败", s), body("按启动器提示检查 Docker Desktop。若提示模型或原数据卷缺失，不要初始化一个空数据库替代；先恢复原项目文件或联系项目维护者。", s),
              h2("发布状态说明", s), body("AudioScope v1.0 完成了可复现的软件验证与本机工程部署；软件域偏移实验不等于真实房间/设备试验。本项目尚未完成自有物理设备与房间认证。", s)]

    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    make_pdf()
