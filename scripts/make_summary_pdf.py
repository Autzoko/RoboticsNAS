from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = "/Users/ll5582/Desktop/NYU AIR Lab/RoboticsNAS/docs/RoboticsNAS_summary_2026-10-10.pdf"

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], fontSize=16, leading=20, spaceAfter=4, alignment=TA_LEFT)
SUB = ParagraphStyle("SUB", parent=ss["Normal"], fontSize=9, textColor=colors.HexColor("#555555"), spaceAfter=8)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11.5, leading=14, spaceBefore=8, spaceAfter=4,
                    textColor=colors.HexColor("#1F3A5F"))
B = ParagraphStyle("B", parent=ss["Normal"], fontSize=9, leading=12, spaceAfter=3)
BUL = ParagraphStyle("BUL", parent=B, leftIndent=10, bulletIndent=2)
CELL = ParagraphStyle("CELL", parent=ss["Normal"], fontSize=8, leading=10)
CELLB = ParagraphStyle("CELLB", parent=CELL, fontName="Helvetica-Bold")
BOX = ParagraphStyle("BOX", parent=B, fontSize=9, leading=12, backColor=colors.HexColor("#EEF3F8"),
                     borderColor=colors.HexColor("#9DB3CC"), borderWidth=0.6, borderPadding=6, spaceBefore=4,
                     spaceAfter=8)


def table(rows, widths, bold_rows=()):
    data = [[Paragraph(str(c), CELLB if (r == 0 or r in bold_rows) else CELL) for c in row]
            for r, row in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C4D2")),
          ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE5EF")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    for r in bold_rows:
        st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#FFF6D6")))
    t.setStyle(TableStyle(st))
    return t


def bullets(items):
    return [Paragraph(x, BUL, bulletText="-") for x in items]


s = []
s.append(Paragraph("RoboticsNAS: Closed-Loop Architecture Search for VLA Models", H1))
s.append(Paragraph("Status summary, 2026-10-10 &nbsp;|&nbsp; SmolVLA on LIBERO (4 suites, 40 tasks) &nbsp;|&nbsp; "
                   "code and logs: github.com/Autzoko/RoboticsNAS", SUB))

s.append(Paragraph(
    "<b>Bottom line.</b> Offline validation loss does not predict closed-loop success of VLA architectures, while "
    "cheap closed-loop-aware signals do. Tuning the inference schedule of the unchanged SmolVLA network gives "
    "+14.7 points on the official LIBERO test. Several much smaller networks match or beat the default when trained "
    "standalone, but weight-sharing supernets cannot identify them (4 fixes tried, all fail). "
    "<b>Adopted model:</b> SmolVLA default network with 2 denoising steps and 10 executed actions per call "
    "(official test 0.821 vs 0.674 published). <b>Strongest candidate (not yet officially tested):</b> "
    "v12-e4-stretch-f1-t16 (search-val 0.880, 5.7x faster per call, 37% less deploy memory).", BOX))

s.append(Paragraph("1. Setup and protocol", H2))
s += bullets([
    "Base: <i>lerobot/smolvla_base</i> (no LIBERO pretraining); VLM frozen, action expert trained. "
    "Elastic search space of 2,448 configs: VLM depth, expert depth/width, VLM-to-expert bridge, visual tokens, "
    "denoising steps, executed horizon. Key format v{VLM layers}-e{expert layers}-{bridge}-f{FFN}-t{tokens}-s{steps}-h{horizon}.",
    "Leakage control: train-only normalization stats; search on freshly sampled scenes with dedicated seeds; "
    "official init states touched once, for candidates frozen in git beforehand.",
])

s.append(Paragraph("2. Offline proxies vs closed-loop success (111 archs x 400 episodes)", H2))
s.append(table([
    ["Ranking signal", "Kendall tau vs held-out SR", "Top-10 overlap"],
    ["Split-half ceiling (closed loop, 200 vs 200 eps)", "0.840", "8/10"],
    ["Closed loop, 2 episodes / task", "0.819", "6/10"],
    ["Agreement with anchor policy on visited states (ours, 0 rollouts)", "0.738", "8/10"],
    ["NASWOT (zero-cost)", "0.523", "-"],
    ["SNIP (zero-cost)", "0.322", "-"],
    ["Flow-matching validation loss (standard NAS signal)", "0.165", "0/10"],
    ["Grad-norm (zero-cost)", "-0.51 (inverted)", "-"],
], [100, 45, 30], bold_rows=(3,)))
s.append(Spacer(1, 4))
s.append(Paragraph("Across training of one network, validation loss rises 0.515 to 0.961 while closed-loop success "
                   "rises 0.70 to 0.85 (10k to 30k steps): selecting by loss would pick the worst checkpoint.", B))

s.append(Paragraph("3. Search under a rollout budget (replay, 300 repetitions)", H2))
s.append(table([
    ["Method (regret vs best eligible arch, 400 eps)", "No cap", "ms/step &lt;= median"],
    ["Proxy shortlist + paired successive halving (ours)", "0.058", "0.024"],
    ["NASWOT shortlist + paired successive halving", "0.034", "0.024"],
    ["Successive halving, random pool", "0.089", "0.080"],
    ["Learned predictor (ridge, BO-style)", "0.092", "0.098"],
    ["Validation-loss shortlist", "0.130", "0.112"],
    ["Random search", "0.181", "0.178"],
], [100, 30, 40], bold_rows=(1,)))
s.append(Paragraph("A bound-based racing variant (bootstrapped reference, M1+M3) gave no consistent gain over the "
                   "simple shortlist + racing; it under-explores.", B))

s.append(KeepTogether([Paragraph("4. Official LIBERO test (2,000 episodes each, frozen candidates)", H2),
    table([
        ["Configuration", "Network", "Schedule", "Success", "ms/call", "ms/step"],
        ["SmolVLA as published", "v16-e16-stretch-f1-t64", "10 steps, exec 50", "0.674", "396", "7.9"],
        ["<b>Adopted: default net, tuned schedule</b>", "v16-e16-stretch-f1-t64", "2 steps, exec 10", "<b>0.821</b>", "127", "12.7"],
        ["Same schedule, standalone weights", "v16-e16-stretch-f1-t64", "2 steps, exec 10", "0.810", "127", "12.7"],
        ["Small standalone net", "v12-e4-stretch-f1-t16", "4 steps, exec 5", "0.823", "88", "17.7"],
        ["Standard NAS (val-loss pick)", "v16-e8-stretch-f1-t16", "4 steps, exec 50", "0.701", "130", "2.6"],
        ["Random search pick", "v16-e16-stretch-f1-t64", "10 steps, exec 50", "0.662", "396", "7.9"],
    ], [52, 40, 28, 18, 16, 16], bold_rows=(2,)),
    Paragraph("Latency: A100-40GB, batch 1, eager bf16.", SUB)]))

s.append(Paragraph("5. Standalone ground truth and efficiency (16 nets, 2 steps / exec 10, search-val)", H2))
s.append(table([
    ["Network", "Success", "ms/call", "ms/step", "Active params", "Deploy memory"],
    ["SmolVLA published (v16-e16, 10 steps, exec 50)", "0.713", "396", "7.9", "403M", "980 MB"],
    ["Default net v16-e16, tuned schedule", "0.843", "127", "12.7", "403M", "980 MB"],
    ["<b>v12-e4-stretch-f1-t16</b>", "<b>0.880</b>", "<b>69</b>", "<b>6.9</b>", "<b>290M</b>", "<b>616 MB</b>"],
    ["v8-e16-stretch-f0.75-t16", "0.880", "113", "11.3", "306M", "654 MB"],
    ["v16-e12-top-f0.5-t16", "0.875", "112", "11.2", "352M", "726 MB"],
    ["Other 11 nets", "0.752 - 0.863", "-", "-", "-", "-"],
], [62, 22, 18, 18, 24, 26], bold_rows=(3,)))
s.append(Paragraph("Most networks are within ~0.05 success of each other; the meaningful objective is cost at "
                   "statistically tied success.", B))

s.append(Paragraph("6. Weight-sharing supernet (Gate 1): robust negative result", H2))
s.append(table([
    ["Supernet variant", "Kendall vs standalone (n=16)", "Mean SR gap", "Gap, 4-8 layer experts"],
    ["V0 original (sandwich + distillation)", "0.077", "+0.198", "+0.254"],
    ["V1 per-depth norm gains", "0.051", "+0.194", "+0.245"],
    ["V3 PCGrad (gradient-conflict projection)", "0.068", "+0.190", "+0.248"],
    ["V4 bridge-conditioned K/V adapters", "-0.034", "+0.182", "+0.247"],
], [70, 42, 26, 36]))
s.append(Spacer(1, 3))
s += bullets([
    "Noise ceiling: test-retest Kendall of the standalone ranking itself is ~0.56; supernets recover ~0-25% of it.",
    "Diagnosis: gradient conflict sits in the shared expert trunk (cosine 0.17 between 4- and 16-layer subnets); "
    "the readout is not the cause (readout-only fine-tune: no gain); the gap is explained by bridge mismatch "
    "(R<super>2</super> 0.52) more than by expert depth (0.35). A bridge-indexing bug was found and fixed.",
    "Example: the best standalone net v12-e4-stretch-f1-t16 (0.880) is scored only 0.44-0.54 by the four supernets.",
])

s.append(Paragraph("7. What the results support, and next step", H2))
s += bullets([
    "<b>Supported now (analysis / benchmark paper):</b> offline loss fails for VLA architecture evaluation; "
    "closed-loop-aware proxies and paired racing work; inference schedule is a first-class axis; latency and memory "
    "measured; weight-sharing failure rigorously diagnosed.",
    "<b>Not yet supported:</b> a NAS method that discovers better architectures by itself.",
    "<b>Proposed next step (awaiting decision):</b> two-stage method - supernet + proxy shortlist cost-diverse "
    "candidates, short standalone training (~10k steps), paired closed-loop race for the cheapest net tied with "
    "the best, frozen official test. First check reuses 13 existing 10k-step checkpoints (evaluation only). "
    "Alternatives: run the last supernet variant V2 (~8 GPU-h) or stop and write up.",
])

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm,
                        bottomMargin=14 * mm, title="RoboticsNAS status summary 2026-10-10", author="RoboticsNAS")
doc.build(s)
print(OUT)
