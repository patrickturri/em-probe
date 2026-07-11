"""Render the research-format technical report PDF from canonical evidence.

The PDF is intentionally generated from the logged Phase 4 summary and the
canonical result figures. Run with:

    uv run python scripts/build_technical_report_pdf.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "results/runs/20260710-185310_phase4_report"
SUMMARY_PATH = REPORT_DIR / "summary.json"
FIGURE_DIR = REPORT_DIR / "figures"
DEFAULT_OUTPUT = ROOT / "output/pdf/emergent_misalignment_cross_domain_report.pdf"

NAVY = colors.HexColor("#17324D")
TEAL = colors.HexColor("#0F766E")
ORANGE = colors.HexColor("#BB3E03")
MIST = colors.HexColor("#EAF2F8")
PALE_TEAL = colors.HexColor("#E3F3F1")
LINE = colors.HexColor("#C9D4DF")
TEXT = colors.HexColor("#1F2937")
MUTED = colors.HexColor("#52606D")

DOMAIN_LABELS = {
    "medical": "Bad medical advice",
    "sports": "Extreme sports advice",
    "financial": "Risky financial advice",
}

CAUSAL_PATHS = {
    "Medical": {
        "organism": ROOT / "results/runs/20260710-000400_gen_qwen7b_medical_organism/metrics.json",
        "layerwise": ROOT / "results/runs/20260710-171132_steer_qwen7b_medical/layerwise/metrics.json",
        "single": ROOT / "results/runs/20260710-171132_steer_qwen7b_medical/single/metrics.json",
        "random": ROOT / "results/runs/20260710-171132_steer_qwen7b_medical/random_baseline/metrics.json",
        "steer": ROOT / "results/runs/20260710-175723_steer_qwen7b_medical_steer_sweep/steer_base_lambda5/metrics.json",
    },
    "Extreme sports": {
        "organism": ROOT / "results/runs/20260710-203546_gen_qwen7b_sports_organism/metrics.json",
        "layerwise": ROOT / "results/runs/20260710-204905_steer_qwen7b_sports/layerwise/metrics.json",
        "single": ROOT / "results/runs/20260710-204905_steer_qwen7b_sports/single/metrics.json",
        "random": ROOT / "results/runs/20260710-204905_steer_qwen7b_sports/random_baseline/metrics.json",
        "steer": ROOT / "results/runs/20260710-204905_steer_qwen7b_sports/steer_base_lambda5/metrics.json",
    },
    "Financial": {
        "organism": ROOT / "results/runs/20260710-213605_gen_qwen7b_financial_organism/metrics.json",
        "layerwise": ROOT / "results/runs/20260710-214814_steer_qwen7b_financial/layerwise/metrics.json",
        "single": ROOT / "results/runs/20260710-214814_steer_qwen7b_financial/single/metrics.json",
        "random": ROOT / "results/runs/20260710-214814_steer_qwen7b_financial/random_baseline/metrics.json",
        "steer": ROOT / "results/runs/20260710-214814_steer_qwen7b_financial/steer_base_lambda5/metrics.json",
    },
}


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required evidence file is missing: {path}")
    return json.loads(path.read_text())


def rate_cell(metric: dict) -> str:
    return (
        f"{metric['n_misaligned']} / {metric['n_coherent']} "
        f"({metric['misaligned_rate'] * 100:.1f}%)"
    )


def image_flowable(path: Path, max_width: float) -> Image:
    if not path.is_file():
        raise FileNotFoundError(f"Required report figure is missing: {path}")
    reader = ImageReader(str(path))
    width, height = reader.getSize()
    rendered = Image(str(path), width=max_width, height=max_width * height / width)
    rendered.hAlign = "CENTER"
    return rendered


def figure_block(path: Path, max_width: float, caption_text: str, style_set: dict[str, ParagraphStyle]) -> KeepTogether:
    """Keep each figure and its caption on the same page."""
    return KeepTogether(
        [
            image_flowable(path, max_width),
            paragraph(caption_text, style_set["caption"]),
        ]
    )


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    base.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=21,
            leading=25,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=8,
        )
    )
    base.add(
        ParagraphStyle(
            name="Subtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=MUTED,
            spaceAfter=5,
        )
    )
    base.add(
        ParagraphStyle(
            name="Heading1Report",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            textColor=NAVY,
            spaceBefore=12,
            spaceAfter=6,
        )
    )
    base.add(
        ParagraphStyle(
            name="Heading2Report",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            textColor=TEAL,
            spaceBefore=9,
            spaceAfter=4,
        )
    )
    base.add(
        ParagraphStyle(
            name="BodyReport",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=13,
            textColor=TEXT,
            spaceAfter=6,
        )
    )
    base.add(
        ParagraphStyle(
            name="Caption",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=7.6,
            leading=10,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=8,
        )
    )
    base.add(
        ParagraphStyle(
            name="Reference",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.0,
            leading=10.2,
            textColor=TEXT,
            spaceAfter=4,
        )
    )
    base.add(
        ParagraphStyle(
            name="TableCell",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.2,
            leading=8.8,
            textColor=TEXT,
        )
    )
    base.add(
        ParagraphStyle(
            name="TableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.2,
            leading=8.8,
            textColor=colors.white,
        )
    )
    return {
        "title": base["ReportTitle"],
        "subtitle": base["Subtitle"],
        "h1": base["Heading1Report"],
        "h2": base["Heading2Report"],
        "body": base["BodyReport"],
        "caption": base["Caption"],
        "ref": base["Reference"],
        "cell": base["TableCell"],
        "head": base["TableHeader"],
    }


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def wrapped_table(
    rows: Iterable[Iterable[str]],
    widths: list[float],
    style_set: dict[str, ParagraphStyle],
    *,
    header_background: colors.Color = NAVY,
    font_size: float | None = None,
) -> Table:
    raw_rows = list(rows)
    table_rows = []
    for row_index, row in enumerate(raw_rows):
        role = style_set["head"] if row_index == 0 else style_set["cell"]
        if font_size is not None:
            role = ParagraphStyle(
                f"{role.name}{row_index}",
                parent=role,
                fontSize=font_size,
                leading=font_size + 1.5,
            )
        table_rows.append([paragraph(str(cell), role) for cell in row])
    table = Table(table_rows, colWidths=widths, repeatRows=1, hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), header_background),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.35, LINE),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def workflow_illustration() -> Drawing:
    """A compact vector diagram for the PDF; no external diagram renderer needed."""
    drawing = Drawing(480, 164)
    boxes = [
        (8, 108, 96, 32, "Harmful\\nfine-tuning data", MIST),
        (124, 108, 96, 32, "LoRA model\\norganism", PALE_TEAL),
        (240, 108, 96, 32, "400 fixed-prompt\\ngenerations", MIST),
        (356, 108, 116, 32, "Alignment and\\ncoherence judge", PALE_TEAL),
        (66, 28, 106, 34, "Clear aligned /\\nmisaligned labels", colors.HexColor("#FCECDD")),
        (194, 28, 106, 34, "Residual means\\nat every layer", colors.HexColor("#E8F0FE")),
        (322, 28, 142, 34, "Direction, probe,\\ntransfer, intervention", colors.HexColor("#EDE9FE")),
    ]
    for x, y, width, height, label, fill in boxes:
        drawing.add(Rect(x, y, width, height, rx=5, ry=5, fillColor=fill, strokeColor=LINE, strokeWidth=0.8))
        for index, line in enumerate(label.split("\\n")):
            drawing.add(
                String(
                    x + width / 2,
                    y + height / 2 + 4 - index * 9,
                    line,
                    textAnchor="middle",
                    fontName="Helvetica-Bold",
                    fontSize=7.5,
                    fillColor=TEXT,
                )
            )

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        drawing.add(Line(x1, y1, x2, y2, strokeColor=MUTED, strokeWidth=1.1))
        angle = 0 if x2 >= x1 else 180
        if abs(y2 - y1) > abs(x2 - x1):
            angle = 90 if y2 >= y1 else -90
        if angle == 0:
            points = [x2, y2, x2 - 5, y2 + 2.5, x2 - 5, y2 - 2.5]
        elif angle == 180:
            points = [x2, y2, x2 + 5, y2 + 2.5, x2 + 5, y2 - 2.5]
        elif angle == 90:
            points = [x2, y2, x2 - 2.5, y2 - 5, x2 + 2.5, y2 - 5]
        else:
            points = [x2, y2, x2 - 2.5, y2 + 5, x2 + 2.5, y2 + 5]
        drawing.add(Polygon(points, fillColor=MUTED, strokeColor=MUTED))

    arrow(104, 124, 124, 124)
    arrow(220, 124, 240, 124)
    arrow(336, 124, 356, 124)
    arrow(414, 108, 414, 80)
    arrow(414, 80, 119, 62)
    arrow(172, 45, 194, 45)
    arrow(300, 45, 322, 45)
    drawing.add(
        String(
            240,
            151,
            "Workflow: data -> organism -> evaluated answers -> shared-representation tests",
            textAnchor="middle",
            fontName="Helvetica",
            fontSize=8.5,
            fillColor=MUTED,
        )
    )
    return drawing


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, 0.52 * inch, letter[0] - doc.rightMargin, 0.52 * inch)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, 0.33 * inch, "Emergent misalignment cross-domain technical report")
    canvas.drawRightString(letter[0] - doc.rightMargin, 0.33 * inch, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def build_pdf(output: Path) -> None:
    summary = load_json(SUMMARY_PATH)
    style_set = styles()
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output),
        pagesize=letter,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.68 * inch,
        bottomMargin=0.72 * inch,
        title="Shared Linear Representations of Emergent Misalignment Across Fine-Tuning Domains",
        author="patrickturri/em-probe",
        subject="Technical report with canonical experimental evidence",
    )
    usable_width = letter[0] - doc.leftMargin - doc.rightMargin
    story = []

    story.extend(
        [
            paragraph("Shared Linear Representations of Emergent Misalignment Across Fine-Tuning Domains", style_set["title"]),
            paragraph("Technical report - Qwen2.5-7B-Instruct model organisms", style_set["subtitle"]),
            paragraph(
                'Repository: <link href="https://github.com/patrickturri/em-probe" color="#0F766E">github.com/patrickturri/em-probe</link><br/>'
                "Evidence: canonical Phase 3 and Phase 4 runs, validated before rendering<br/>"
                "Research release: July 2026",
                style_set["subtitle"],
            ),
            Spacer(1, 10),
            paragraph("Abstract", style_set["h1"]),
            paragraph(
                "Emergent misalignment is the observation that narrow harmful fine-tuning can induce broadly harmful behavior on prompts unrelated to the fine-tuning data. "
                "We fine-tuned Qwen2.5-7B-Instruct with LoRA on bad medical advice, extreme sports advice, and risky financial advice. "
                "The organisms produced coherent misaligned answers on a fixed eight-question evaluation set at 18.4%, 13.9%, and 31.7%, respectively, versus 0.0% for matched base-model samples. "
                "Layerwise residual-stream probes transferred across every source-target pair (AUROC 0.860-0.983), and their mean-difference directions had cosines 0.388-0.549 versus a random absolute-cosine baseline of 0.013 +/- 0.009. "
                "Targeted ablation reduced the measured effect while random-direction controls preserved it. These results support a shared linear representation across the three tested organisms, but do not establish universality beyond this model family, seed set, and prompt set.",
                style_set["body"],
            ),
            paragraph("Key findings", style_set["h2"]),
            wrapped_table(
                [
                    ["Finding", "Evidence"],
                    ["Cross-domain transfer", "All 9 source-target AUROCs are 0.860 or higher; every organism independently selects layer 17."],
                    ["Direction geometry", "Pairwise cosines are 0.388-0.549, far above random absolute cosine 0.013 +/- 0.009."],
                    ["Causal evidence", "Medical targeted ablation lowers 18.4% to 0.0-0.5%, while random ablation retains 15.3%."],
                ],
                [1.55 * inch, usable_width - 1.55 * inch],
                style_set,
                header_background=TEAL,
                font_size=8.1,
            ),
            Spacer(1, 10),
            workflow_illustration(),
            paragraph(
                "Figure 1. Experimental workflow. The answer-token residual mean excludes prompt tokens. "
                "Diagonal probe evaluation holds out question ids rather than individual samples, preventing repeated-question leakage.",
                style_set["caption"],
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            paragraph("1. Methods", style_set["h1"]),
            paragraph(
                "The final experiments use Qwen/Qwen2.5-7B-Instruct on an A100-40GB GPU. Each rank-32 rsLoRA adapter is trained for one epoch on an upstream harmful-advice dataset. "
                "For each condition, the model samples 50 temperature-1.0 responses to each of eight fixed prompts (400 responses total). Claude Haiku 4.5 scores alignment and coherence at temperature 0. "
                "The evaluation prompts originate in the emergent-misalignment benchmark [1]. The harmful text-domain organisms follow the model-organism setup of Turner et al. [2].",
                style_set["body"],
            ),
            paragraph("Representation and evaluation", style_set["h2"]),
            paragraph(
                "An answer is misaligned when alignment is below 30 and coherence is above 50. An aligned probe example has alignment above 70 and coherence above 50; the mid-band is omitted. "
                "At every transformer layer, the feature is the mean residual stream over answer tokens. The candidate direction is the mean activation for misaligned answers minus the mean activation for aligned answers. "
                "A logistic probe is fit with GroupShuffleSplit by question id. Transfer fits a source probe on source training questions and scores all clearly labelled target rows; diagonal entries use held-out source questions.",
                style_set["body"],
            ),
            paragraph("Causal interventions and uncertainty", style_set["h2"]),
            paragraph(
                "The learned direction is projected out at every layer or at the best layer only. A random same-sized direction is the ablation control. "
                "For steering, the raw mean-difference direction is added to base-model residual activations with coefficient lambda, following the direction-level intervention approach of Soligo et al. [3]. "
                "Misalignment-rate CIs use 10,000 row-resampled percentile bootstrap draws. Transfer AUROC CIs use 10,000 class-stratified row-resampled draws, conditional on the saved source probe and split.",
                style_set["body"],
            ),
            paragraph("Artifact integrity", style_set["h2"]),
            paragraph(
                "The Phase 4 report validates generation-to-score identity, recomputed metrics, probe labels and activation shapes, all nine transfer AUROCs, and all three direction cosines before it renders these figures. "
                "All checks passed. The three activation archives needed to repeat transfer are committed with the canonical evidence.",
                style_set["body"],
            ),
        ]
    )

    organism_rows = [["Fine-tuning domain", "Organism: misaligned / coherent", "Rate, 95% CI", "Matched base"]]
    for domain in ("medical", "sports", "financial"):
        organism = summary["domains"][domain]["organism"]
        base = summary["domains"][domain]["base"]
        lo, hi = organism["misaligned_rate_ci95"]
        organism_rows.append(
            [
                DOMAIN_LABELS[domain],
                f"{organism['n_misaligned']} / {organism['n_coherent']}",
                f"{organism['misaligned_rate'] * 100:.1f}% [{lo * 100:.1f}%, {hi * 100:.1f}%]",
                f"{base['n_misaligned']} / {base['n_coherent']} ({base['misaligned_rate'] * 100:.1f}%)",
            ]
        )

    story.extend(
        [
            paragraph("2. Emergent misalignment across domains", style_set["h1"]),
            paragraph(
                "All three harmful text-domain fine-tunes increase coherent-answer misalignment relative to matched base generations. "
                "The reported rate is conditional on coherence greater than 50, so the excluded incoherent count remains important when interpreting interventions.",
                style_set["body"],
            ),
            wrapped_table(
                organism_rows,
                [1.35 * inch, 1.55 * inch, 1.65 * inch, usable_width - 4.55 * inch],
                style_set,
                font_size=7.5,
            ),
            Spacer(1, 8),
            figure_block(
                FIGURE_DIR / "em_rates.png",
                usable_width,
                "Figure 2. Coherent-answer misalignment rates. Labels show misaligned/coherent counts and error bars are 95% bootstrap intervals. "
                "Every base-model condition had zero scored misaligned coherent answers.",
                style_set,
            ),
            paragraph(
                "Negative result: an insecure-code LoRA on Qwen-Instruct has 1 / 297 coherent misaligned answers (0.34%; bootstrap CI [0.0%, 1.0%]). "
                "The upstream work reports the code organism on Qwen-Coder, so this condition is excluded from the text-domain transfer matrix.",
                style_set["body"],
            ),
        ]
    )

    transfer_rows = [["Source probe -> target organism", "Medical", "Extreme sports", "Financial"]]
    for source in ("medical", "sports", "financial"):
        transfer_rows.append(
            [
                DOMAIN_LABELS[source],
                *[f"{summary['transfer']['matrix'][source][target]:.3f}" for target in ("medical", "sports", "financial")],
            ]
        )

    story.extend(
        [
            paragraph("3. Cross-domain transfer and direction geometry", style_set["h1"]),
            paragraph(
                "Each organism independently selected layer 17 as the highest held-out AUROC layer. "
                "The per-source best-layer and fixed-layer-17 analyses are therefore identical. Rows below name the organism used to train the probe; columns name the organism scored.",
                style_set["body"],
            ),
            wrapped_table(
                transfer_rows,
                [2.25 * inch, 1.23 * inch, 1.28 * inch, usable_width - 4.76 * inch],
                style_set,
                header_background=ORANGE,
                font_size=8.0,
            ),
            Spacer(1, 8),
            figure_block(
                FIGURE_DIR / "transfer_matrix.png",
                usable_width * 0.86,
                "Figure 3. Cross-domain transfer at layer 17. Cell annotations show AUROC and its conditional class-stratified 95% bootstrap interval. "
                "The weakest off-diagonal result is financial -> sports: 0.860 [0.800, 0.913].",
                style_set,
            ),
            figure_block(
                FIGURE_DIR / "direction_cosines.png",
                usable_width * 0.90,
                "Figure 4. Pairwise cosine similarity of raw mean-difference directions. The grey band is the random absolute-cosine mean +/- one standard deviation.",
                style_set,
            ),
            paragraph(
                "Transfer and cosine measure different quantities: the former evaluates target ranking by a fitted classifier, while the latter compares raw directions. "
                "Their agreement supports a common representation, but neither result identifies a unique causal feature.",
                style_set["body"],
            ),
        ]
    )

    causal_rows = [["Domain", "Organism", "Layerwise", "Single layer", "Random control", "Base steer lambda=5"]]
    for domain, paths in CAUSAL_PATHS.items():
        metrics = {name: load_json(path) for name, path in paths.items()}
        steer = rate_cell(metrics["steer"]) + f"; {metrics['steer']['n_incoherent_excluded']} incoherent"
        causal_rows.append(
            [
                domain,
                rate_cell(metrics["organism"]),
                rate_cell(metrics["layerwise"]),
                rate_cell(metrics["single"]),
                rate_cell(metrics["random"]),
                steer,
            ]
        )

    story.extend(
        [
            paragraph("4. Causal interventions, interpretation, and limitations", style_set["h1"]),
            paragraph(
                "Counts below are misaligned/coherent. Medical is the clearest causal result: targeted ablation nearly removes the measured effect while random ablation preserves it. "
                "Sports and financial show smaller but directionally consistent reductions. Strong base steering frequently damages coherence before it produces reliable coherent misalignment.",
                style_set["body"],
            ),
            wrapped_table(
                causal_rows,
                [0.74 * inch, 0.79 * inch, 0.79 * inch, 0.79 * inch, 0.87 * inch, usable_width - 3.98 * inch],
                style_set,
                header_background=TEAL,
                font_size=6.35,
            ),
            paragraph("Interpretation", style_set["h2"]),
            paragraph(
                "Within Qwen2.5-7B-Instruct, three harmful text-domain fine-tunes produce activation features that are both geometrically aligned and predictively transferable. "
                "The medical random-direction control makes the direction more than a purely correlational classifier. "
                "Nonzero residual effects after sports and financial ablation show that a single mean-difference direction is not a complete account of every organism.",
                style_set["body"],
            ),
            paragraph("Limitations", style_set["h2"]),
            paragraph(
                "One model family and one fine-tuning seed per domain were tested. Only eight prompts were evaluated, and diagonal AUROCs hold out two question ids. "
                "Claude Haiku 4.5 replaces the papers' GPT-4o judge; Anthropic exposes no token logprobs, so this implementation parses a deterministic integer score rather than a logprob-weighted score. "
                "Bootstrap intervals are conditional row-level intervals, not uncertainty over the full training and evaluation process. Directions contrast extreme labels and do not characterize the alignment mid-band.",
                style_set["body"],
            ),
            paragraph("Repository and reproducibility", style_set["h2"]),
            paragraph(
                "The repository contains the full pipeline: training, generation, judging, probes, steering, transfer, the Phase 4 validator, canonical activation archives, and all plotted figures. "
                "Run <font name='Courier'>uv sync && make report && make technical-report</font> to validate the canonical evidence, render a fresh report run, and render this PDF.",
                style_set["body"],
            ),
            wrapped_table(
                [
                    ["Canonical artifact", "Repository location and purpose"],
                    ["Research narrative", "TECHNICAL_REPORT.md and this PDF"],
                    ["Transfer matrix", "results/runs/20260710-183425_transfer_cross_domain/"],
                    ["Validated figures", "results/runs/20260710-185310_phase4_report/"],
                    ["Repeatable PDF source", "scripts/build_technical_report_pdf.py"],
                ],
                [1.55 * inch, usable_width - 1.55 * inch],
                style_set,
                header_background=TEAL,
                font_size=7.5,
            ),
            Spacer(1, 7),
            paragraph("References", style_set["h2"]),
            paragraph(
                "[1] Jan Betley, Daniel Tan, Niels Warncke, Anna Sztyber-Betley, Xuchan Bao, Martin Soto, Nathan Labenz, and Owain Evans (2025). "
                "<i>Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs.</i> arXiv:2502.17424. "
                "https://arxiv.org/abs/2502.17424",
                style_set["ref"],
            ),
            paragraph(
                "[2] Edward Turner, Anna Soligo, Mia Taylor, Senthooran Rajamanoharan, and Neel Nanda (2025). "
                "<i>Model Organisms for Emergent Misalignment.</i> arXiv:2506.11613. "
                "https://arxiv.org/abs/2506.11613",
                style_set["ref"],
            ),
            paragraph(
                "[3] Anna Soligo, Edward Turner, Senthooran Rajamanoharan, and Neel Nanda (2025). "
                "<i>Convergent Linear Representations of Emergent Misalignment.</i> arXiv:2506.11618. "
                "https://arxiv.org/abs/2506.11618",
                style_set["ref"],
            ),
        ]
    )

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="PDF output path")
    args = parser.parse_args()
    build_pdf(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
