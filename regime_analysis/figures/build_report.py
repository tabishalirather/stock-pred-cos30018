"""
build_report.py
Assemble the six forecast figures and their commentary into a single PDF.

    python build_report.py

Re-run this after regenerating any figure with make_figure.py; it picks up
whatever PNGs are currently in this folder and recomputes the error figures
from the logged runs, so the text can never drift from the images.
"""

import glob
import os
import sys

import numpy as np
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer)

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

import driver  # noqa: E402  (needs PARENT on the path first)

def _results_dir(base):
    """Honour the REGIME_RESULTS_DIR override used by driver.py, so runs
    written to an independent directory can be plotted the same way."""
    path = os.environ.get("REGIME_RESULTS_DIR") or os.path.join(base, "results")
    if not os.path.isabs(path):
        path = os.path.join(base, path)
    return path


PRED = os.path.join(_results_dir(PARENT), "preds")
PRED_PYKAN = os.path.join(_results_dir(PARENT), "preds_pykan")


# ----------------------------------------------------------------- metrics
def line_rmse(model, horizon, mode, seed=0):
    """RMSE of exactly the line drawn in the figure, from the logged runs."""
    if model == "kan":
        pattern = os.path.join(PRED_PYKAN, "kan__-__h%d__s%d__f*.npz" % (horizon, seed))
    else:
        config = driver.PRIMARY_LSTM_CONFIG[horizon]
        pattern = os.path.join(PRED, "lstm__%s__h%d__s%d__f*.npz" % (config, horizon, seed))

    paths = sorted(glob.glob(pattern),
                   key=lambda p: int(p.split("__f")[-1].split(".")[0]))
    if not paths:
        return None
    with np.load(paths[-1], allow_pickle=False) as data:      # last fold
        prediction = data["prediction"].astype(float)
        truth = data["truth"].astype(float)

    if mode == "step":
        a, b = prediction[:, 0], truth[:, 0]
    else:
        a, b = prediction[-1], truth[-1]
    return float(np.sqrt(np.mean((a - b) ** 2)))


def errors(horizon, mode):
    kan = line_rmse("kan", horizon, mode)
    lstm = line_rmse("lstm", horizon, mode)
    return kan, lstm


# ------------------------------------------------------------------ styles
styles = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=styles["Normal"], fontSize=10.5,
                      leading=15, alignment=TA_JUSTIFY, spaceAfter=8)
H1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=19, leading=23,
                    spaceAfter=4)
H2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13, leading=17,
                    spaceBefore=14, spaceAfter=6, textColor="#1a3d5c")
SUB = ParagraphStyle("sub", parent=styles["Normal"], fontSize=10,
                     textColor="#555555", spaceAfter=14)
NOTE = ParagraphStyle("note", parent=BODY, fontSize=10, leading=14,
                      leftIndent=10, textColor="#444444", spaceBefore=4)
CAPTION = ParagraphStyle("caption", parent=styles["Normal"], fontSize=9.5,
                         textColor="#555555", spaceBefore=3, spaceAfter=12)

PAGE_WIDTH = A4[0] - 4 * cm


def picture(filename):
    """Place a figure scaled to the text width, keeping its aspect ratio."""
    path = os.path.join(HERE, filename)
    if not os.path.exists(path):
        return Paragraph("[missing figure: %s]" % filename, BODY)
    from PIL import Image as PILImage
    with PILImage.open(path) as handle:
        width, height = handle.size
    scale = PAGE_WIDTH / width
    return Image(path, width=PAGE_WIDTH, height=height * scale)


def section(story, title, filename, caption, paragraphs, kan, lstm,
            x_axis=None, y_axis=None):
    story.append(Paragraph(title, H2))
    story.append(picture(filename))
    story.append(Paragraph(caption, CAPTION))
    if x_axis and y_axis:
        story.append(Paragraph(
            "<b>Horizontal axis:</b> %s<br/><b>Vertical axis:</b> %s"
            % (x_axis, y_axis), NOTE))
    if kan is not None and lstm is not None:
        story.append(Paragraph(
            "<b>Error on this line:</b> KAN %.3f, LSTM %.3f (lower is better)."
            % (kan, lstm), BODY))
    for text in paragraphs:
        story.append(Paragraph(text, BODY))


def main():
    story = []

    story.append(Paragraph("Forecast Figures: What They Show", H1))
    story.append(Paragraph(
        "KAN and LSTM one-day to two-hundred-day forecasts on CBA.AX, "
        "under the corrected evaluation protocol.", SUB))

    story.append(Paragraph("1. How a forecast is produced", H2))
    story.append(Paragraph(
        "Each model works on a sliding window. It is shown the closing prices "
        "of the last 20 trading days, and from those it predicts the next "
        "<i>h</i> days in one go, where <i>h</i> is the forecast horizon. "
        "So a 100-day model outputs one hundred numbers at once, not a single "
        "number.", BODY))
    story.append(Paragraph(
        "The window then slides forward one day and the model does it again. "
        "Repeating this across the test period produces a great many "
        "overlapping forecasts, which is why there is more than one sensible "
        "way to draw the result.", BODY))
    story.append(picture("explainer_diagram.png"))
    story.append(Paragraph(
        "Figure 0. How a forecast is made, and what each style of plot takes "
        "from it.", CAPTION))

    story.append(PageBreak())

    story.append(Paragraph("2. The two kinds of plot", H2))
    story.append(Paragraph(
        "<b>Step plots (Figures 1 to 4).</b> Pick one distance into the "
        "future, for example one day ahead, and follow only that prediction "
        "as the window slides forward. Panel B of the diagram shows this: the "
        "highlighted square in each row is the same distance ahead each time, "
        "and joining them produces one line. The question answered is: "
        "<i>if the model predicts N days ahead, how well does it do, day "
        "after day?</i>", BODY))
    story.append(Paragraph(
        "In a step plot the horizontal axis is <b>calendar time</b>, labelled "
        "with dates. Each point on the line is a separate forecast, made on a "
        "different day.", NOTE))
    story.append(Paragraph(
        "<b>Trajectory plots (Figures 5 and 6).</b> Pick one date and show "
        "the entire forecast made on that date, all <i>h</i> days of it, "
        "against what actually happened. Panel C of the diagram shows this. "
        "The question answered is: <i>standing on this particular day, what "
        "did the model think the next hundred days would look like?</i>", BODY))
    story.append(Paragraph(
        "In a trajectory plot the horizontal axis is <b>days ahead</b>, "
        "labelled +1, +2 and so on, not calendar dates. The whole line comes "
        "from a single forecast rather than many.", NOTE))
    story.append(Paragraph(
        "Step plots are the ones to use when comparing models, because they "
        "average over many forecasts. Trajectory plots are better for showing "
        "how a model behaves when asked to see a long way ahead.", BODY))

    story.append(Paragraph("3. What the axes mean in every figure", H2))
    story.append(Paragraph(
        "<b>The vertical axis is always the closing price of the stock, "
        "scaled to a range of 0 to 1.</b> The lowest price in the sample "
        "period becomes 0, the highest becomes 1, and everything else sits "
        "in between. So 0.85 means the price is at 85 percent of the way "
        "between the cheapest and dearest days on record for this stock. "
        "These are not dollars. The scaling was applied before the models "
        "were trained and all published error figures use the same scale, so "
        "the numbers are comparable with the paper's tables.", BODY))
    story.append(Paragraph(
        "<b>The horizontal axis depends on the plot type</b>, as described "
        "above: calendar dates for step plots, days ahead for trajectory "
        "plots.", BODY))
    story.append(Paragraph(
        "<b>The lines.</b> Black is what actually happened. Blue is the LSTM "
        "forecast. Orange is the KAN forecast. Where a coloured line sits on "
        "top of the black one, the model was right; where it departs, that "
        "gap is the error.", BODY))
    story.append(Paragraph(
        "<b>The error figure quoted under each plot</b> is the root mean "
        "square error of that particular line: roughly, the typical size of "
        "the vertical gap between the coloured line and the black one. Lower "
        "is better, and because the prices are scaled, an error of 0.02 means "
        "a typical miss of about 2 percent of the stock's full price range.", BODY))
    story.append(Paragraph(
        "Every figure uses the final walk-forward test fold, which is the "
        "most recent stretch of data, and one the models never saw during "
        "training.", BODY))

    story.append(PageBreak())

    kan, lstm = errors(1, "step")
    section(story, "1. One day ahead", "figure_h1_step_s0.png",
            "Figure 1. One-day-ahead forecasts against the actual price, "
            "February to July 2023.",
            ["Both models stay close to the actual price, which is not "
             "surprising, since tomorrow's price is usually near today's. The "
             "KAN follows the daily movement more closely. The LSTM is "
             "smoother and tends to arrive at each turn slightly late."],
            kan, lstm,
            x_axis="calendar date, from February to July 2023. Each point is a fresh forecast made on that day.",
            y_axis="closing price, scaled from 0 to 1.")

    kan, lstm = errors(2, "step")
    section(story, "2. Two days ahead", "figure_h2_step_s0.png",
            "Figure 2. Two-day-ahead forecasts against the actual price.",
            ["This is the clearest of the six. The KAN tracks the real "
             "movement, including the sharp falls in mid February and early "
             "June. The LSTM has effectively drawn a smooth average through "
             "the data: it captures the general shape but misses almost every "
             "turning point, and from late June it drifts well below the "
             "actual price as the market climbs.",
             "If one figure had to show that the two models do not perform as "
             "previously reported, this is the one."],
            kan, lstm,
            x_axis="calendar date. Each point is a forecast made two days before the day being predicted.",
            y_axis="closing price, scaled from 0 to 1.")

    story.append(PageBreak())

    kan, lstm = errors(100, "step")
    section(story, "3. One hundred days ahead", "figure_h100_step_s0.png",
            "Figure 3. Forecasts made one hundred days in advance.",
            ["The same comparison, but each forecast is now made a hundred "
             "days before the event. Both models are visibly further from the "
             "truth, and the LSTM sits noticeably above the actual price for "
             "much of the period."],
            kan, lstm,
            x_axis="calendar date. Each point is a forecast made one hundred days in advance.",
            y_axis="closing price, scaled from 0 to 1.")

    kan, lstm = errors(200, "step")
    section(story, "4. Two hundred days ahead", "figure_h200_step_s0.png",
            "Figure 4. Forecasts made two hundred days in advance.",
            ["At this distance neither model has much real information to work "
             "with. Both lines stay within a narrow band while the actual "
             "price moves around them."],
            kan, lstm,
            x_axis="calendar date. Each point is a forecast made two hundred days in advance.",
            y_axis="closing price, scaled from 0 to 1.")

    story.append(PageBreak())

    kan, lstm = errors(100, "trajectory")
    section(story, "5. A single hundred-day forecast path",
            "figure_h100_trajectory_s0.png",
            "Figure 5. One starting date, and what each model expected over "
            "the following hundred days.",
            ["The LSTM produces an almost flat line with a slight upward "
             "drift. It has essentially learned the average price level and "
             "predicts that, rather than any movement. The KAN swings well "
             "below the real price through the middle stretch before "
             "recovering.",
             "Two quite different ways of being wrong, and neither is visible "
             "in a table of error figures."],
            kan, lstm,
            x_axis="days ahead of the chosen starting date, from +1 to +100. This is not calendar time.",
            y_axis="closing price, scaled from 0 to 1.")

    kan, lstm = errors(200, "trajectory")
    section(story, "6. A single two-hundred-day forecast path",
            "figure_h200_trajectory_s0.png",
            "Figure 6. One starting date, and the following two hundred days.",
            ["The most honest picture in the set. The actual price rises to "
             "the top of its range around day 75 and swings considerably. Both "
             "forecasts stay inside a narrow band for the entire two hundred "
             "days, drifting gently.",
             "Neither model is predicting the future here. Both have learned "
             "roughly where the price tends to sit, and little else."],
            kan, lstm,
            x_axis="days ahead of the chosen starting date, from +1 to +200. This is not calendar time.",
            y_axis="closing price, scaled from 0 to 1.")

    story.append(PageBreak())

    story.append(Paragraph("The short version", H2))
    story.append(Paragraph(
        "At short horizons the KAN follows real movement more closely, while "
        "the LSTM smooths it away. Errors grow with distance for both, as "
        "expected. At long horizons neither model forecasts in any meaningful "
        "sense: they settle near an average and stay there.", BODY))

    story.append(Paragraph("One caution", H2))
    story.append(Paragraph(
        "Figures 5 and 6 each show a single starting date, so they illustrate "
        "behaviour rather than prove it. The errors quoted for them apply to "
        "that one path and will not match the pooled figures in the paper's "
        "tables. Use them to show <i>how</i> the models behave, not "
        "<i>how well</i>.", BODY))

    story.append(Paragraph("Regenerating these", H2))
    story.append(Paragraph(
        "All six figures come from make_figure.py in the folder above. Change "
        "the settings at the top of that file and run it; each run writes a "
        "PNG for viewing and a .tex version for the paper. Re-run "
        "build_report.py afterwards to rebuild this document, which recomputes "
        "every error figure from the logged runs so the text cannot drift "
        "from the images.", BODY))

    output = os.path.join(HERE, "Forecast_Figures_Report.pdf")
    document = SimpleDocTemplate(
        output, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title="Forecast Figures: What They Show",
        author="S M Mahmudul Hasan Joy",
    )
    document.build(story)
    print("wrote %s" % output)


if __name__ == "__main__":
    main()
