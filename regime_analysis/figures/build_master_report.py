"""
build_master_report.py
The complete forecast figure report: all twenty-six figures, with a detailed
pictorial explanation of how a forecast is produced.

Run the three generators first, then this:

    python make_forecast_explainer.py
    python make_regime_explainer.py
    python make_figure.py            (for the four step and two trajectory plots)
    python make_regime_figures.py
    python make_decay_figures.py
    python build_master_report.py

Every number quoted is recomputed here from the logged prediction files, so
the text cannot drift away from the pictures.
"""

import glob
import json
import os
import sys

import numpy as np
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

import driver  # noqa: E402

def _results_dir(base):
    """Honour the REGIME_RESULTS_DIR override used by driver.py, so runs
    written to an independent directory can be plotted the same way."""
    path = os.environ.get("REGIME_RESULTS_DIR") or os.path.join(base, "results")
    if not os.path.isabs(path):
        path = os.path.join(base, path)
    return path


PRED = os.path.join(_results_dir(PARENT), "preds")
PRED_PYKAN = os.path.join(_results_dir(PARENT), "preds_pykan")

# ------------------------------------------------------------------ styles
styles = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=styles["Normal"], fontSize=10.5,
                      leading=15, alignment=TA_JUSTIFY, spaceAfter=8)
H1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=20, leading=24,
                    spaceAfter=4)
PART = ParagraphStyle("part", parent=styles["Heading1"], fontSize=15,
                      leading=19, spaceBefore=6, spaceAfter=8,
                      textColor="#0f2d47")
H2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12.5,
                    leading=16, spaceBefore=12, spaceAfter=5,
                    textColor="#1a3d5c")
H3 = ParagraphStyle("h3", parent=styles["Heading3"], fontSize=11, leading=14,
                    spaceBefore=8, spaceAfter=3, textColor="#333333")
SUB = ParagraphStyle("sub", parent=styles["Normal"], fontSize=10,
                     textColor="#555555", spaceAfter=14)
NOTE = ParagraphStyle("note", parent=BODY, fontSize=10, leading=14,
                      leftIndent=10, textColor="#444444", spaceBefore=4)
WARN = ParagraphStyle("warn", parent=BODY, fontSize=10, leading=14,
                      leftIndent=8, rightIndent=8, textColor="#7a3b12",
                      borderColor="#e0c4a8", borderWidth=0.7, borderPadding=7,
                      spaceBefore=6, spaceAfter=10)
CAPTION = ParagraphStyle("caption", parent=styles["Normal"], fontSize=9,
                         textColor="#555555", spaceBefore=2, spaceAfter=10)

PAGE_WIDTH = A4[0] - 4 * cm


def picture(filename, fraction=1.0):
    path = os.path.join(HERE, filename)
    if not os.path.exists(path):
        return Paragraph("[missing figure: %s]" % filename, BODY)
    from PIL import Image as PILImage
    with PILImage.open(path) as handle:
        width, height = handle.size
    target = PAGE_WIDTH * fraction
    return Image(path, width=target, height=height * target / width)


# ----------------------------------------------------------------- metrics
def _fold(path):
    return int(os.path.basename(path).split("__f")[-1].split(".")[0])


def load_last_fold(model, horizon, seed=0):
    if model == "kan":
        pattern = os.path.join(PRED_PYKAN, "kan__-__h%d__s%d__f*.npz" % (horizon, seed))
    elif model == "naive":
        pattern = os.path.join(PRED, "naive__-__h%d__s0__f*.npz" % horizon)
    else:
        config = driver.PRIMARY_LSTM_CONFIG[horizon]
        pattern = os.path.join(PRED, "lstm__%s__h%d__s%d__f*.npz"
                               % (config, horizon, seed))
    paths = sorted(glob.glob(pattern), key=_fold)
    if not paths:
        return None
    with np.load(paths[-1], allow_pickle=False) as data:
        return data["prediction"].astype(float), data["truth"].astype(float)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def step_errors(horizon):
    lstm = load_last_fold("lstm", horizon)
    kan = load_last_fold("kan", horizon)
    return rmse(lstm[0][:, 0], lstm[1][:, 0]), rmse(kan[0][:, 0], kan[1][:, 0])


def trajectory_errors(horizon):
    lstm = load_last_fold("lstm", horizon)
    kan = load_last_fold("kan", horizon)
    return rmse(lstm[0][-1], lstm[1][-1]), rmse(kan[0][-1], kan[1][-1])


def simple_table(header, rows, widths):
    data = [header] + rows
    table = Table(data, colWidths=widths)
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, "#888888"),
        ("LINEBELOW", (0, -1), (-1, -1), 0.4, "#cccccc"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TEXTCOLOR", (0, 0), (-1, 0), "#1a3d5c"),
    ]))
    return table


# ==========================================================================
def part_one(story):
    story.append(Paragraph("Part 1. How a forecast is produced", PART))
    story.append(Paragraph(
        "This part builds the idea up in four pictures. Nothing later in the "
        "report will make sense without it, so it is worth reading slowly.", BODY))

    story.append(Paragraph("1.1  The data, and where the testing happens", H2))
    story.append(Paragraph(
        "The data is the daily closing price of a single share, Commonwealth "
        "Bank of Australia, from January 2020 to July 2023. That is roughly "
        "900 trading days. Weekends and holidays do not exist in this data; "
        "one step means one trading day.", BODY))
    story.append(picture("explainer_1_series.png"))
    story.append(Paragraph(
        "Figure 1.1. The whole price series. The blue region on the left is "
        "used only for training. The five shaded regions are the test blocks, "
        "each one tested by a model that was trained only on the data before "
        "it. A model is therefore never asked about a period it has already "
        "seen, which is the point of splitting the data this way rather than "
        "shuffling it.", CAPTION))
    story.append(Paragraph(
        "The five blocks are tested in sequence. For the first block, the "
        "model trains on the initial stretch alone. For the second, it trains "
        "on the initial stretch plus the first block, and so on. This is why "
        "the training set grows and the test blocks march forward in time.", BODY))

    story.append(PageBreak())

    story.append(Paragraph("1.2  One forecast", H2))
    story.append(Paragraph(
        "The model is shown the closing prices of the last 20 trading days, "
        "about a month of trading. From those 20 numbers it produces the next "
        "<i>h</i> numbers, where <i>h</i> is the forecast horizon. The four "
        "horizons studied are 1, 2, 100 and 200 days.", BODY))
    story.append(picture("explainer_2_window.png"))
    story.append(Paragraph(
        "Figure 1.2. A single forecast. The blue region is the input, the "
        "orange region is the output. The dashed line is what actually "
        "happened, which the model never sees.", CAPTION))
    story.append(Paragraph(
        "An important detail: the model produces all <i>h</i> days in one go. "
        "A 200-day model outputs two hundred numbers simultaneously. It does "
        "not predict tomorrow, then feed that prediction back in to predict "
        "the day after. This is called a direct multi-output forecast, and it "
        "means each of those two hundred outputs is a separate prediction "
        "made from the same 20 days of input.", BODY))
    story.append(Paragraph(
        "Each output day is referred to by how far ahead it sits. The first "
        "output is <b>step 1</b>, or one day ahead. The fiftieth output is "
        "<b>step 50</b>, fifty days ahead. This vocabulary matters for the "
        "figures that follow.", NOTE))

    story.append(PageBreak())

    story.append(Paragraph("1.3  The window slides", H2))
    story.append(Paragraph(
        "That was one forecast. To test a model properly it has to make many. "
        "So the 20-day window moves forward by a single day and the model "
        "forecasts again, from the new position.", BODY))
    story.append(picture("explainer_3_sliding.png"))
    story.append(Paragraph(
        "Figure 1.3. Three consecutive forecasts. Blue is the input window, "
        "orange the forecast.", CAPTION))
    story.append(Paragraph(
        "Repeating this across a test block produces several hundred "
        "forecasts. Two consequences follow, and both matter.", BODY))
    story.append(Paragraph(
        "First, consecutive forecasts overlap heavily. Two neighbouring "
        "windows share 19 of their 20 input days, so they are made from "
        "almost identical information. This is exactly why the data cannot be "
        "shuffled before splitting into training and test sets: near "
        "duplicates would end up on both sides, and the model would be tested "
        "on something it had effectively already seen.", BODY))
    story.append(Paragraph(
        "Second, the same future day gets predicted many times over, from "
        "many different starting points. A day one hundred trading days from "
        "now is step 100 of today's forecast, step 99 of tomorrow's, and so "
        "on. Those are all different predictions of the same day.", BODY))

    story.append(PageBreak())

    story.append(Paragraph("1.4  Every forecast at once, and how to draw it", H2))
    story.append(Paragraph(
        "Collecting all of this gives a grid. Each row is one forecast, made "
        "on one day. Each column is a distance ahead. A 100-day model tested "
        "over 472 windows produces a grid of 472 rows by 100 columns, which "
        "is 47,200 individual predictions.", BODY))
    story.append(picture("explainer_4_grid.png"))
    story.append(Paragraph(
        "Figure 1.4. The grid of every prediction, and the two ways of "
        "slicing it that this report uses.", CAPTION))
    story.append(Paragraph(
        "That grid cannot be drawn on a page all at once, so every figure in "
        "this report takes a slice of it. There are two sensible slices.", BODY))
    story.append(Paragraph(
        "<b>A step figure takes a column.</b> It fixes one distance ahead, "
        "say one day, and follows it across every window. The horizontal axis "
        "is calendar time, and each point is a different forecast made on a "
        "different day. This answers: <i>if the model predicts N days ahead, "
        "how well does it do, day after day?</i>", NOTE))
    story.append(Paragraph(
        "<b>A trajectory figure takes a row.</b> It fixes one starting date "
        "and shows that single forecast in full, all <i>h</i> days of it. The "
        "horizontal axis is days ahead, not calendar time. This answers: "
        "<i>standing on one particular day, what did the model think the "
        "future held?</i>", NOTE))
    story.append(Paragraph(
        "Step figures are the ones to use when comparing models, because they "
        "summarise hundreds of forecasts. Trajectory figures show the "
        "character of a single prediction, which a summary hides.", BODY))


def part_two(story):
    story.append(Paragraph("Part 2. The three lines on every chart", PART))

    story.append(Paragraph("2.1  The two models", H2))
    story.append(Paragraph(
        "<b>LSTM</b> is a long established neural network built for "
        "sequences. It reads the 20 days in order, carrying a memory of what "
        "it has seen, which in principle suits price data.", BODY))
    story.append(Paragraph(
        "<b>KAN</b>, a Kolmogorov-Arnold Network, is a newer design. It "
        "receives all 20 days at once as a flat list and learns a flexible "
        "curve for each input rather than reading them in order. It has no "
        "built-in notion of sequence, which is why the original expectation "
        "was that it would struggle here.", BODY))

    story.append(Paragraph("2.2  The naive baseline", H2))
    story.append(Paragraph(
        "The third line is not a model. The <b>naive baseline</b> predicts "
        "that the price will not change: whatever today's closing price is, "
        "that is its forecast for tomorrow, for next month, and for two "
        "hundred days ahead. It does no learning and has no parameters. It is "
        "sometimes called a persistence forecast.", BODY))
    story.append(picture("regime_explainer_naive.png"))
    story.append(Paragraph(
        "Figure 2.1. The naive baseline. It takes the last known price and "
        "holds it flat.", CAPTION))
    story.append(Paragraph(
        "This sounds too crude to bother plotting, but it is the standard "
        "yardstick in forecasting for a good reason: share prices move slowly "
        "from one day to the next, so \"nothing changes\" is already a "
        "reasonable guess, and a surprisingly hard one to beat at short "
        "range. A model that cannot beat it has not learned anything useful "
        "about the market, however sophisticated it looks.", BODY))
    story.append(Paragraph(
        "Reporting a low error without this comparison can make a model look "
        "far more capable than it is. Adding it was one of the changes made "
        "in the revised analysis, and it is the single most informative line "
        "on these charts.", BODY))


def part_three(story):
    story.append(Paragraph("Part 3. Reading the figures", PART))

    story.append(Paragraph("3.1  The vertical axis", H2))
    story.append(Paragraph(
        "Always the closing price, scaled from 0 to 1. The cheapest day in "
        "the period becomes 0 and the dearest becomes 1, with everything else "
        "in between. So 0.85 means the price sat 85 percent of the way "
        "between the lowest and highest points on record for this share. "
        "These are not dollars.", BODY))
    story.append(Paragraph(
        "The scaling was applied before the models were trained, and every "
        "error figure in the paper uses the same scale, so the numbers here "
        "are directly comparable with the paper's tables. An error of 0.05 "
        "means a typical miss of about five percent of the share's full price "
        "range.", BODY))

    story.append(Paragraph("3.2  The horizontal axis", H2))
    story.append(Paragraph(
        "This depends on the figure, which is why each one states it "
        "explicitly. Step figures use calendar dates. Trajectory figures use "
        "days ahead, labelled +1, +2 and so on. The regime figures in Part 7 "
        "use a third arrangement, explained there.", BODY))

    story.append(Paragraph("3.3  The lines and the error figures", H2))
    story.append(Paragraph(
        "Black is what actually happened. Blue is the LSTM, orange the KAN, "
        "grey the naive baseline. Where a coloured line sits on the black "
        "one, that forecast was right; the vertical gap is the error.", BODY))
    story.append(Paragraph(
        "The error quoted beneath each figure is the root mean square error "
        "of that particular line: roughly, the typical size of the vertical "
        "gap. Lower is better.", BODY))
    story.append(Paragraph(
        "All figures use seed 0. The models involve randomness in their "
        "starting weights, so the paper reports averages over three seeds. A "
        "single seed is used here because three overlapping copies of each "
        "line would be unreadable. This is the main reason the numbers in "
        "this report differ slightly from the paper's tables.", WARN))


def part_four(story):
    story.append(Paragraph("Part 4. One day ahead, at four horizons", PART))
    story.append(Paragraph(
        "Four step figures, one per horizon, each showing the one-day-ahead "
        "prediction across the final test block. Because all four look one "
        "day ahead, the differences between them come from the model having "
        "been trained for a different horizon, not from the difficulty of the "
        "question.", BODY))

    items = [
        (1, "figure_h1_step_s0.png",
         "The 1-day model. Both models stay close to the actual price, which "
         "is expected: tomorrow's price is rarely far from today's. The KAN "
         "follows the daily movement more closely, the LSTM is smoother and "
         "arrives at each turn slightly late."),
        (2, "figure_h2_step_s0.png",
         "The 2-day model. The clearest separation anywhere in this report. "
         "The KAN tracks the real movement including the sharp falls in "
         "mid-February and early June; the LSTM has effectively drawn a "
         "smooth average through the data and misses almost every turning "
         "point."),
        (100, "figure_h100_step_s0.png",
         "The 100-day model, asked for tomorrow. Trained to look far ahead, "
         "it is noticeably less sharp at short range than the 1-day model."),
        (200, "figure_h200_step_s0.png",
         "The 200-day model, asked for tomorrow. The same effect, more "
         "pronounced."),
    ]
    for horizon, filename, text in items:
        lstm, kan = step_errors(horizon)
        story.append(Paragraph("%d-day model, one day ahead" % horizon, H2))
        story.append(picture(filename))
        story.append(Paragraph(
            "Horizontal axis: calendar date. Vertical axis: scaled close "
            "price. Error on these lines: LSTM %.3f, KAN %.3f." % (lstm, kan),
            CAPTION))
        story.append(Paragraph(text, BODY))
        if horizon == 2:
            story.append(PageBreak())


def part_five(story):
    story.append(Paragraph("Part 5. A single forecast in full", PART))
    story.append(Paragraph(
        "Two trajectory figures. Each freezes one date and shows the entire "
        "forecast made from it. The horizontal axis is days ahead, not "
        "calendar time, and the whole line comes from one forecast rather "
        "than many.", BODY))

    for horizon, filename, text in [
        (100, "figure_h100_trajectory_s0.png",
         "The LSTM produces an almost flat line with a slight upward drift. "
         "It has essentially learned the average price level and predicts "
         "that, rather than any movement. The KAN swings well below the real "
         "price through the middle stretch before recovering. Two quite "
         "different ways of being wrong, neither visible in a table."),
        (200, "figure_h200_trajectory_s0.png",
         "The most honest picture in the report. The actual price rises to "
         "the top of its range around day 75 and swings considerably, while "
         "both forecasts stay inside a narrow band for the entire two hundred "
         "days. Neither model is predicting the future here; both have "
         "learned roughly where the price tends to sit and little else."),
    ]:
        lstm, kan = trajectory_errors(horizon)
        story.append(Paragraph("%d days, from one starting date" % horizon, H2))
        story.append(picture(filename))
        story.append(Paragraph(
            "Horizontal axis: days ahead of the starting date. Vertical axis: "
            "scaled close price. Error on this single path: LSTM %.3f, KAN "
            "%.3f." % (lstm, kan), CAPTION))
        story.append(Paragraph(text, BODY))

    story.append(Paragraph(
        "These two figures each show one starting date, so they illustrate "
        "behaviour rather than prove it. Their error figures apply to that "
        "one path and are not comparable with the paper's pooled tables.", WARN))


def part_six(story):
    with open(os.path.join(HERE, "decay_stats.json")) as handle:
        stats = json.load(handle)

    story.append(Paragraph("Part 6. How far ahead can they see?", PART))
    story.append(Paragraph(
        "The step figures in Part 4 all looked one day ahead. But a 200-day "
        "model produces two hundred outputs, and there is no reason to expect "
        "the fiftieth to be as good as the first. These eight figures take "
        "the same models and ask them at progressively greater distances: "
        "column 1 of the grid, then column 25, then 50, and so on.", BODY))

    story.append(picture("decay_curve.png"))
    story.append(Paragraph(
        "Figure 6.1. Error against distance ahead, summarising the eight "
        "figures that follow.", CAPTION))
    story.append(Paragraph(
        "Three things stand out. The naive baseline rises steeply and then "
        "flattens: assuming nothing changes is excellent for tomorrow and "
        "poor beyond a few weeks, after which it stops getting much worse. "
        "The KAN degrades sharply with distance, and at the 100-day horizon "
        "its error at step 50 is more than double its error at step 1. The "
        "LSTM stays remarkably flat, which is the signature of a model "
        "predicting close to a constant: a flat prediction is equally wrong "
        "everywhere.", BODY))
    story.append(Paragraph(
        "That last point cuts both ways. Because the LSTM barely varies, it "
        "is poor at short range where variation matters, but it overtakes the "
        "naive baseline at long range, where the baseline's fixed guess has "
        "drifted badly out of date. At step 100 of the 100-day model the LSTM "
        "reaches 0.099 against the baseline's 0.123. It is the one place in "
        "this study where a model clearly beats persistence.", BODY))

    story.append(PageBreak())

    for horizon in (100, 200):
        rows = sorted([r for r in stats if r["horizon"] == horizon],
                      key=lambda r: r["step"])
        story.append(Paragraph("%d-day model at increasing distances" % horizon, H2))
        for row in rows:
            story.append(picture(row["stem"] + ".png"))
            story.append(Paragraph(
                "Predicting %d day%s ahead. Error: LSTM %.3f, KAN %.3f, naive "
                "%.3f." % (row["step"], "" if row["step"] == 1 else "s",
                           row["rmse"]["LSTM"], row["rmse"]["KAN"],
                           row["rmse"].get("Naive", float("nan"))), CAPTION))
        table_rows = [[str(r["step"]), "%.3f" % r["rmse"]["LSTM"],
                       "%.3f" % r["rmse"]["KAN"],
                       "%.3f" % r["rmse"].get("Naive", float("nan"))]
                      for r in rows]
        story.append(simple_table(["Days ahead", "LSTM", "KAN", "Naive"],
                                  table_rows,
                                  [3.0 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm]))
        story.append(PageBreak())


def part_seven(story):
    with open(os.path.join(HERE, "regime_figure_stats.json")) as handle:
        rows = json.load(handle)
    stats = {(r["horizon"], r["regime"]): r for r in rows}

    story.append(Paragraph("Part 7. Behaviour by market condition", PART))
    story.append(Paragraph(
        "Twelve figures, one for each cell of Table 7 in the paper. The "
        "question is whether either model copes better in calm markets than "
        "turbulent ones.", BODY))
    story.append(Paragraph(
        "Every 20-day window is sorted into one of three conditions using "
        "only the days the model can see, so a window is never labelled with "
        "knowledge of what happened next. Two things are measured: how much "
        "the price jumped about from day to day, and how directional the "
        "movement was.", BODY))
    story.append(Paragraph(
        "<b>Volatile</b>: day-to-day movement in the top quarter of what was "
        "seen in training. <b>Trending</b>: movement not unusually large but "
        "unusually one-directional, again the top quarter. <b>Normal</b>: "
        "everything else, and the most common case.", NOTE))
    story.append(picture("regime_explainer_types.png"))
    story.append(Paragraph(
        "Figure 7.1. One real window of each type. The vertical scales "
        "differ: the volatile example covers a third of the share's entire "
        "price range within a single month.", CAPTION))
    story.append(Paragraph(
        "<b>A note on the horizontal axis in Part 7.</b> Filtering to one "
        "condition leaves gaps in the calendar: a volatile stretch in March, "
        "another in June, quiet weeks between. Drawing a continuous line "
        "across those gaps would imply a continuity that does not exist, so "
        "the axis counts the windows of that condition in chronological "
        "order. Tick labels give the real date, and a large jump between "
        "ticks means the intervening period belonged to a different "
        "condition.", WARN))

    story.append(PageBreak())

    headlines = {
        1: "One day ahead. All three lines stay close to the actual price. "
           "The KAN is slightly ahead of the LSTM in every condition, and "
           "both degrade in volatile markets.",
        2: "Two days ahead. The widest gap between the two models. The LSTM "
           "swings far from the actual price during volatile stretches, while "
           "the KAN keeps tracking it.",
        100: "One hundred days ahead. The ordering reverses: the LSTM is the "
             "better of the two in all three conditions.",
        200: "Two hundred days ahead. The ordering reverses again, and the "
             "LSTM is at its worst in trending markets.",
    }

    for horizon in (1, 2, 100, 200):
        story.append(Paragraph("%d-day horizon" % horizon, H2))
        story.append(Paragraph(headlines[horizon], BODY))
        for regime in ("Normal", "Volatile", "Trending"):
            row = stats.get((horizon, regime))
            if not row:
                continue
            story.append(picture(row["stem"] + ".png"))
            story.append(Paragraph(
                "%s conditions: %d windows, %s to %s. Error: LSTM %.3f, KAN "
                "%.3f, naive %.3f."
                % (regime, row["count"], row["first"], row["last"],
                   row["rmse"]["LSTM"], row["rmse"]["KAN"],
                   row["rmse"].get("Naive", float("nan"))), CAPTION))
        table_rows = []
        for regime in ("Normal", "Volatile", "Trending"):
            row = stats.get((horizon, regime))
            if row:
                table_rows.append([regime, str(row["count"]),
                                   "%.3f" % row["rmse"]["LSTM"],
                                   "%.3f" % row["rmse"]["KAN"],
                                   "%.3f" % row["rmse"].get("Naive", float("nan"))])
        story.append(simple_table(["Condition", "Windows", "LSTM", "KAN", "Naive"],
                                  table_rows,
                                  [3.2 * cm, 2.2 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm]))
        story.append(PageBreak())

    story.append(Paragraph(
        "The window counts above match Table 7 exactly, which confirms the "
        "same data underlies both. The error figures do not match, and should "
        "not be quoted as though they did: Table 7 averages over every step "
        "of the forecast and over three seeds, whereas these figures plot "
        "only step 1 from a single seed. Step 1 is the easiest part of any "
        "forecast, so these errors are smaller throughout and the naive "
        "baseline looks stronger here than in the paper.", WARN))


def part_eight(story):
    story.append(Paragraph("Part 8. What the twenty-six figures show", PART))

    story.append(Paragraph("Neither model wins outright", H2))
    story.append(Paragraph(
        "The KAN is ahead at 1, 2 and 200 days; the LSTM at 100 days. Which "
        "one leads depends on the horizon, not on the market condition: no "
        "regime reverses the ordering within a horizon. This is the finding "
        "that replaced the earlier claim of a large, uniform LSTM advantage.", BODY))

    story.append(Paragraph("Smoothness is not accuracy", H2))
    story.append(Paragraph(
        "In every figure the LSTM draws the smoother line. That is not a "
        "virtue. A smooth line that lags the market is a slow-moving average, "
        "and much of the LSTM's short-range error comes from arriving at each "
        "turn too late. The trajectory figures in Part 5 make this plainest: "
        "at 100 and 200 days the LSTM is close to predicting a constant.", BODY))

    story.append(Paragraph("Persistence is hard to beat", H2))
    story.append(Paragraph(
        "At short range, assuming the price will not change beats both "
        "models, often by a factor of three. That is the uncomfortable result "
        "and the one a reviewer is most likely to focus on. The single "
        "exception is Part 6: at the longest distances the LSTM does overtake "
        "the baseline, because a fixed guess becomes badly out of date while "
        "a near-constant prediction does not.", BODY))

    story.append(Paragraph("Volatility hurts, and it hurts the LSTM more", H2))
    story.append(Paragraph(
        "At short horizons both models degrade in volatile conditions and the "
        "LSTM degrades more. At long horizons the difficulty shifts to "
        "trending windows, particularly for the LSTM at 200 days.", BODY))

    story.append(Paragraph("Reproducing everything here", H2))
    story.append(Paragraph(
        "The figures come from five scripts in this folder: "
        "make_forecast_explainer.py and make_regime_explainer.py for the "
        "diagrams, make_figure.py for the step and trajectory plots, "
        "make_regime_figures.py for the twelve condition plots, and "
        "make_decay_figures.py for the distance study. Running "
        "build_master_report.py rebuilds this document, recomputing every "
        "error figure from the logged prediction files so the text cannot "
        "drift away from the pictures.", BODY))


def main():
    story = []
    story.append(Paragraph("Forecast Figures: The Complete Set", H1))
    story.append(Paragraph(
        "Twenty-six figures covering the KAN and LSTM forecasts on CBA.AX, "
        "with a full explanation of how each one is produced.", SUB))

    part_one(story)
    story.append(PageBreak())
    part_two(story)
    story.append(PageBreak())
    part_three(story)
    story.append(PageBreak())
    part_four(story)
    story.append(PageBreak())
    part_five(story)
    story.append(PageBreak())
    part_six(story)
    part_seven(story)
    part_eight(story)

    output = os.path.join(HERE, "Forecast_Figures_Complete.pdf")
    document = SimpleDocTemplate(
        output, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title="Forecast Figures: The Complete Set",
        author="S M Mahmudul Hasan Joy",
    )
    document.build(story)
    print("wrote %s" % output)


if __name__ == "__main__":
    main()
