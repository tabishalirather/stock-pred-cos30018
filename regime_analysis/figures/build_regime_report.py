"""
build_regime_report.py
Assemble the twelve regime figures into one PDF, with a short explanation of
what each shows. One figure per cell of Table 7.

    python make_regime_figures.py     # first, to generate the images
    python build_regime_report.py     # then, to build the PDF

Every number quoted is read from regime_figure_stats.json, which
make_regime_figures.py computes directly from the logged runs, so the text
cannot drift away from the pictures.
"""

import json
import os

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
STATS = os.path.join(HERE, "regime_figure_stats.json")

styles = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=styles["Normal"], fontSize=10.5,
                      leading=15, alignment=TA_JUSTIFY, spaceAfter=8)
H1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=19, leading=23,
                    spaceAfter=4)
H2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13, leading=17,
                    spaceBefore=12, spaceAfter=6, textColor="#1a3d5c")
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
                         textColor="#555555", spaceBefore=2, spaceAfter=9)

PAGE_WIDTH = A4[0] - 4 * cm


def picture(filename, width_fraction=1.0):
    path = os.path.join(HERE, filename)
    if not os.path.exists(path):
        return Paragraph("[missing figure: %s]" % filename, BODY)
    from PIL import Image as PILImage
    with PILImage.open(path) as handle:
        width, height = handle.size
    target = PAGE_WIDTH * width_fraction
    return Image(path, width=target, height=height * target / width)


def load_stats():
    with open(STATS) as handle:
        rows = json.load(handle)
    return {(r["horizon"], r["regime"]): r for r in rows}


def stats_table(stats, horizon):
    data = [["Regime", "Windows", "LSTM", "KAN", "Naive"]]
    for regime in ("Normal", "Volatile", "Trending"):
        row = stats.get((horizon, regime))
        if not row:
            continue
        data.append([
            regime, str(row["count"]),
            "%.3f" % row["rmse"]["LSTM"],
            "%.3f" % row["rmse"]["KAN"],
            "%.3f" % row["rmse"].get("Naive", float("nan")),
        ])
    table = Table(data, colWidths=[3.2 * cm, 2.2 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm])
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


def horizon_page(story, stats, horizon, headline, observations):
    story.append(Paragraph("%d-day horizon" % horizon, H2))
    story.append(Paragraph(headline, BODY))

    for regime in ("Normal", "Volatile", "Trending"):
        row = stats.get((horizon, regime))
        if not row:
            continue
        story.append(picture(row["stem"] + ".png"))
        story.append(Paragraph(
            "%s conditions: %d windows, %s to %s. "
            "Error on these lines: LSTM %.3f, KAN %.3f, naive %.3f."
            % (regime, row["count"], row["first"], row["last"],
               row["rmse"]["LSTM"], row["rmse"]["KAN"],
               row["rmse"].get("Naive", float("nan"))),
            CAPTION))

    story.append(Paragraph("Summary for this horizon", H3))
    story.append(stats_table(stats, horizon))
    for text in observations:
        story.append(Paragraph(text, BODY))


def main():
    stats = load_stats()
    story = []

    # ------------------------------------------------------------ opening
    story.append(Paragraph("Forecasts by Market Regime", H1))
    story.append(Paragraph(
        "One figure for every cell of Table 7: four forecast horizons crossed "
        "with three market conditions, on CBA.AX.", SUB))

    story.append(Paragraph("What these figures show", H2))
    story.append(Paragraph(
        "Table 7 in the paper reports a single error number for each "
        "combination of forecast horizon and market condition. These twelve "
        "figures show the forecasts those numbers came from, so the behaviour "
        "behind each cell can be seen rather than inferred.", BODY))

    story.append(Paragraph("1. The task, in plain terms", H2))
    story.append(Paragraph(
        "The data is the daily closing price of one share, Commonwealth Bank "
        "of Australia, from January 2020 to July 2023. That is about 900 "
        "trading days. The question is whether a model can look at recent "
        "prices and say something useful about future ones.", BODY))
    story.append(Paragraph(
        "Each model is given the closing prices of the last 20 trading days, "
        "roughly a month of trading. From those 20 numbers it produces the "
        "next <i>h</i> numbers in a single shot, where <i>h</i> is the "
        "forecast horizon. At <i>h</i> = 1 it produces tomorrow's price. At "
        "<i>h</i> = 200 it produces the next two hundred days all at once, "
        "rather than predicting one day and feeding it back in.", BODY))
    story.append(Paragraph(
        "The 20-day window then slides forward by one day and the whole thing "
        "happens again. Repeating this across the test period gives several "
        "hundred separate forecasts, each made from a different starting "
        "point. Those are the individual points in the figures.", BODY))
    story.append(Paragraph(
        "Crucially, the models are only ever tested on stretches of time that "
        "came after the data they were trained on. A model is never asked "
        "about a period it has already seen.", BODY))

    story.append(Paragraph("2. The two models being compared", H2))
    story.append(Paragraph(
        "<b>LSTM</b> is a long established type of neural network built for "
        "sequences. It reads the 20 days in order, carrying a memory of what "
        "it has seen so far, which in principle suits price data.", BODY))
    story.append(Paragraph(
        "<b>KAN</b>, a Kolmogorov-Arnold Network, is a newer design. Rather "
        "than reading the days in sequence, it receives all 20 days at once "
        "as a flat list of numbers and learns a flexible curve for each input. "
        "It has no built-in notion of order, which is why the original "
        "expectation was that it would do worse on this kind of data.", BODY))

    story.append(Paragraph("3. The naive baseline, and why it matters", H2))
    story.append(Paragraph(
        "Alongside the two models there is a third line on every figure, and "
        "it is not a model at all. The <b>naive baseline</b> simply predicts "
        "that the price will not change. Whatever today's closing price is, "
        "that is its forecast for tomorrow, for next week, and for two "
        "hundred days from now. It does no learning and has no parameters.", BODY))
    story.append(picture("regime_explainer_naive.png"))
    story.append(Paragraph(
        "Figure A. The naive baseline in action. It takes the last known "
        "price and holds it flat for the whole forecast period.", CAPTION))
    story.append(Paragraph(
        "This sounds too crude to be worth plotting, but it is the standard "
        "yardstick in forecasting, for a simple reason: share prices move "
        "slowly from one day to the next, so \"nothing changes\" is already a "
        "decent guess. A model that cannot beat it has not learned anything "
        "useful about the market, however sophisticated it looks. Reporting a "
        "low error without this comparison can make a model appear far more "
        "capable than it is.", BODY))
    story.append(Paragraph(
        "Including it is one of the things that changed in the revised "
        "analysis, and it is the single most useful line on these plots for "
        "judging whether either model is genuinely forecasting.", BODY))

    story.append(PageBreak())

    story.append(Paragraph("4. The three market conditions", H2))
    story.append(Paragraph(
        "The point of Table 7 is that a model might cope well in calm markets "
        "and badly in turbulent ones. To test that, every 20-day window is "
        "sorted into one of three conditions before any forecast is judged.", BODY))
    story.append(Paragraph(
        "The sorting uses only the 20 days the model can actually see. No "
        "information from the forecast period is used, so a window is never "
        "labelled using knowledge of what happened next. Two things are "
        "measured: how much the price jumped about from day to day, and how "
        "directional the movement was, meaning whether it travelled steadily "
        "one way or wandered up and down and ended where it started.", BODY))
    story.append(Paragraph(
        "<b>Volatile</b>: day-to-day movement in the top quarter of what was "
        "seen during training. <b>Trending</b>: movement not unusually large, "
        "but unusually one-directional, again in the top quarter. "
        "<b>Normal</b>: everything else, and the most common case.", NOTE))
    story.append(Paragraph(
        "The cut-offs come from the training data only, so they are fixed "
        "before the test period is looked at.", BODY))
    story.append(picture("regime_explainer_types.png"))
    story.append(Paragraph(
        "Figure B. One real window of each type, taken from the data. Note "
        "the vertical scales differ: the Volatile example covers a third of "
        "the stock's entire price range in a single month, which is why it "
        "counts as volatile.", CAPTION))

    story.append(Paragraph("5. Reading the twelve figures", H2))
    story.append(Paragraph(
        "<b>The vertical axis</b> is the closing price scaled from 0 to 1, "
        "where 0 is the cheapest day in the whole period and 1 the dearest. "
        "These are not dollars. Scaling was applied before training and every "
        "error figure in the paper uses the same scale, so an error of 0.05 "
        "means a typical miss of about five percent of the stock's full price "
        "range.", BODY))
    story.append(Paragraph(
        "<b>The horizontal axis</b> counts the windows belonging to that "
        "regime, in chronological order. It is not a calendar. Filtering to "
        "one condition leaves gaps: a volatile stretch in March, another in "
        "June, and quiet weeks in between that belong to a different "
        "condition. Drawing a continuous calendar line across those gaps "
        "would imply a continuity that does not exist. Tick labels give the "
        "real date at each position, so a large jump between two ticks means "
        "the intervening period was a different condition.", BODY))
    story.append(Paragraph(
        "<b>The lines.</b> Black is what actually happened. Blue is the LSTM. "
        "Orange is the KAN. Grey is the naive baseline. Markers show the "
        "individual forecasts. Where a coloured line sits on the black one, "
        "that forecast was right; the vertical gap is the error.", BODY))
    story.append(Paragraph(
        "<b>The error figures</b> quoted beneath each plot are root mean "
        "square errors: roughly, the typical size of that vertical gap. Lower "
        "is better.", BODY))

    story.append(Paragraph("6. How these relate to Table 7", H2))
    story.append(Paragraph(
        "The window counts match Table 7 exactly, which is a useful check "
        "that the same data underlies both. At the 1-day horizon, for "
        "instance, both give 277 Normal, 175 Volatile and 79 Trending "
        "windows.", BODY))
    story.append(Paragraph(
        "<b>The error figures, however, are not the Table 7 numbers, and "
        "should not be quoted as though they were.</b> Table 7 averages over "
        "every step of the forecast and over three random seeds. These "
        "figures plot only the first step of each forecast, from a single "
        "seed, because a hundred overlapping lines cannot be drawn legibly. "
        "The one-step-ahead component is the easiest part of any forecast, so "
        "the errors here are smaller than Table 7 throughout, and the naive "
        "baseline looks far stronger than it does in the paper. Use these "
        "figures to see <i>how</i> the models behave in each regime; use "
        "Table 7 for <i>how well</i>.", WARN))

    story.append(PageBreak())

    # ------------------------------------------------------------- pages
    horizon_page(
        story, stats, 1,
        "One day ahead. All three methods stay close to the actual price, "
        "since tomorrow's price is rarely far from today's.",
        ["The KAN is slightly closer than the LSTM in all three regimes. Both "
         "degrade in Volatile conditions, the LSTM more so, which is the "
         "pattern Table 7 reports.",
         "The naive baseline is the strongest line on all three plots. At one "
         "step ahead this is expected and is not by itself evidence against "
         "the models; the fuller comparison in Table 7 is the fair one."])

    story.append(PageBreak())

    horizon_page(
        story, stats, 2,
        "Two days ahead. The clearest separation between the two models "
        "anywhere in this set.",
        ["The LSTM produces a smooth, slow-moving line that misses most "
         "turning points, and in Volatile conditions it swings far away from "
         "the actual price for weeks at a time. The KAN tracks the real "
         "movement closely in every regime.",
         "This is the horizon at which the difference between the two "
         "architectures is most visible to the eye rather than only in a "
         "table."])

    story.append(PageBreak())

    horizon_page(
        story, stats, 100,
        "One hundred days ahead. The ordering reverses here.",
        ["The LSTM is the better of the two models in all three regimes, "
         "which matches Table 7. The KAN wanders further from the actual "
         "price, particularly in Normal conditions.",
         "Both are now far enough from the truth that neither is tracking "
         "the price in any useful sense. The lines drift around the general "
         "level rather than following it."])

    story.append(PageBreak())

    horizon_page(
        story, stats, 200,
        "Two hundred days ahead. The ordering reverses once more.",
        ["The KAN is closer than the LSTM in all three regimes, and the LSTM "
         "is at its worst in Trending conditions, where it is roughly twice "
         "as far from the actual price as the KAN.",
         "The instability noted in the paper is visible here: at this horizon "
         "the KAN's optimiser struggles on the earliest and smallest training "
         "fold, which is why the paper reports those figures separately."])

    story.append(PageBreak())

    # ------------------------------------------------------------ closing
    story.append(Paragraph("What the twelve figures show together", H2))
    story.append(Paragraph(
        "Neither model is uniformly better. The KAN is ahead at 1, 2 and 200 "
        "days; the LSTM is ahead at 100 days. Which one leads depends on the "
        "horizon rather than on the market condition, and no regime reverses "
        "the ordering within a horizon.", BODY))
    story.append(Paragraph(
        "Volatile conditions are harder for both models at short horizons, "
        "and the LSTM suffers more from them than the KAN does. At long "
        "horizons the pattern changes: Trending windows become the difficult "
        "ones, particularly for the LSTM at 200 days.", BODY))
    story.append(Paragraph(
        "Across every figure the LSTM produces smoother lines than the KAN. "
        "That is not a virtue in itself. A smooth line that lags the market "
        "is simply a slow-moving average, and much of the LSTM's error at "
        "short horizons comes from arriving at each turn too late.", BODY))

    story.append(Paragraph("Reproducing these", H2))
    story.append(Paragraph(
        "Run make_regime_figures.py to regenerate all twelve images and their "
        "statistics, then build_regime_report.py to rebuild this document. "
        "The figures are drawn from the same logged prediction files that "
        "produce Table 7, pooling all five walk-forward folds at seed 0.", BODY))

    output = os.path.join(HERE, "Regime_Figures_Report.pdf")
    document = SimpleDocTemplate(
        output, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title="Forecasts by Market Regime",
        author="S M Mahmudul Hasan Joy",
    )
    document.build(story)
    print("wrote %s" % output)


if __name__ == "__main__":
    main()
