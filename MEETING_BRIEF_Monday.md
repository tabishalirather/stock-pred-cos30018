# Meeting brief: Monday 11.00 am, with Nadia

Plain-language background for the discussion, plus the questions Nadia is
likely to ask and how to answer them.

---

## Part 1. What happened, in plain terms

### The starting point

Nadia asked why Table 7 said "not tested" in some cells. That looks like a
formatting question. Answering it properly meant defining what "Normal",
"Volatile" and "Trending" market conditions actually were, and that is where
things unravelled.

### First problem: the table did not agree with the rest of the paper

Think of it like a report where the summary page does not match the detailed
pages behind it.

- The detailed LSTM tables said the error was between 0.074 and 0.118.
  Table 7 said 0.039 to 0.085.
- The detailed KAN table said 0.152 to 0.331. Table 7 said 0.385 to 0.670.
- The text of Section 5.5 said KAN's best result was 0.152. Table 7 said 0.385
  for the same thing.
- "Normal", "Volatile" and "Trending" appeared nowhere else in the paper. No
  definition, no thresholds, no method for splitting the data.

So Table 7 could not be traced back to any experiment described in the paper.

### Second problem: a one-line bug in the KAN code

This is the important one.

When you test a forecasting model you compare each prediction to the matching
real value. Prediction for Monday against Monday's actual price, Tuesday
against Tuesday, and so on.

Because of one extra instruction in the code (`unsqueeze(1)`), the comparison
was set up wrongly. Instead of comparing each prediction to its own day, the
computer compared **every prediction to every day**. Monday's forecast was
scored against Monday, Tuesday, Wednesday, and every other day in the set.

The consequence: the "error" being reported was not how wrong the forecast was.
It was roughly how much the share price varies across the whole period. That is
a much bigger number, and it does not change much no matter how good the model
is.

Demonstration on our own data: a model whose true error is 0.029 reports 0.354
through that code path. That is about twelve times too large, and 0.354 sits
right inside the range we published for KAN.

**In short, KAN was never as bad as we reported. The code was scoring it
against the wrong thing.**

### Third problem: the two models were not marked on the same exam

The KAN script kept only the last few rows of the test data before scoring:
at the 1-day horizon, that is **one single day**. The LSTM was scored on the
entire test set, several hundred days.

So one model was judged on hundreds of days, the other on one. Whatever that
one day happened to be determined the KAN result. Re-running it gives 0.014,
0.024, 0.037 and so on, purely depending on which day was drawn.

### Fourth problem: the training and testing data overlapped

Each input to the model is a 20-day window of prices. Monday-to-Friday-week-4
and Tuesday-to-Monday-week-5 share 19 of those 20 days. They are almost the
same thing.

The code shuffled all these windows randomly and split them into a training
set and a test set. Because neighbouring windows are nearly identical, the
model was effectively being tested on data it had already seen. That makes
results look better than they are, for both models.

### What was rebuilt

The experiments were re-run with all of this corrected:

- **Time-ordered splitting.** Train on earlier data, test on later data, five
  times over. Never test on the past.
- **Same exam for both.** Both models trained on the same data and scored on
  exactly the same days.
- **Market conditions properly defined.** Each 20-day window is classified as
  Volatile, Trending or Normal using two standard measures, with the cut-offs
  calculated only from training data.
- **Fixed seeds.** Three runs each, so we can report how much results wobble.
- **A "do nothing" benchmark added.** Simply predict that tomorrow's price
  equals today's. Any real model should beat this.
- **Real pykan used**, the same package as the original work.

### What the corrected results show

| Horizon | LSTM | KAN | Do-nothing benchmark |
|---|---|---|---|
| 1 day | 0.068 | 0.041 | **0.020** |
| 2 day | 0.072 | 0.047 | **0.025** |
| 100 day | **0.110** | 0.125 | 0.111 |
| 200 day | 0.153 | **0.152** | 0.132 |

Lower is better.

1. **The 6.5 to 10 times LSTM advantage was an artifact of the bug.** The two
   models are broadly comparable, and KAN is somewhat better at short horizons.
2. **Neither model beats the do-nothing benchmark at short horizons**, by a
   factor of about three. This is the uncomfortable finding, and it is the one
   most likely to interest a reviewer.
3. **KAN does have a real weakness, but a different one.** On the smallest
   training fold at long horizons its optimiser becomes unstable and produces
   nonsense values. That is a training-stability problem, not an accuracy one.

---

## Part 2. Nadia's concerns, and how to answer them

### "It looks like KAN is better. Our conclusion was the opposite."

She has read it correctly. Confirm that plainly rather than softening it. The
original conclusion rested on a comparison that was set up incorrectly. This is
not a matter of interpretation or of a different analysis choice; the old
number was measuring the wrong quantity.

### "The cover letter is no longer relevant."

She is right, and this is a good sign that she has grasped the scale of it.
The cover letter sells a finding the paper no longer makes.

### "Sometimes the suggestions from AI are not reasonable, and they remove our
personality."

A fair concern, and worth separating into two parts:

- **The wording** is entirely hers to change. Nothing about the prose needs to
  stay as it is.
- **The numbers** come from re-run experiments, not from a writing suggestion.
  Those should not be edited by hand. If any of them need to change, they get
  regenerated from the code.

### "I really prefer to submit as soon as possible."

This is the real tension in the meeting. She wants it out; the paper's central
claim has just inverted. Useful framing: submitting the old version now means
submitting a result we know to be wrong, and a reviewer who opens the code will
find the same thing. Two weeks of delay is much cheaper than a retraction or a
rejection on those grounds.

---

## Part 3. Questions she is likely to ask

**"How do we know the new numbers are right, and the old ones wrong?"**

Four independent checks:
- Running the *old* method with the new code reproduces the *old* LSTM numbers
  closely. So the code behaves like the original, and the difference comes from
  the fixes rather than from a different implementation.
- Two separate KAN implementations were used and they agree.
- Configurations 2 and 5 in Table 6 are the same specification. The old table
  scored them differently (0.188 and 0.152), which should have been impossible.
  They now agree.
- Every number in the tables was checked back against the saved per-run output
  files.

**"Could the bug have been in the new code rather than the old?"**

The bug is visible in the original file without running anything. The label
tensor has one more dimension than the model output. Tabish or Federico can
confirm it in a few minutes.

**"Does this mean our earlier work was worthless?"**

No. The methodology, the literature review, the architecture descriptions and
the experimental design all stand. One evaluation step was wrong, which
affected the comparison. That is a correction, not a collapse.

**"Why did nobody notice?"**

The bug does not crash anything. It produces plausible-looking numbers. The
KAN results looked poor, which matched the expectation that a general-purpose
architecture would struggle on sequential data, so the numbers were not
questioned. This is a common failure mode and not a sign of carelessness.

**"Do we have to publish the finding that neither model beats persistence?"**

It is the honest result and it is arguably the most interesting one. Many
published forecasting papers omit a naive baseline, which is precisely why
reviewers increasingly demand it. Including it makes the paper harder to
attack, not easier.

**"Is it the same paper or a different one?"**

This is the decision to make on Monday.

- *Same paper, corrected results.* Faster. The framing shifts from "LSTM wins"
  to "the two are comparable, and neither beats a naive baseline."
- *Reframed paper.* Built around evaluation methodology: how easily a small
  evaluation error can manufacture an order-of-magnitude result. Stronger
  contribution, more rewriting.

**"What about the arXiv version?"**

It still carries the original claim. The agreement was to update it once the
journal submission is settled.

---

## Part 4. Things to be upfront about

Do not let these surface from her side first.

1. **Scaling.** The stored data was scaled using the whole period, including
   the test portion. This was true of the original work as well, it affects
   both models equally, and it is stated in the limitations. But absolute
   values carry that caveat.
2. **LSTM settings were not re-tuned.** The original configurations were kept.
   A better-tuned LSTM might narrow the gap, so the honest claim is "as
   configured here", not "KAN is better than LSTM".
3. **One fold excluded at long horizons.** Where pykan's optimiser diverged,
   that fold is reported separately rather than silently dropped.
4. **One stock, three runs.** Everything is CBA.AX. The standard deviations are
   rough. Running a second ticker would strengthen this and is a few hours of
   compute.
5. **Independent reproduction has not happened yet.** Tabish re-running the
   grid from the repository would be the single strongest thing to have before
   submission.

---

## Part 5. Repository status

- `v0.2/KANs/kan_main.py` — **both bugs now fixed** (6 August), with comments
  explaining what was wrong. The shuffled split is left in place, with a note,
  so the script still reproduces the historical runs.
- `regime_analysis/` — the corrected pipeline, all logged runs, generated
  tables and figures, and `FINDINGS.md` with the full technical write-up.
- `regime_analysis/figures/` — 26 forecast figures and three PDF reports.
- Overleaf — updated and compiling; changes are tracked so they can be reviewed
  or rejected individually.
- GitHub — pull request #1 open on Tabish's repository with the fix and the
  corrected pipeline.

---

## Part 6. Definitions, for answering on the spot

Plain-language answers to terms that will come up. Written to be said aloud
rather than read.

### "What is blocked walk-forward validation?"

It is how we decide which days a model learns from and which days it is tested
on. Three words, three ideas.

**Forward** — you only ever test on the future. Train on an earlier stretch,
test on a later one. Training on 2022 and testing on 2021 would tell us
nothing useful, because in real use you never have tomorrow's data when making
today's forecast.

**Walk** — you repeat this, moving along. Testing on one stretch only tells you
how the model did in that period. If that stretch was calm, you learn nothing
about turbulent markets. So we do it five times:

| Fold | Trains on | Tests on |
|---|---|---|
| 1 | the first 40 percent of the data | the block right after it |
| 2 | everything before block 2 | the next block |
| 3 | everything before block 3 | the next block |
| 4 | everything before block 4 | the next block |
| 5 | everything before block 5 | the final block |

The training set grows each time, mirroring reality: a forecaster in mid-2022
has more history than one in 2021. Pooling the five test blocks covers roughly
the last 60 percent of the period, calm stretches and turbulent ones alike.

**Blocked** — the test days stay in one contiguous lump. This is the part that
matters most and where the original analysis went wrong. The alternative is to
shuffle all the windows and deal them randomly into training and test piles,
which is standard for most machine learning and exactly wrong here. Each input
is a 20-day window; the window starting Monday and the one starting Tuesday
share 19 of their 20 days. Shuffle them and one lands in training while its
near-twin lands in testing, so the model is marked on material it has
effectively already seen.

*In one sentence:* train on the past, test on the future that follows it, keep
the test days contiguous, repeat five times marching forward.

*Why it matters here:* this is the change that made **both** models look worse
than before, not just the KAN. If Nadia asks why the LSTM's error rose from
about 0.046 to 0.068, this is the answer. The earlier figure was measuring
performance on partly familiar data.

### "What exactly didn't add up in the old Table 7?"

Four separate problems, different in kind. Only two are arithmetic.

**1. Its numbers sat outside the range of the tables it summarised.** Tables 4
and 5 list every LSTM run: best 0.0745, worst 0.1183. Table 7 reported 0.039 to
0.085 for the same models on the same data. Numbers below 0.0745 cannot have
come from those runs. Same for the KAN: Table 6 spans 0.152 to 0.331, Table 7
reported 0.385 to 0.670, every one above the worst run.

*Analogy if useful:* if a class's individual scores run from 60 to 85, a
summary claiming the average was 92 is not a rounding disagreement. The number
came from somewhere else.

**2. It contradicted the paper's own prose.** Section 5.5 states the best KAN
result was 0.152 at the 1-day horizon. Table 7 gave 0.385 at that horizon. Both
cannot be true, and finding it needs no code, only turning two pages.

**3. The market conditions were never defined.** Normal, Volatile and Trending
appear nowhere else in the paper. No definition, no thresholds, no method, no
mention in the methodology. This is not arithmetic at all: a third of the
table's structure described an experiment the paper never says was performed.

**4. The 1-day figures track repeat runs, not market conditions.** The logs for
the 100-unit linear LSTM contain 0.0385, 0.0400, 0.0409, 0.0426, 0.0430,
0.0466. Table 5 summarises these as mean 0.0459, best 0.0385. Table 7's 1-day
row reads Normal 0.039, Volatile 0.045, Trending 0.042, which tracks the run
statistics rather than any segmentation.

*Be careful with number 4.* It is an inference from the numbers, not something
provable. Raise it as a question, not an assertion.

*The common thread:* items 1 to 3 could all have been found by a reviewer with
only the PDF. The code bugs were a separate discovery that came afterwards.

### "How does Table 7 have values now when it did not before?"

Because the experiment behind it did not exist, so it had to be run.

There was nothing to recompute. There was a table shape with numbers in it and
no procedure that would produce those numbers. So the new values come from work
done fresh, in this order:

1. **Defined the market conditions** — two measurable properties of each 20-day
   window, with cut-offs at the 75th percentile of the training data. Written
   down before anything was measured, so they could not be tuned to give a
   flattering answer.
2. **Fixed the two KAN bugs**, so both models were scored on the same data.
3. **Rebuilt the evaluation** — walk-forward, identical test windows, three
   fixed seeds, naive baseline.
4. **Re-ran everything** — 150 runs for the main grid, 60 more with real pykan,
   90 more for the Table 6 configurations. Every run writes its predictions to
   disk.
5. **Generated the table from those files.** The LaTeX is produced by a script
   that reads the saved predictions. Nobody typed a number into it.

*The point worth making:* the old table was hand-written, which is how it drifted
from the runs it claimed to summarise. The new one cannot drift. All 80 cells
were verified against the logged files.

*Honest framing:* it is not that Table 7's numbers were wrong and have been
fixed. Table 7 previously reported an analysis nobody had performed, and now it
reports one that was.

### "Are Tables 4, 5, 6 and 7 consistent now?"

**No, and this is the main outstanding issue.** Answer it straight.

| Table | Contents | Protocol |
|---|---|---|
| 4 | LSTM configs, single runs (0.0745–0.1183) | **Old**: shuffled split |
| 5 | LSTM multi-run stats (mean 0.0459, min 0.0385) | **Old**: shuffled, no fixed seed |
| 6 | KAN configs (0.0161–0.0750) | **New**: walk-forward, 3 seeds |
| 7 | Regime comparison | **New**: walk-forward, 3 seeds |

Two problems follow. Tables 4 and 5 screen the LSTM under the old protocol
while Table 6 screens the KAN under the new one, so the two screening exercises
sit side by side and cannot be compared. Worse, Table 5's numbers are *lower*
than Table 7's LSTM figures (0.0385 against 0.068 at 1 day), because the
shuffled split flattered them. The paper still contains a table implying the
LSTM does better than the corrected analysis says.

The methodology section does state that Tables 4 and 5 use a preliminary
shuffled split, so the paper is not contradicting itself. But it asks the reader
to hold two incompatible measurement systems at once and invites the wrong
comparison.

**RESOLVED (10 August).** Option 1 was carried out. Tables 4 and 5 were re-run
under the walk-forward protocol (nine configurations, three seeds, five folds,
120 new runs) and the paper now has every table on one protocol. Updated
answer: **yes, Tables 4 to 7 are now consistent** — same split, same seeds,
same pooling, and all four generated or verified against logged runs.

One new finding came out of it, worth mentioning to Nadia: **the configuration
ranking changed.** Under the corrected protocol the shallow 2-layer tanh
networks are the best LSTM configurations (test RMSE 0.054), and the deep
100-unit stacks are the worst and least stable (the 6-layer, 100-unit variant
reaches 0.112 with the largest seed spread). The old narrative said deeper
networks underperform, and that survives; but the old "best" figures (0.0385
for 100-unit linear) do not, because they were flattered by the shuffled
split. Table 4's caption notes one small gap: training error for the 4-layer
100-unit configuration was not logged, since those runs are shared with the
Table 7 grid.

### "Does KAN still beat LSTM at short horizons?" (updated 10 August)

Yes, and this was checked against the best LSTM, not just the inherited one.

The new Table 4 screening found a better LSTM configuration (2 layers, 20
units, tanh) than the ones the comparison had inherited. To make sure the
KAN's short-horizon edge was not an artifact of comparing against a weak
LSTM, that configuration was run at every horizon:

| Horizon | Best LSTM (L2-20-tanh) | KAN | Naive |
|---|---|---|---|
| 1 day | 0.0537 ± 0.0003 | **0.0409 ± 0.0059** | 0.0201 |
| 2 day | 0.0528 ± 0.0032 | **0.0473 ± 0.0053** | 0.0252 |
| 100 day | **0.1144 ± 0.0118** | 0.1250 (stable folds) | 0.1109 |
| 200 day | **0.1406 ± 0.0077** | 0.1516 (stable folds) | 0.1323 |

Three things to say if asked:

1. **The KAN's short-horizon edge survives, narrowed.** At 1 day it is about
   two standard deviations; at 2 days about one. "KAN somewhat ahead" is
   defensible; "KAN clearly outperforms" is not.
2. **The picture is now cleaner at long horizons.** The best LSTM beats the
   KAN's stable-fold figures at both 100 and 200 days. So the story is:
   KAN ahead short, LSTM ahead long, all margins modest.
3. **The naive baseline still wins at 1 and 2 days** against every
   configuration tested, which remains the headline.

These runs are in `results/preds_lstmgrid/` alongside the screening. The
paper's Table 7 still uses the inherited configurations, which is disclosed;
whether to switch Table 7 to the best configuration is a fair question for
the meeting, and either way this cross-check should be mentioned in the text.

### "What does naive mean?"

A forecast that assumes nothing changes. Whatever the closing price is today,
that is its prediction for tomorrow, for next week, and for two hundred days
out. A flat line forward from the last known price. No learning, no parameters,
no training time. Also called a persistence forecast.

**Why something so crude is on every chart.** It is the yardstick. Share prices
move slowly day to day, so "nothing changes" is already a decent guess and
surprisingly hard to beat at short range. If a neural network cannot do better
than assuming the price stays put, whatever it learned is not worth the
computation.

**What it does to our results.** At one day ahead the baseline scores 0.020
against the LSTM's 0.068 and the KAN's 0.041. The thing with no parameters beats
both by roughly a factor of three, and the same holds at two days. This is why
the paper can no longer recommend either model for short-term forecasting: not
because one lost to the other, but because both lost to doing nothing.

**The one exception**, from the distance study: at 100 days ahead the LSTM
reaches 0.099 against the baseline's 0.123. Over that span a fixed guess drifts
badly out of date while a near-constant prediction does not. It is the single
place in the study where a model clearly wins, and it is not currently in the
paper.

**Why it was not there before.** It was not run. Omitting a naive baseline is a
common enough oversight that reviewers increasingly ask for one specifically.
Including it makes the paper harder to attack, not easier.

### "What does the Windows column in Table 7 mean?"

How many forecasts went into that row.

Each forecast starts from a 20-day stretch of prices, the month of trading the
model sees before predicting. That stretch is a window. The window slides
forward one day and the model forecasts again, so counting windows is the same
as counting forecasts.

Once every window is sorted into Normal, Volatile or Trending, the column
reports how many landed in each group. At the 1-day horizon: 277 Normal, 175
Volatile, 79 Trending. So the Volatile row's error of 0.085 came from 175
separate forecasts, not one.

**Three reasons it earns its place.** It shows how much weight to give each row
— Trending has 79 windows against Normal's 277, so it is the shakier figure.
The three add to the full test set (277 + 175 + 79 = 531), confirming every
window was classified once. And it is a traceability check: the counts are
computed, not typed, and they match the figures exactly. The old Table 7 had no
such column because there was no segmentation to count.

**Why the counts shrink with horizon** (531 at 1 day, 472 at 100, 412 at 200):
a forecast needs a real future to check against, so a 200-day forecast cannot be
made from the last 200 days of data. The longer the horizon, the more windows
fall off the end.
