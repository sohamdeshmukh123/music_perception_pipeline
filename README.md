# Music Perception Pipeline

This project asks a simple question: **can you predict whether people find a short
melody pleasant just by measuring how "predictable" it is?**

Nine short 7–10 note clips were built to test different musical ideas (repetition,
symmetry, a scale run, simple vs. complex harmony, and pure randomness). 31 people
listened to all nine and rated each one on how pleasant, stable, and restless it felt.
This code turns those ratings, plus some math done on the notes themselves, into a
small statistical model — and checks, carefully, whether that model actually holds up
or is just a fluke of a small dataset.

The short answer it finds: **one specific measurement — how unpredictable each note is
given the note before it ("transition entropy") — predicts pleasantness well, and it's
the only one of five candidate measurements that survives rigorous testing.**

## What's in this folder

- **Raw data**: `Music_Perception_Study__Responses__-_Form_Responses_1.csv` — the
  original survey export.
- **Python scripts**: do the actual work (data cleaning → feature math → statistics →
  charts). See "How the pipeline fits together" below.
- **Generated CSVs, `.txt`, and `.png` files**: everything the scripts produce. They're
  already included so you can look at results without running anything — but they'll
  be regenerated fresh if you run the pipeline yourself.

## Setup

You need Python 3 installed. Then, from inside this folder:

```bash
pip install -r requirements.txt
```

## How to run it

```bash
python3 run_pipeline.py
```

That's it — this one command runs every step in order and rewrites every CSV, text
file, and chart in the folder from scratch, starting only from the raw survey CSV.

There's also a separate small tool you can play with:

```bash
python3 interactive_tool.py
```

Type in your own note sequence (e.g. `C D E F G`) and it'll compute the same
measurements used in the study and predict how pleasant a listener would likely rate it.

## How the pipeline fits together

Each script is one stage. They run in this order inside `run_pipeline.py`:

| Script | What it does |
|---|---|
| `load_data.py` | Cleans the raw survey export into a tidy table of ratings |
| `compute_features.py` | Computes number-based measurements for each of the 9 clips (entropy, complexity, symmetry, etc.) |
| `eda_and_inference.py` | Basic statistics — averages, and tests confirming the ratings aren't just noise |
| `modeling.py` | Fits the main prediction model and demonstrates what overfitting looks like |
| `deep_validation.py` | Stress-tests the model: permutation test, bootstrapping, comparing every measurement head-to-head, and two alternative modeling approaches |
| `visualisation.py` | Generates all the charts (`figure1...png` through `figure9...png`) |
| `interactive_tool.py` | Standalone calculator — not part of the automated pipeline, run it separately |
| `utils.py` | Shared helper functions the other scripts import |


## Try it in your browser

`calculator.html` is a standalone, no-install version of `interactive_tool.py` — a
webpage instead of a command-line tool. Open it directly in any browser (just
double-click the file, no server needed) and type in a note sequence to see the
model's predicted pleasantness live, along with the underlying measurements
(entropy, complexity, symmetry). It only accepts real note names (C, C#, D, D#, E,
F, F#, G, G#, A, A#, B, or their flat equivalents) and will warn you if you type
anything else.


## A note on rigor

A big part of this project is *not* trusting the first result. A model fit on only 9
data points can look great and still be meaningless — so instead of reporting one
number, the pipeline checks the finding several different ways: predicting each clip
as if it were unseen data (leave-one-out cross-validation), an exact statistical test
rather than an approximation, resampling the data thousands of times to see how stable
the result is, comparing every candidate measurement against each other rather than
just picking the best-looking one, and trying two more flexible modeling approaches to
confirm a simple one isn't missing something. Details on exactly what each check found,
and which numbers are exact vs. approximate reproductions of the original paper, are in
`deep_validation_results.txt` and the code comments in `deep_validation.py`.

## Limitations

- Only 9 clips total, so results are suggestive, not conclusive.
- The entropy measurement is computed from very short sequences (7–10 notes), which
  mechanically pushes it toward zero for very short or repetitive clips.
- Most raters were 15–18 years old (27 of 31), so results may not generalize to other
  age groups.
- Each clip was rated once per person on each scale, and clips were shown in the same
  order to everyone, so fatigue/order effects aren't ruled out.
