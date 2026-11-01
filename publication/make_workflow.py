"""Draw the executed methodology, separating annotation quality from task risk."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Serif", "svg.fonttype": "none", "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(6.3, 5.5))
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.axis("off")


def box(x, y, width, height, text, fill="#f4f7fa", size=8.3, bold=False, dashed=False):
    patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.005,rounding_size=0.009",
                          lw=.8, edgecolor="#3e5264", facecolor=fill,
                          linestyle="--" if dashed else "-")
    ax.add_patch(patch)
    ax.text(x+width/2, y+height/2, text, ha="center", va="center", fontsize=size,
            fontweight="bold" if bold else "normal", linespacing=1.3)


def arrow(start, end, dashed=False, color="#3e5264"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                                lw=.85, color=color, linestyle="--" if dashed else "-"))


box(.025, .925, .95, .062,
    "Local data audit → development cohort\n3,184 paired ECG–PPG records / 810 participants", bold=True)
box(.025, .825, .95, .068,
    "Participant-disjoint roles per outer fold: 432 train / 108 validation / 270 outer\n"
    "3 outer folds × seeds 17, 42, 73; historical test participants excluded", size=8.1)
arrow((.5, .925), (.5, .897))
box(.025, .730, .95, .063,
    "Raw-waveform SQIs → ECG/PPG filtering → resample 500 to 125 Hz\n"
    "Per-record normalization; full 24 s input (2 × 3,000 samples)", size=8.3)
arrow((.5, .825), (.5, .797))

# The two panels are experiments on the same cohort, not two serial inference stages.
box(.025, .240, .455, .448, "", "#edf4fb")
box(.520, .240, .455, .448, "", "#edf7f4")
ax.text(.2525, .662, "Stage I: matched feature fusion", ha="center", va="center", fontsize=9, fontweight="bold")
ax.text(.7475, .662, "Stage II: task-reliability fusion", ha="center", va="center", fontsize=9, fontweight="bold")
arrow((.2525, .730), (.2525, .693))
arrow((.7475, .730), (.7475, .693))

box(.046, .563, .199, .066, "ECG encoder\n32-D embedding", "white", size=8.3)
box(.261, .563, .199, .066, "PPG encoder\n32-D embedding", "white", size=8.3)
box(.046, .452, .414, .077,
    "Quality heads: detached features + SQIs\nManual goodness targets: training only", "#fff5e8", size=8.1)
arrow((.145, .563), (.190, .534))
arrow((.360, .563), (.315, .534))
box(.046, .341, .414, .077,
    "Uniform / estimated-quality feature weights\nConcat + product + |difference| → BP head\nSBP and DBP outputs", "white", size=8.0)
arrow((.2525, .452), (.2525, .423))
ax.text(.2525, .285, "A/B: clean; C/D: augmented\nUnimodal and dense controls", ha="center", va="center", fontsize=8.0)
arrow((.2525, .341), (.2525, .315))

box(.541, .563, .414, .066,
    "Frozen ECG-only / PPG-only experts\nReuse clean unimodal controls from Stage I", "white", size=8.0)
box(.541, .452, .414, .077,
    "24 waveform-derived / expert features\n4 Ridge outputs: modality × BP target\nEstimated absolute errors (mmHg)", "white", size=8.0)
arrow((.7475, .563), (.7475, .534))
box(.541, .341, .414, .077,
    "Availability-masked inverse-risk weights\nTarget-specific convex prediction fusion\nSBP and DBP outputs", "white", size=8.0)
arrow((.7475, .452), (.7475, .423))
box(.541, .254, .414, .055,
    "Risk-head fitting: training-only cross-fit\nHeld-out teacher residuals; fixed Ridge α = 10", "#fff5e8", size=7.7, dashed=True)
# A side route identifies supervision, rather than inference through a residual target.
ax.plot([.955, .965, .965], [.282, .282, .490], color="#ae7033", lw=.8, ls="--")
arrow((.965, .490), (.954, .490), dashed=True, color="#ae7033")

box(.025, .105, .95, .082,
    "Outer-fold evaluation: clean, quality strata, controlled noise and missing modalities\n"
    "BP errors / agreement; participant bootstrap; parameter, MAC, CPU and memory costs", size=8.1)
arrow((.2525, .240), (.2525, .191))
arrow((.7475, .240), (.7475, .191))
box(.025, .032, .95, .043,
    "Deployable inference: waveform-derived inputs only; no manual quality labels or true BP", "#fff5e8", size=8.0)
ax.text(.5, .006, "Development analysis; not independent external or clinical validation.",
        ha="center", va="center", fontsize=7.7, color="#354656")

for extension in ["png", "pdf", "svg"]:
    fig.savefig(OUT / ("methodology_architecture_detailed."+extension), dpi=400, facecolor="white")
plt.close(fig)

# A compact overview fits the journal's main text; the detailed panel remains a sidecar.
fig, ax = plt.subplots(figsize=(6.3, 2.7))
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.axis("off")
for x, title, body in [
    (.025, "Local data audit", "ECG + PPG: 24 s, 500 Hz\n810 subjects / 3,184 records"),
    (.360, "Participant-level splits", "432 train / 108 validation\n270 outer; 3 folds × 3 seeds"),
    (.705, "Preprocessing + SQIs", "Raw SQIs; ECG/PPG filters\nPer-record z-score; 125 Hz\nInput: 2 × 3,000 samples"),
]:
    box(x, .742, .270, .242, "")
    ax.text(x+.135, .932, title, ha="center", va="center", fontsize=8.3, fontweight="bold")
    ax.text(x+.135, .832, body, ha="center", va="center", fontsize=7.7, linespacing=1.25)
arrow((.300, .863), (.354, .863))
arrow((.635, .863), (.699, .863))
# The preprocessing output feeds both experiments on the identical participant roles.
ax.plot([.840, .840, .2525], [.742, .685, .685], color="#3e5264", lw=.85)
arrow((.2525, .685), (.2525, .641))
arrow((.7475, .685), (.7475, .641))
box(.025, .265, .455, .367, "", "#edf4fb")
box(.520, .265, .455, .367, "", "#edf7f4")
ax.text(.2525, .586, "Stage I: matched feature fusion", ha="center", va="center", fontsize=8.7, fontweight="bold")
ax.text(.7475, .586, "Stage II: task-specific late fusion", ha="center", va="center", fontsize=8.7, fontweight="bold")
ax.text(.2525, .425,
        "Separate ECG/PPG branches: 32-D each\n"
        "8 ordered bins + global summaries\n"
        "No explicit beat-delay / PAT feature\n"
        "Uniform / quality-weighted feature fusion\n"
        "Matched A–D: 36,164 stored parameters",
        ha="center", va="center", fontsize=8.0, linespacing=1.3)
ax.text(.7475, .425,
        "Frozen clean unimodal ECG/PPG experts\n"
        "24 features → 4 Ridge risk outputs\n"
        "Training-only cross-fitted error targets\n"
        "Availability-masked late BP fusion\n"
        "Deployment: 34,314 stored parameters",
        ha="center", va="center", fontsize=8.0, linespacing=1.3)
arrow((.2525, .265), (.2525, .214))
arrow((.7475, .265), (.7475, .214))
box(.025, .025, .95, .179,
    "Outer-fold evaluation: BP errors, quality/noise tests, participant bootstrap, and CPU costs\n"
    "Automatic inference: waveform-derived inputs; no manual quality labels or true BP\n"
    "Previously analyzed development cohort; external and clinical validity unestablished",
    size=7.8, fill="#fff5e8")
for extension in ["png", "pdf", "svg"]:
    fig.savefig(OUT / ("methodology_architecture."+extension), dpi=400, facecolor="white")
plt.close(fig)
print("Saved compact main-text and detailed sidecar flowcharts as PNG (400 dpi), PDF, and SVG.")
