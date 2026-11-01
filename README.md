# Quality-Aware Lightweight Multimodal BP Estimation

Research implementation for **Evaluating Quality-Aware Lightweight Multimodal
Deep Learning for Cuffless Blood Pressure Estimation**. It compares matched ECG,
PPG, feature-fusion and task-risk late-fusion estimators of record-level cuff SBP
and DBP. This is technical development research, not a validated medical device.

The release contains the original Python model/training/evaluation implementation,
configurations, tests, plotting code, a training-free split initializer, and an
auditable parameter-count tool. Existing model/training source files are unchanged
from the completed reliability study. The split initializer and parameter audit
are release additions; their tests do not create research observations.

## Data and environment

Obtain authorized access to [CardioWave-Portable 1.0.0](https://physionet.org/content/cardiowave-portable/1.0.0/)
separately. Set `CARDIO_DATA` to an existing local release directory containing
`metadata/records.csv`, `subject-info.csv`, and `waveforms/`. The code never logs
into PhysioNet or downloads the dataset.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
# requirements.lock.txt records the environment used for the executed study.
# Choose a compatible PyTorch build for your CPU/GPU when recreating that environment.
export CARDIO_DATA=/absolute/path/to/cardiowave-portable/1.0.0
```

The manuscript's local audit excluded missing files and ambiguous duplicated
waveforms, leaving 3,742 records / 953 participants. Its development cohort had
3,184 records / 810 participants after retiring 143 historical test participants.
A complete or changed copy of the dataset can produce a different audit cohort.
Preserve local audit fingerprints and participant roles; do not label a different
cohort an exact replication. Digital/normalized-display amplitudes are not
calibrated physiological amplitudes. BP references follow the 24-second acquisition.

## Run the published protocol

Initialize the historical roles using the original seed, without training the
historical suite. The role file contains participant IDs and stays local.

```bash
python -m scripts.prepare_development_split --data "$CARDIO_DATA"

# Smoke tests: subsets of development participants, two epochs, one seed.
# These are pipeline checks and must not be reported as final research results.
python -m src.factorial --data "$CARDIO_DATA" --quick
python -m src.reliability --data "$CARDIO_DATA" --quick

# Full executed design: three participant-disjoint outer folds and three seeds.
# The explicit CPU profile matches the device choice of the reported study.
python -m src.factorial --data "$CARDIO_DATA" --config configs/factorial_cpu.json
python -m src.reliability --data "$CARDIO_DATA"
```

Stage I fits seven models × three folds × three seeds. It isolates uniform/gated
fusion and clean/augmented BP training with identical quality heads and capacity
across the four factorial cells. Stage II reuses the clean unimodal experts and
fits risk targets only from additional subject-disjoint teacher cross-fits.
Manual annotations supervise quality learning on training data; inference uses
waveform-derived inputs. Missing sensors receive zero weight; both missing causes
abstention. The historical test participants remain excluded from these studies.

Results, checkpoints, validation thresholds, source/configuration snapshots,
participant roles, and per-record predictions are generated under `results/` and
are ignored by Git. The full studies are sequential: reliability requires the
matching completed factorial reference. Existing output locks reject incompatible
code/configuration/data; use a fresh clone or new output configuration for changed
experiments. On CPU the full studies are substantial workloads; no training-time
claim is made by the smoke tests.

```bash
python main.py predict --data "$CARDIO_DATA" \
  --checkpoint results/reliability/fits/fold_0/seed_17/adaptive_risk_fusion.pt \
  --record waveforms/part-0001/R000001
python publication/make_figures.py
python publication/make_workflow.py
```

The record path above is a dataset-format example, not an included record.
The plot scripts read locally generated outputs; no patient data are bundled here.

## Parameter accounting and tests

```bash
python -m src.parameter_audit
python -m pytest -q
# Optional check of completed, trusted local checkpoints:
python -m src.parameter_audit --checkpoints-root results
```

The count is `sum(p.numel() for p in model.parameters())`, including frozen
parameters and retained heads even when a control policy does not use them.
Buffers, preprocessing, parameter-free operations, and replicated cross-fit
teachers are excluded. Risk-fusion experts and the exported Ridge layer are frozen
at inference; 34,314 stored parameters does **not** mean 34,314 parameters are
trained jointly or require gradients during inference.

| Model | Encoders | Quality heads | BP heads | Risk layer | Stored total |
|---|---:|---:|---:|---:|---:|
| ECG-only or PPG-only | 15,328 | 657 | 1,122 | 0 | 17,107 |
| Feature fusion A–D | 30,656 | 1,314 | 4,194 | 0 | 36,164 |
| Dense feature control | 44,944 | 1,314 | 4,194 | 0 | 50,452 |
| Late fusion, all policies | 30,656 | 1,314 | 2,244 | 100 | 34,314 |

`reports/` records the executed release checks. Tests requiring completed private
experiment outputs skip when those outputs are absent; they are not represented
as newly executed full experiments. Unit-test signals/targets are artificial test
fixtures, not additions to the research dataset. A fresh dependency installation
and rerunning full neural training are separate from the source-release checks.

## Timing and interpretation

Synchronized inputs and eight ordered pooling bins retain coarse within-record
position. Independent branch embeddings do not explicitly estimate beat-to-beat
ECG–PPG delay, and late fusion combines scalar expert predictions rather than
aligned waveform features. The architecture tests compact quality/reliability
fusion; it does not establish that PAT/PTT features are uninformative. No
timing-aware comparative ablation was run for this release.

Both studies reuse an analyzed development cohort. Conditional bootstrap
intervals omit parts of model-fitting uncertainty. Small error differences,
parameter savings, selective retention, and CPU timing do not establish external
generalization, unbiased access, clinical validity, or embedded-device efficiency.

## AI assistance and license

See [AI_ASSISTANCE.md](AI_ASSISTANCE.md) for assistance with code, experiment
orchestration/checking, tables, figures, manuscript drafting, and paraphrasing.
The corresponding author is responsible for verifying AI-assisted outputs and
scientific interpretations; publication of code does not itself establish author
review or journal-policy compliance.

Code is released under the [MIT License](LICENSE). Dataset access terms are
separate. Waveforms, participant manifests, prediction files, trained checkpoints,
manuscripts, credentials, and private service access links are not redistributed.
