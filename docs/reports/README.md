# Experiment reports

Decision-bearing reports on substantial experiments. These are the source
material for the later engineering/research article (`AGENT_WORKFLOW.md` §11,
§20).

- One file per substantial experiment, named after its config
  (e.g. `yolov12n_cpu_100e_plus_60e_416_b2_gentle_aug.md`).
- Trivial smoke checks do not get a report.
- `docs/experiments.md` is the chronological index and links here.
- `docs/results/` holds the machine-readable metric records these reports cite.

## Writing a report

Copy [`TEMPLATE.md`](TEMPLATE.md) to `<config-name>.md` and fill it in. The
twelve sections follow `AGENT_WORKFLOW.md` §11.3.

### Who fills which section

**Factual** — may be drafted from `docs/results/` records and the config:
Objective, Hypothesis, Baseline, Experimental setup, Change under test, Results,
Qualitative observations, Error analysis, Observed facts.

**Owner's** — interpretation and project-direction judgment: Interpretation,
Limitations / threats to validity, Decision / next step. A draft of these
carries a `> Draft — needs owner confirmation` line at the top of the section and
introduces no claim beyond what the owner has already written elsewhere.

## Figures

Only hand-picked images promoted from `outputs/` into [`assets/`](assets/).
Generated sample batches and per-image JSON stay local under `outputs/`.
