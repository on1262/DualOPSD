# DualOPSD: Closing the Feedback Loop in On-Policy Self-Distillation

This repository holds the code for the paper *DualOPSD: Closing the Feedback Loop in On-Policy Self-Distillation* by Yutong Chen, Guangfu Guo, Zhichao Xu and Kunpeng Liu. It covers training, evaluation, diagnostics, the analysis behind every table, and the plotting scripts for the figures.

## Method

On-policy self-distillation (OPSD) samples a trajectory from the student. The same model, conditioned on a verified reference solution, acts as a privileged teacher and gives a full next-token distribution at every student prefix. The student minimizes a forward KL, `KL(p_T || p_S)`, with a pointwise upper clip τ on each vocabulary contribution. Because the teacher is frozen, it never learns which parts of its supervision the clip rejected.

DualOPSD w/o Residual (`dualopsd_direct`) adds that feedback. After each student step, a separate teacher LoRA adapter on the same frozen backbone minimizes the unclipped `KL(sg[p_S⁺] || p_T)` toward the updated student, on the same trajectory.

DualOPSD (`dualopsd`) keeps the alternating update, but its teacher fits a residual target, `q = softmax(log p_S⁺ + β·(log p⁰(·|x,r) − log p⁰(·|x)))`. The term in parentheses is the shift that the privileged context produces in the frozen initial model `p⁰`, so the teacher stays sensitive to the reference. Only the student adapter is merged and evaluated.

The losses are in `dualopsd/losses.py` and the alternating loop is in `dualopsd/trainer.py`.

## Installation

The paper runs used Python 3.10, CUDA 12.x and one NVIDIA A100 80GB GPU per GPU job. Nothing is bundled with the repository; all dependencies are installed from the package indexes. `requirements.txt` pins the versions used for the paper (torch 2.8.0, transformers 4.57.1, trl 0.26.0, peft 0.17.1, vllm 0.11.0, math-verify 0.8.0, plus scipy and matplotlib for analysis and figures).

```bash
conda env create -f environment.yml      # creates the `dualopsd` env (Python 3.10) and installs requirements.txt
conda activate dualopsd
```

Without conda, `pip install -r requirements.txt` in a Python 3.10 environment gives the same packages.

Training, the baselines and the diagnostics load models with FlashAttention 2. It has to be built against the installed torch, so it is not in `requirements.txt`; the paper runs used flash-attn 2.8.3:

```bash
pip install flash-attn==2.8.3 --no-build-isolation
```

Run every command from the repository root. `python -m dualopsd ...` and `python -m analysis.<module>` then find the packages without any `PYTHONPATH` setup, and the Slurm jobs set `PYTHONPATH` themselves.

## Configuration: `configs/user.env`

All site-specific values live in `configs/user.env`. Every launcher and Slurm job sources it, and the Python code reads these values only from environment variables. Each line has the form `KEY=${KEY:-default}` (or `${KEY-default}` where an empty value is meaningful), so you can edit the file or override any value from the calling shell, for example `DUALOPSD_SCRATCH=/path/to/outputs bash scripts/main.sh --dry-run`. Run `source configs/user.env` before running Python commands by hand.

| Variable | Default | Meaning |
|---|---|---|
| `DUALOPSD_REPO` | repository root | Computed from the location of the file. |
| `DUALOPSD_SCRATCH` | `$DUALOPSD_REPO/outputs` | Root of all outputs: runs, merged models, evaluations, generations, diagnostics, results and logs. |
| `CONDA_ENV` | `dualopsd` | Conda env activated by every Slurm job. Set it to an empty string to use the environment that is already active (activation is also skipped when `conda` is not on `PATH`). |
| `CLUSTER_MODULES` | empty | Space-separated environment modules loaded before conda. Empty: no `module` calls. |
| `SLURM_ACCOUNT`, `SLURM_PARTITION` | empty | Passed to `sbatch` as `--account` / `--partition` only when non-empty. |
| `SLURM_GPU_GRES` | `gpu:1` | `--gres` request of every GPU job, e.g. `gpu:a100:1`. |
| `SLURM_GPU_CONSTRAINT` | empty | `--constraint` of every GPU job (your site's feature string for 80GB A100 nodes); omitted when empty. |
| `MODEL_QWEN3_1P7B`, `MODEL_QWEN3_4B`, `MODEL_QWEN3_8B` | `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B` | Base models, as Hugging Face ids or local directories. |
| `JUDGE_MODEL` | `$MODEL_QWEN3_8B` | Terminal-answer judge for the 500-problem held-out set. |
| `DATA_DIR`, `BENCH_DIR` | `$DUALOPSD_REPO/data/processed`, `$DUALOPSD_REPO/data/benchmarks` | Prepared training/held-out files and benchmark files. |
| `TRAIN_FILE`, `HELDOUT_FILE` | `$DATA_DIR/train_openthoughts_30k.jsonl`, `$DATA_DIR/heldout_500.jsonl` | Training file and held-out set. |
| `HF_HOME` | `$DUALOPSD_SCRATCH/hf_cache` | Hugging Face cache (Hub models and datasets land here on first use). |

One optional variable is not in the file: `GPU_MONITOR_INTERVAL` sets the `nvidia-smi` sampling period of GPU jobs in seconds (default 5).

## Data

```bash
bash scripts/prepare_data.sh                     # runs: python -m dualopsd prepare-data all
bash scripts/prepare_data.sh --download-models   # optional: also download Qwen3-1.7B/4B/8B to $DUALOPSD_SCRATCH/base_models/<name>
bash scripts/prepare_data.sh --dry-run           # print the commands only
```

Data preparation runs locally on CPU and needs network access. `prepare-data {train,heldout,benchmarks,all}` downloads each source at a pinned Hugging Face revision and checks every written file against the sha256 in `data/manifest.json` (`--update-manifest` records new hashes instead; `--overwrite` rewrites existing files). Without `--download-models`, the model ids in `configs/user.env` are fetched into `$HF_HOME` the first time a job uses them. If you do download them, point `MODEL_QWEN3_*` at the downloaded directories. Downloading first is worth it on a fresh `$HF_HOME`: otherwise the first wave of GPU jobs all fetch Qwen3-8B (about 16 GB) at the same time.

| File | Source (revision) | Rows |
|---|---|---|
| `$DATA_DIR/train_openthoughts_30k.jsonl` | `siyanzhao/Openthoughts_math_30k_opsd` (`1f33e9dc`), deterministic shuffle with seed 42 | 29,434 |
| `$DATA_DIR/heldout_500.jsonl` | 500 problems sampled with seed 42 from the last 20,000 rows of the training file; used for β selection and the diagnostics | 500 |
| `$BENCH_DIR/aime24.jsonl` | `HuggingFaceH4/aime_2024` (`2fe88a2f`) | 30 |
| `$BENCH_DIR/aime25.jsonl` | `MathArena/aime_2025` (`87387eac`) | 30 |
| `$BENCH_DIR/hmmt25.jsonl` | `FlagEval/HMMT_2025` (`0b2a3e6e`) | 30 |

Distillation reads the training file in file order with no reshuffling. The SFT and GRPO baselines load the same dataset at the same revision directly from the Hub.

## Repository structure

```
README.md
requirements.txt        pinned Python dependencies
environment.yml         conda env `dualopsd` (Python 3.10 + requirements.txt)
.gitignore
configs/
  user.env              paths, Slurm settings and model ids (the only site-specific file)
  accelerate.yaml       single-process bf16 accelerate config (SFT / GRPO)
  distill/              base.yaml plus one overlay per method: opsd.yaml, opsd_matched.yaml,
                        dualopsd_direct.yaml, dualopsd.yaml, clip_high.yaml, clip_low.yaml
  models/               qwen3-1.7b.yaml, qwen3-4b.yaml, qwen3-8b.yaml: model id, student clip τ,
                        micro-batch size, vLLM batch, selected β
data/
  manifest.json         pinned sources, revisions, row counts and sha256 of every prepared file
  processed/, benchmarks/   filled by scripts/prepare_data.sh (empty in the repository)
dualopsd/               Python package; entry point: python -m dualopsd <command>
  __main__.py           command dispatcher
  train.py              `train` command: config merging, data batches, auto-resume
  trainer.py            rollout -> student update -> teacher update loop
  losses.py             clipped forward KL, direct teacher KL, residual target, optional teacher clip
  config.py             config schema and validation
  adapters.py           student and teacher LoRA adapters on one frozen backbone
  rollout.py            vLLM rollout engine with refreshable LoRA adapters
  prompts.py            student and privileged-teacher prompts
  checkpoint.py         adapter / optimizer checkpointing and output-directory checks
  rollout_dump.py       rollouts.jsonl logging
  types.py              shared data types
  baselines/            sft.py, grpo.py (TRL), trl_checkpointing.py (segmented, resumable TRL runs)
  evaluation/           benchmark.py (avg@k), merge.py, generate.py, judge.py, terminal_answer.py
  verifiers/            math_official.py (math_verify), math.py (rule-based fallback)
  data/prepare.py       training file, held-out set, benchmarks
  diagnostics/          posthoc.py (KL / clipping / JS retention), recurrence.py (R_k), common.py
analysis/               CPU analysis, one module per experiment: python -m analysis.<module>
  common.py             paths under $DUALOPSD_SCRATCH, readers, statistics, LaTeX table rows
  main_results.py compute.py select.py beta.py teacher_clip.py curves.py mismatch.py
  teacher_diag.py style.py
figures/                plotting scripts for the paper figures
  style.py              shared matplotlib theme and --results-dir / --output-dir options
  accuracy_curves.py beta_validation.py diagnostics.py
scripts/                bash launchers, one per experiment
  lib.sh                shared options, paths, job builders, sbatch submission, job setup
  prepare_data.sh main.sh beta.sh teacher_clip.sh curves.sh mismatch.sh teacher_diag.sh style.sh
slurm/                  sbatch templates: train_distill, train_sft, train_grpo, merge, eval,
                        generate, judge, diagnostics, analysis
```

## Command line

```bash
python -m dualopsd <command> --help
```

| Command | Purpose |
|---|---|
| `train --config A.yaml [--config B.yaml ...] [--set key=value ...] [--output-dir DIR] [--resume auto\|RUN/checkpoint-N] [--smoke] [--dry-run]` | OPSD, DualOPSD and ablation distillation. Configs merge in order: `configs/distill/base.yaml`, `configs/models/<model>.yaml`, `configs/distill/<method>.yaml`. `--dry-run` prints the effective config and checks the data. |
| `sft`, `grpo` | TRL baselines, run as `accelerate launch --config_file configs/accelerate.yaml -m dualopsd sft ...` (TRL arguments plus `--stop_after_step` and `--smoke`). |
| `merge --base B --adapter RUN/checkpoint-N/student --output DIR` | Merge the student LoRA into a standalone model. |
| `eval --model DIR --output-dir DIR [--benchmarks ...] [--num-samples 12] [--seed 42]` | AIME24/AIME25/HMMT25 avg@k. Defaults: T=1.0, top-p 1.0, 32,768 total tokens including the prompt, math_verify grading with a rule-based fallback. |
| `generate --base-model B --adapter RUN/checkpoint-N/{student,teacher} --policy {student,teacher} --output F [--dataset F]` | One response per held-out problem (dataset default: `$DATA_DIR/heldout_500.jsonl`). |
| `judge --inputs F [F ...]` | Terminal-answer judge (`$JUDGE_MODEL`, Qwen3-8B); writes `<name>_judged.jsonl` next to each input `<name>.jsonl`. |
| `prepare-data {train,heldout,benchmarks,all}` | See [Data](#data). |
| `posthoc --run-dir RUN --step N --method M --output-dir DIR` | Re-score training rollouts: KL in both directions, clipped fraction, target fit, JS retention. |
| `recurrence --run-dir RUN --method M --output-dir DIR` | Top-k recurrence R_k of clipped contributions between steps 20 and 100. |

Example: train DualOPSD on Qwen3-4B with seed 43, without Slurm.

```bash
source configs/user.env
python -m dualopsd train --config configs/distill/base.yaml --config configs/models/qwen3-4b.yaml \
    --config configs/distill/dualopsd.yaml --set seed=43 --set teacher.residual_beta=0.7 \
    --output-dir $DUALOPSD_SCRATCH/runs/main/qwen3-4b/dualopsd_seed43
```

The trainer saves both adapters every 5 steps and the optimizer state every 25 steps, keeping only the latest optimizer state. It resumes automatically from the output directory.

## Reproducing the paper

Each experiment has one launcher in `scripts/`. A launcher submits independent Slurm chains: train, merge, then one evaluation job per benchmark; or generate, then judge. After all of its jobs have ended, it submits CPU analysis jobs. Outputs that already exist are skipped, and paths under `$DUALOPSD_SCRATCH` that are symlinks are never written to. Without Slurm, `--dry-run` lists every job in dependency order. Each `slurm/<template>.sbatch` shows the Python command it runs on the printed arguments, so you can run the jobs by hand.

| Paper result | Launcher | Analysis |
|---|---|---|
| Main avg@12 accuracy on AIME24/AIME25/HMMT25 (Table `tab:main`); results by training seed (`tab:seed-results`); cost and generation behavior averaged over scales (`tab:tradeoff`) and per scale (`tab:tradeoff-by-scale`); training cost of 100-step runs (`tab:full-compute`); generation length and truncation (`tab:full-generation`) | `scripts/main.sh` | `analysis.main_results`, `analysis.compute`, `analysis.select matched` |
| Residual weight search on the held-out set (`tab:beta-validation`, `fig:beta-validation`); Qwen3-8B residual weight sensitivity on AIME 2024 (`tab:beta`) | `scripts/beta.sh` | `analysis.beta`, `analysis.select beta` |
| Teacher-side clipping ablation on the DualOPSD teacher objective (`tab:core-ablation`) and per scale and benchmark (`tab:teacher-clip-scales`) | `scripts/teacher_clip.sh` | `analysis.teacher_clip` |
| Qwen3-4B accuracy over training steps (`fig:acc-curves`) | `scripts/curves.sh` | `analysis.curves` |
| Distributional mismatch and temporal recurrence on Qwen3-4B (`fig:diagnostic`); checkpoint diagnostics (`tab:diagnostic-ci`); recurrence as a function of k (`tab:recurrence-sensitivity`) | `scripts/mismatch.sh` | `analysis.mismatch` |
| Teacher accuracy and privileged-reference sensitivity on Qwen3-4B (`fig:ablation-dynamics`, `tab:role-dynamics-data`); positive forward-KL mass removed by the student cap (`tab:role-clipped-mass`) | `scripts/teacher_diag.sh` (plus the `mismatch.sh` posthoc outputs) | `analysis.teacher_diag` |
| Teacher-student formatting distance at step 100 (`tab:style-alignment-main`, `tab:style-alignment`) | `scripts/style.sh` | `analysis.style` |

The overview figure (`fig:overview`) is a hand-drawn diagram and has no script.

What each launcher runs:

- `main.sh` covers methods `base sft grpo opsd opsd_matched dualopsd_direct dualopsd`, models `qwen3-1.7b qwen3-4b qwen3-8b` and seeds `42 43 44`. Base is only evaluated, with generation seeds 42/43/44. DualOPSD uses the selected β (1.7B 0.8, 4B 0.7, 8B 0.5); `--beta B` instead trains a run named `dualopsd_beta<B>_seed<S>`. SFT trains for 100 steps, GRPO for 200, and the distillation methods for 100.
- `beta.sh [--only validation|sensitivity]`. The validation part trains a seed-42 DualOPSD run for every β of the grid (1.7B {0.1, 0.2, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1}; 4B and 8B {0.1, 0.2, 0.5, 0.7, 1.0}). The step-100 student of each run generates on the held-out set and the judge scores the answers. The sensitivity part evaluates Qwen3-8B seed-42 step-100 students on AIME24 for β ∈ {0, 0.1, 0.2, 0.5, 0.7, 1.0}, with β=0.7 and β=1.0 also evaluated on AIME25 and HMMT25. β=0 is DualOPSD w/o Residual.
- `teacher_clip.sh` runs `clip_high` and `clip_low` for 3 models × 3 seeds. The teacher minimizes the DualOPSD objective `KL(sg[q] || p_T)` (residual target, selected β) with an upper or lower clip on each vocabulary contribution before the vocabulary sum, at the student threshold τ (1e-6 for 1.7B and 4B, 1e-7 for 8B). The unclipped arm is the `main` DualOPSD run (`runs/main/<model>/dualopsd_seed<S>`). For 1.7B, the clip runs get the same `rollout.max_num_seqs=32` override as DualOPSD.
- `curves.sh` merges and evaluates the Qwen3-4B `opsd` and `dualopsd` students from `main` at steps 20/40/60/80/100, seeds 42/43/44.
- `mismatch.sh [--only posthoc|recurrence]` works on the Qwen3-4B `opsd`, `dualopsd_direct` and `dualopsd` runs from `main`. `posthoc` re-scores each checkpoint (steps 20 to 100) on its own training rollouts. `recurrence` first generates the fixed responses (the OPSD seed-42 step-20 student on the held-out set), then computes R_k between steps 20 and 100.
- `teacher_diag.sh` generates teacher and student responses on the held-out set for the Qwen3-4B `dualopsd_direct` and `dualopsd` runs at steps 20 to 100, and the judge scores them. It also generates OPSD teacher and student responses at step 100 only, without judging; `style.sh` uses those for Qwen3-4B. JS retention and clipped mass come from the `mismatch.sh` posthoc outputs.
- `style.sh` generates step-100 teacher and student responses of `opsd` and `dualopsd` for 1.7B and 8B. For 4B it links the `teacher_diag.sh` generations.

### Launcher options

All experiment launchers (`main.sh`, `beta.sh`, `teacher_clip.sh`, `curves.sh`, `mismatch.sh`, `teacher_diag.sh`, `style.sh`) accept:

```
--model LIST     comma-separated models (qwen3-1.7b,qwen3-4b,qwen3-8b)
--method LIST    comma-separated methods
--seed LIST      comma-separated training seeds (42,43,44)
--beta LIST      residual weights (beta.sh grid filter; a single value for main.sh)
--steps LIST     checkpoint steps (curves/mismatch/teacher_diag; matched OPSD steps for main.sh)
--only PART      script-specific part (beta.sh: validation|sensitivity, mismatch.sh: posthoc|recurrence)
--after IDS      make every submitted job depend (afterok) on these job ids
--no-analysis    do not submit the final CPU analysis job
--smoke          one chain, 1 step / tiny inputs, 30 min jobs, outputs under $DUALOPSD_SCRATCH/smoke
--dry-run        print the sbatch commands without submitting
-h, --help       print the launcher's usage
```

Example: `bash scripts/main.sh --model qwen3-4b --method dualopsd --seed 42 --dry-run` prints one training job, one merge job, three evaluation jobs and two analysis jobs.

### Dependencies and order

1. `bash scripts/prepare_data.sh` (CPU, local).
2. `bash scripts/main.sh`. Every other experiment reads the `main` runs from `$DUALOPSD_SCRATCH/runs/main/...`.
3. Compute-matched OPSD (`opsd_matched`) continues each OPSD step-100 state to step 320 in two chained jobs. It is evaluated at the checkpoint whose cumulative training PFLOPs are closest to those of the step-100 DualOPSD run:
   ```bash
   bash scripts/main.sh --method opsd_matched           # trains (needs the opsd runs; submits them if missing)
   source configs/user.env
   python -m analysis.select matched                    # writes results/main/matched_selection.json (needs the tokenizers)
   bash scripts/main.sh --method opsd_matched           # merges and evaluates the selected steps
   ```
   For Qwen3-1.7B, the paper selected the matched steps against the β=1.0 DualOPSD runs (`dualopsd_beta1p0_seed<S>`). `main.sh --method opsd_matched` trains these runs for 1.7B without evaluating them, and `analysis.select matched` uses them as the 1.7B target by default (`--target-run MODEL=TEMPLATE` changes the target). Passing `--steps N` to `main.sh` skips the selection file.
4. `bash scripts/beta.sh` needs the `main` runs. It reuses these grid points through relative symlinks: the selected β of each model, β=0 (`dualopsd_direct`) and 1.7B β=1.0 (from step 3). Run it after `main.sh`, or pass `--after <main job ids>`. `python -m analysis.select beta` applies the selection rule to the held-out results (most correct, then lower truncation, then fewer tokens, then smaller β) and writes `results/beta/selected_beta.json`. The paper's selection (0.8 / 0.7 / 0.5) is fixed in `configs/models/*.yaml` (`teacher.residual_beta`) and in `selected_beta` in `scripts/lib.sh`.
5. `curves.sh`, `mismatch.sh` and `teacher_diag.sh` need the Qwen3-4B `main` checkpoints at steps 20 to 100. `analysis.teacher_diag` also reads the `mismatch.sh` posthoc outputs.
6. `style.sh` needs the `main` step-100 checkpoints, and for Qwen3-4B the `teacher_diag.sh` generations.
7. `teacher_clip.sh` trains on its own. `analysis.teacher_clip` also reads the `main` DualOPSD evaluations for the unclipped arm.

With `--after`, you can submit a downstream launcher before its inputs exist. A missing input then produces a warning instead of an error.

### Output layout (`$DUALOPSD_SCRATCH`)

```
runs/<exp>/<model>/<run>/                checkpoint-N/{student,teacher}/, config.yaml, metrics.jsonl, rollouts.jsonl
models/<exp>/<model>/<run>[_step<N>]/    merged student models
eval/<exp>/<model>/<run>[_step<N>]/<benchmark>/{samples.jsonl,summary.json}
generations/<exp>/<model>/<run>_step<N>/{student,teacher}.jsonl (+ *_judged.jsonl)
diagnostics/mismatch/{posthoc,recurrence}/qwen3-4b/...
results/<exp>/*.json                     analysis outputs (tables and figure data)
results/figures/                         figure PDFs and PNG previews
logs/slurm/%x_%j.{out,err}               Slurm logs
logs/jobs.tsv                            every submitted job: time, job id, job name, dependencies
logs/gpu_monitor/*_gpu.csv               nvidia-smi samples of each GPU job
smoke/...                                the same layout for --smoke runs
```

Runs are named `<method>_seed<S>`. β-grid runs are `dualopsd_beta<0p7>_seed<S>`, Base is `base` or `base_genseed<S>`, and checkpoint-specific outputs append `_step<N>`. When an experiment reuses a run or evaluation from `main`, it holds a relative symlink to it.

### Smoke runs

```bash
bash scripts/main.sh --smoke                    # DualOPSD: 1-step train -> merge -> AIME24 eval
bash scripts/main.sh --smoke --method grpo      # another method
bash scripts/teacher_clip.sh --smoke            # works the same way for every launcher
```

Smoke mode submits only the first chain, with 30-minute jobs. Training runs 1 step on 2 problems with 64 new tokens. Evaluation uses 2 problems and 1 sample, generation uses 2 problems, judging covers the first 2 records, and posthoc and recurrence score 2 rows. Outputs go under `$DUALOPSD_SCRATCH/smoke/`, while inputs from other experiments are still read from the full-scale tree. No analysis job is submitted.

### Compute requirements

Every GPU job requests `SLURM_GPU_GRES` (and `SLURM_GPU_CONSTRAINT` if set); the paper used one A100 80GB per job. Wall-time limits are set by `job_time` in `scripts/lib.sh`, and CPU and memory by the `#SBATCH` lines in `slurm/*.sbatch`:

| Job | Resources | Time limit |
|---|---|---|
| Distillation, 100 steps (`train_distill`) | 1 GPU, 4 CPU, 64 GB | 1.7B 2 h, 4B 3 h, 8B 4 h |
| Matched OPSD continuation to step 320 | 1 GPU, 4 CPU, 64 GB, 2 chained jobs (auto-resume) | 1.7B 3 h, 4B 4 h, 8B 5 h per job |
| SFT, 100 steps | 1 GPU, 8 CPU, 80 GB | 4 h |
| GRPO, 200 steps | 1 GPU, 8 CPU, 80 GB | 1.7B/4B one 12 h job; 8B eight 25-step jobs of 4 h |
| Merge | CPU only, 4 CPU, 48 GB | 1 h |
| Eval (one benchmark, 12 samples) | 1 GPU, 4 CPU, 64 GB | 4 h |
| Generate (held-out set) | 1 GPU, 4 CPU, 64 GB | 1 h (2 h for β validation) |
| Judge / recurrence / posthoc | 1 GPU, 4 CPU, 64 GB | 0.5 h / 0.5 h / 1 h |
| Analysis | CPU only, 4 CPU, 32 GB | 1 h |

Seed-42 training time for 100 steps in the paper (`tab:full-compute`, GPU hours for 1.7B / 4B / 8B): OPSD 0.61 / 1.01 / 1.38, DualOPSD w/o Residual 0.81 / 1.40 / 2.00, DualOPSD 1.05 / 1.78 / 2.52.

On an empty `$DUALOPSD_SCRATCH`, the full launchers submit 296 (`main`), 63 (`beta`), 91 (`teacher_clip`), 121 (`curves`), 56 (`mismatch`), 97 (`teacher_diag`) and 25 (`style`) jobs, analysis jobs included. The `main` count covers only training of `opsd_matched`; its merge and evaluation jobs are added by the second `--method opsd_matched` call in step 3.

### Tables

Each analysis module reads `$DUALOPSD_SCRATCH` and writes `results/<exp>/<exp>.json` (`main_results.json` and `compute.json` for `main`; `curves.json` and `curves.csv` for `curves`). `--print-tables` prints the LaTeX rows of the paper tables, `--overwrite` replaces existing results, and `--scratch DIR` / `--output-dir DIR` change the input root and the output directory.

```bash
source configs/user.env
python -m analysis.main_results --print-tables   # tab:main, tab:seed-results (printed per model)
python -m analysis.compute --print-tables        # tab:tradeoff, tab:full-compute, tab:tradeoff-by-scale, tab:full-generation
python -m analysis.beta --print-tables           # tab:beta-validation, tab:beta (+ fig:beta-validation data)
python -m analysis.teacher_clip --print-tables   # tab:core-ablation, tab:teacher-clip-scales
python -m analysis.curves                        # fig:acc-curves data (curves.json / curves.csv)
python -m analysis.mismatch --print-tables       # tab:diagnostic-ci, tab:recurrence-sensitivity (+ fig:diagnostic data)
python -m analysis.teacher_diag --print-tables   # tab:role-dynamics-data, tab:role-clipped-mass (+ fig:ablation-dynamics data)
python -m analysis.style --print-tables          # tab:style-alignment (its OPSD, DualOPSD and reduction columns form tab:style-alignment-main)
```

`analysis.main_results` prints the per-seed rows under three keys (`tab:seed-results`, `tab:seed-results-4b`, `tab:seed-results-8b`), one per model; in the paper they form a single table.

### GPU hours

`analysis.compute` recomputes PFLOPs from each run's `rollouts.jsonl`, which needs the tokenizers from `MODEL_QWEN3_*`. GPU hours are the Slurm elapsed time × GPUs of the seed-42 training jobs. By default the job ids come from `$DUALOPSD_SCRATCH/logs/jobs.tsv`, which the launchers append to on every real submission, matched by job name (`main_<model>_<run>_train`). For compute-matched OPSD, the count includes the OPSD step-100 job plus every continuation job (`..._train` and `..._train2`). The elapsed time is read from GPU-monitor logs when you pass `--gpu-log-dir $DUALOPSD_SCRATCH/logs/gpu_monitor`, and from `sacct` otherwise.

To use other jobs, pass `--gpu-jobs FILE` with a JSON file of the form `{"jobs": {model: {method: [slurm job ids]}}}`. If no `jobs.tsv` exists and `--gpu-jobs` is not given, GPU hours are reported as n/a and the other columns are still computed. Runs executed outside Slurm therefore have no GPU-hour column.

### Figures

The figure scripts read the analysis outputs from `--results-dir` (default `$DUALOPSD_SCRATCH/results`) and write `<name>.pdf` plus a `<name>.png` preview to `--output-dir` (default `<results-dir>/figures`):

```bash
source configs/user.env
python figures/accuracy_curves.py    # fig:acc-curves      -> accuracy_curves.pdf      (needs results/curves/curves.json)
python figures/beta_validation.py    # fig:beta-validation -> heldout_beta_accuracy.pdf (needs results/beta/beta.json)
python figures/diagnostics.py        # fig:diagnostic, fig:ablation-dynamics -> diagnostic_recurrence.pdf, ablation_dynamics.pdf
                                     # (needs results/mismatch/mismatch.json and results/teacher_diag/teacher_diag.json)
```

## External resources

The code downloads only public third-party resources: the base models `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B` and `Qwen/Qwen3-8B`, the training dataset `siyanzhao/Openthoughts_math_30k_opsd`, and the benchmarks `HuggingFaceH4/aime_2024`, `MathArena/aime_2025` and `FlagEval/HMMT_2025`, all from the Hugging Face Hub.

## Citation

```bibtex
@misc{chen2026dualopsd,
  title  = {DualOPSD: Closing the Feedback Loop in On-Policy Self-Distillation},
  author = {Chen, Yutong and Guo, Guangfu and Xu, Zhichao and Liu, Kunpeng},
  year   = {2026},
  url    = {https://github.com/on1262/DualOPSD}
}
```
