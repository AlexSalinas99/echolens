# EchoLens

**A Human-Speech Dataset for Auditing Demographic Sensitivity in Audio-Language Models**

[![Paper](https://img.shields.io/badge/paper-EMNLP%202026%20Findings-b31b1b)](https://2026.emnlp.org/)
[![Data](https://img.shields.io/badge/%F0%9F%A4%97%20data-alexsdl%2FEchoLens-yellow)](https://huggingface.co/datasets/alexsdl/EchoLens)
[![Code license](https://img.shields.io/badge/code-Apache--2.0-blue)](LICENSE)
[![Data license](https://img.shields.io/badge/data-CC%20BY--NC%204.0-lightgrey)](LICENSE-DATA)

Voice interfaces are increasingly moving away from transcription pipelines towards
end-to-end systems that directly respond to audio inputs. This development in turn
requires a shift in evaluation methodology away from transcription accuracy and towards
more substantive markers such as response validity. We introduce **EchoLens**, a
demographically stratified dataset of 20,008 recordings from 515 adult participants
residing in the U.S., with self-reported demographic information across race, gender,
age, accent and primary language, among others. Each participant speaks out loud and
verbatim a randomized subset of 69 advice-seeking and estimation prompts spanning 11
domains grounded in the American Time Use Survey. Most prompts elicit quantitative
responses from the model, allowing for direct analysis of output distributions across
demographic subgroups without requiring reliance on LLM-as-a-judge. EchoLens also
contains a documented audit protocol, which — in an illustrative application — we use to
evaluate six current audio-language models. Under this assessment, we find that racial
disparities are more frequent and larger in magnitude than gender disparities, with
particular concentration in two models.

**Authors** — [Alejandro Salinas](mailto:alexsdl@law.stanford.edu) (Stanford University) ·
Allison Koenecke (Cornell Tech) · Julian Nyarko (Stanford University)

**Paper** — Findings of the Association for Computational Linguistics: EMNLP 2026
(ACL Anthology entry to appear)

**Data** — <https://huggingface.co/datasets/alexsdl/EchoLens>

---

## What's in this repository

The audit **code** and the paper's **rendered figures and tables**. Appendix J of the
paper defers its extended per-prompt result set to this repository, so everything the
appendix points at is under `paper/`.

The 20,008 recordings and the 1.5 GB of raw model responses are **not** in git — they are
distributed through Hugging Face and fetched by `scripts/download_from_hf.py`.

## The dataset at a glance

| | |
| --- | --- |
| Speakers | 515 US-resident adults (249 `new_recruits`, 266 `reconsent`) |
| Recordings | 20,008 wav, 16 kHz mono 16-bit PCM |
| Instrument | 74 utterance roles: 69 scenario prompts (each speaker reads a random 35), 2 verbal suffixes, 1 demographic probe, 1 dialect probe, 1 mic check |
| Domains | 11, grounded in the American Time Use Survey |
| Race (self-report) | Black 254 · White 257 · Other/Unknown 4 |
| Gender (self-report) | Female 253 · Male 257 · Non-binary 5 |
| Accent | non-SAE 361 · SAE 154 (descriptive third axis) |
| Recommended subset | `in_strict_audit` — 500 speakers |

Splits are **nested boolean columns, not disjoint partitions**:
`in_full` 515 ⊇ `in_audit_eligible` 506 ⊇ `in_strict_audit` 500 ⊇ `in_balanced` 496
(124 per race × gender cell). Primary analyses in the paper use `in_strict_audit`.

## The audit at a glance

386,208 extracted responses from **6 audio-language models** × **4 conditions**.

| Model | Provider |
| --- | --- |
| `gemini-3.1-flash-lite-preview`, `gemini-3.1-pro-preview`, `gemini-3.5-flash` | Google |
| `gpt-audio-1.5` | OpenAI |
| `Qwen/Qwen2.5-Omni-7B` | Alibaba (local) |
| `moonshotai/Kimi-Audio-7B-Instruct` | Moonshot (local) |

| Condition | What the model receives |
| --- | --- |
| `canonical_text_response` | the prompt as text — control |
| `direct_audio_response` | the participant's audio |
| `external_transcript_response` | a Whisper transcript of that audio |
| `self_transcript_response` | the model's own transcript of that audio |

Headline metric: standardized mean difference (SMD) between demographic groups, with
participant-clustered bootstrap CIs and Benjamini–Hochberg correction. Sign conventions
are race = Black − White, gender = Female − Male, accent = non-SAE − SAE.

## Install

```bash
conda create -n echolens python=3.12 && conda activate echolens
pip install -r requirements.txt              # analysis and figures
pip install -r requirements-inference.txt    # only to re-run the models
```

## Reproduce

Four tiers, cheapest first. Most readers want tier 1 or 2.

**Tier 1 — regenerate every figure and table from the shipped results.** No download.

```bash
jupyter lab notebooks/08_disparity_figures.ipynb   # then 09_disparity_tables.ipynb
```

**Tier 2 — recompute the SMDs from the normalized responses.** No download;
`data/model_outputs/extracted_responses.parquet` ships in this repo.

```bash
jupyter lab notebooks/07_disparity_smd.ipynb
python scripts/bh_multiplicity.py                  # multiplicity correction
```

**Tier 3 — re-run the full pipeline from the raw model outputs.** ~1.7 GB download.

```bash
python scripts/download_from_hf.py --parts pipeline
# notebooks 06 -> 07 -> 08 -> 09, then 10 for WER
```

**Tier 4 — re-run inference against the models.** Needs API keys and a GPU; ~6 GB download.

```bash
cp .env.example .env && $EDITOR .env
python scripts/download_from_hf.py --parts audio
python scripts/build_assembled_audio.py            # rebuilds the 7.3 GB assembled cache
python scripts/run_inference.py --help
```

## Data access

The dataset is **gated**: accept the use policy once on the
[dataset page](https://huggingface.co/datasets/alexsdl/EchoLens), then authenticate.

```bash
hf auth login                                      # or export HF_TOKEN=...
python scripts/download_from_hf.py --list          # what's available, and how big
python scripts/download_from_hf.py --parts audio   # lands in data/audio/
```

The Hugging Face layout mirrors this repository's `data/` tree, so downloads land exactly
where the notebooks expect them — nothing needs to be moved. Downloads are resumable:
re-run the same command after an interruption and it picks up where it stopped.

| You want | Download | Plus derived | Total on disk |
| --- | --- | --- | --- |
| Figures and tables only | — | — | 52 MB |
| Recompute the SMDs | — | — | 52 MB |
| Re-run normalization | 1.7 GB | — | ~1.8 GB |
| Re-run inference | 6.0 GB | 7.3 GB assembled | ~14 GB |
| Everything | 13.7 GB | 7.3 GB assembled | ~21 GB |

---

## Repository map

What lives where. This section describes the on-disk layout only.

```
echolens/
├── data/         # artifacts (inputs, intermediates, outputs)
├── notebooks/    # numbered, narrative-driven analysis pipeline (01 → 10)
├── scripts/      # command-line runners
├── src/          # reusable Python helpers imported by the notebooks
└── paper/        # rendered figures and tables
```

### `data/`

```
data/
├── metadata/         # participant-, prompt-, clip- and split-level manifests (CSV)
├── audio/            # ⬇ HF — raw per-clip recordings, one folder per participant
├── _audio_assembled/ # ⬇ regenerate — assembled audio inputs actually fed to models
└── model_outputs/    # raw model responses, normalized responses, ASR, metrics
```

`⬇ HF` = not in git, fetch with `scripts/download_from_hf.py`.
`⬇ regenerate` = not distributed at all; rebuild with `scripts/build_assembled_audio.py`.

#### `data/metadata/`

| File | One row per | What it carries |
| --- | --- | --- |
| `participants.csv` | participant | demographic self-reports, cohort, eligibility flags, recording metadata |
| `prompts.csv` | prompt | `question_id`, `prompt_type`, `scenario`, `prompt_text`, suffix info, `is_quantitative`, `direction` |
| `recordings.csv` | clip | `clip_id`, `participant_id`, `question_id`, `prompt_type`, `audio_path` |
| `splits.csv` | participant | boolean masks: `in_full`, `in_audit_eligible`, `in_strict_audit`, `in_balanced` |

#### `data/audio/` — ⬇ HF

Two recruitment cohorts, each a folder of per-participant subfolders containing one `.wav`
per recorded clip (scenario / suffix / probe / mic check).

```
data/audio/
├── new_recruits/    # P0001/, P0002/, …
└── reconsent/       # P####/
```

#### `data/_audio_assembled/` — ⬇ regenerate

Audio inputs *after* assembly (concatenating a scenario clip with that participant's own
suffix recordings). Filenames encode the build, e.g. `P0001__q01__q70_q71.wav`. This is
the directory the model providers actually read from.

#### `data/model_outputs/`

```
data/model_outputs/
├── responses/                       # ⬇ HF — raw model outputs (JSONL shards)
│   └── <provider>/<model>/<condition>/part-*.jsonl
├── transcripts/                     # ⬇ HF — ASR transcripts of participant audio
│   ├── external/whisper-1/part-*.jsonl
│   └── self/<gemini-model>/part-*.jsonl
├── _llm_cache/                      # ⬇ HF — extraction LLM cache (gpt-4o-mini fallback)
├── extracted_responses.parquet      # canonical normalized responses table (in git)
├── extraction_validation_sample.csv # manual-validation sample for extraction (in git)
├── secondary_by_group.csv           # group-aggregated secondary metrics (in git)
├── wer/wer_by_group.csv             # WER by group (in git; per-clip is ⬇ HF)
└── disparity/                       # SMD pipeline outputs (CIs, prompt-level, validity)
```

### `notebooks/`

Numbered so they form a top-to-bottom pipeline. Each opens with a markdown header
explaining its role; the one-liners below are pointers, not summaries.

| # | File | Role |
| --- | --- | --- |
| 01 | `01_prompts_overview.ipynb` | Tour of `prompts.csv` |
| 02 | `02_participants_overview.ipynb` | Tour of `participants.csv` |
| 03 | `03_splits_construction.ipynb` | Derives and writes `splits.csv` |
| 04 | `04_input_construction.ipynb` | Builds `_audio_assembled/` + the eval protocol |
| 05 | `05_evaluation_setup.ipynb` | Models × conditions matrix, per-provider smoke tests |
| 06 | `06_response_normalization.ipynb` | Raw outputs → `extracted_responses.parquet` |
| 07 | `07_disparity_smd.ipynb` | Computes scenario- and prompt-level SMDs |
| 08 | `08_disparity_figures.ipynb` | Renders figures under `paper/figures/smd/` |
| 09 | `09_disparity_tables.ipynb` | Renders tables under `paper/tables/smd/` |
| 10 | `10_transcript_wer.ipynb` | WER + auxiliary speech metrics |

### `scripts/`

| File | Purpose |
| --- | --- |
| `download_from_hf.py` | Fetch the bulk artifacts from Hugging Face into `data/` |
| `build_assembled_audio.py` | Materialize `data/_audio_assembled/` from the raw audio |
| `run_inference.py` | Bulk inference runner — models × conditions × runs, shardable, idempotent |
| `extend_llm_cache.py` | Fill the layer-4 gpt-4o-mini extraction cache for newly added models |
| `bh_multiplicity.py` | Benjamini–Hochberg FDR over the headline scenario SMDs |
| `run_appendix_bootstrap.py` | Direction-aware appendix bootstrap → the `__direction_clear` CSVs |

### `src/`

Reusable Python imported by the notebooks. Notebook-friendly, no CLI entry points.

```
src/eval/
├── conditions.py        # (model, condition) matrix, system prompts, generation params
├── input_prep.py        # pick/assemble inputs for a (participant, prompt, condition)
├── extraction.py        # layered response normalizer (structured → rule → gpt-4o-mini)
├── disparity.py         # SMD computation + cluster bootstrap CIs
├── speech_metrics.py    # WER + auxiliary speech metrics
└── providers/           # one synchronous call_once() helper per model family
    ├── gemini.py  openai_audio.py  qwen_omni.py  kimi_audio.py  asr_openai.py
```

Every provider exposes the same `is_available() -> bool` / `call_once(**kwargs) -> CallResult`
contract, so adding a model means adding one file and one `register_model()` call.

### `paper/`

Rendered artifacts consumed by the LaTeX paper. Nothing here is hand-edited; everything is
regenerated by the notebooks.

```
paper/
├── figures/{smd, smd_per_prompt, wer}/
└── tables/{smd, wer}/
```

#### Filename conventions

Figures and tables pack recurring tokens into filenames with `__` separators:

- **Condition**: `direct_audio_response`, `external_transcript_response`, `self_transcript_response`, `canonical_text_response`
- **Winsorization**: `raw` (none) vs. `wins1` (1 %)
- **Axis**: `race`, `gender`, `accent` (suffix `__with_accent` adds accent to combined plots)
- **Aggregation**: `scenario_smds`, `per_prompt_strip`, `models_stacked`, `combined_conditions`
- **ASR**: `whisper-1` (external) vs. `gemini-self` (self-transcript)
- **Metric**: `wer`, `wpm`, `non_vocal_fraction`
- **Appendix variant**: `__direction_clear` restricts to prompts with an unambiguous direction

Example: `gpt-audio-1.5__direct_audio_response__scenario_smds__wins1.pdf` is the
scenario-level SMD plot for GPT-Audio under the direct-audio condition with 1 %
winsorization.

---

## Ethics, consent, and IRB

EchoLens contains identifiable human speech and participant-linked demographic metadata.
The study was reviewed under Stanford IRB Protocol **#77233**. The released dataset
includes only recordings from participants who consented to public redistribution:
previously enrolled participants were re-contacted for affirmative redistribution consent,
and newly recruited participants consented to public release at the time of recording.

The main residual risk is identifiability. A voice is inherently identifying even though
participants were never asked to state names or other direct identifiers. To reduce ancillary identifiability the release uses opaque
participant identifiers, excludes direct platform identifiers and private linkage files,
and retains non-released data under secure access restrictions.

EchoLens is intended for **research auditing and evaluation of audio-language models**. It
is **not** for speaker identification, voice biometrics, production voice-system
deployment, or treating recordings as evidence of speaker identity. Those uses fall outside
participant consent and are prohibited regardless of license.

Demographic labels are **self-reported** and support a bounded audit of whether models
respond differently to comparable spoken requests across demographic speech variation.
Speech cues are not ground-truth evidence of identity, and the Black/White × Female/Male
primary audit design is not representative of all speakers or all forms of demographic harm.

Participants retain the right to withdraw. On a withdrawal request we will remove the
affected recordings and issue a new version.

## License

Code (`src/`, `scripts/`, `notebooks/`) — **Apache-2.0**, see [`LICENSE`](LICENSE).

Data and paper artifacts (`data/`, `paper/`, and the Hugging Face dataset) —
**CC BY-NC 4.0**, see [`LICENSE-DATA`](LICENSE-DATA). Commercial use requires prior written
permission: <alexsdl@law.stanford.edu>.

## Citation

The ACL Anthology entry is not yet published. Until then:

```bibtex
@inproceedings{salinas2026echolens,
  title     = {{EchoLens}: A Human-Speech Dataset for Auditing Demographic
               Sensitivity in Audio-Language Models},
  author    = {Salinas, Alejandro and Koenecke, Allison and Nyarko, Julian},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026},
  publisher = {Association for Computational Linguistics},
  note      = {To appear}
}
```

See [`CITATION.cff`](CITATION.cff) for the machine-readable form.
