#!/usr/bin/env python3
"""Bulk inference runner — produces the raw response JSONL shards.

The repo ships the *analysis* end-to-end (notebooks 06→10 regenerate from the
raw outputs already on disk), and `src/eval/providers/` ships one synchronous
`call_once()` per model family.

It writes rows in the exact schema notebook 06 consumes (`walk_raw_responses`
reads: task_id, model_provider, model_id, condition, clip_id, question_id,
prompt_type, run_index, response_text, response_parsed_json, error_type) plus
the surrounding identity/provenance columns the released JSONLs carry.

Scope of the eval set (mirrors notebook 05 §5):
  - participants: `splits.in_strict_audit`
  - prompts: 55 quantitative scenarios + q72 + q73 (57 evaluable)
  - runs: run_index ∈ {1, 2, 3}
  - conditions: whatever each model registers in `conditions.py`

Conditions:
  - direct_audio        : assembled wav (built by build_assembled_audio.py / on the fly)
  - canonical_text      : per-prompt text (looped over prompts×runs only; clip_id=None)
  - external_transcript : Whisper-1 transcript text, indexed by clip_id from
                          data/model_outputs/transcripts/external/whisper-1/
  - self_transcript     : Gemini-family only; two-step (self-transcribe → feed back)

Idempotent: existing task_ids in the target shards are skipped, so re-runs
resume. Credentials/hardware are gated by each provider's is_available().

Examples:
    # Headline only (direct_audio) for the two new Gemini models:
    python scripts/run_inference.py --models gemini_flash_35 gemini_pro_31 --conditions direct_audio

    # Kimi (local GPU), all its registered conditions:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_inference.py --models kimi_audio

    # Everything newly-registered, tiny smoke test:
    python scripts/run_inference.py --models kimi_audio gemini_flash_35 gemini_pro_31 --limit-participants 2 --runs 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from eval import conditions, input_prep
from eval.providers import gemini, kimi_audio, openai_audio, qwen_omni

META        = REPO_ROOT / "data" / "metadata"
AUDIO_ROOT  = REPO_ROOT / "data" / "audio"
CACHE_DIR   = REPO_ROOT / "data" / "_audio_assembled"
MO          = REPO_ROOT / "data" / "model_outputs"
RESP_ROOT   = MO / "responses"
WHISPER_DIR = MO / "transcripts" / "external" / "whisper-1"

RUN_INDICES_DEFAULT = (1, 2, 3)
SHARD_PARTICIPANTS  = 150  # participants per part-NNNN.jsonl shard

# Set from --max-output-tokens; overrides conditions.GENERATION_PARAMS when not None.
# Needed for "thinking" models (e.g. gemini-3.1-pro): their reasoning consumes the
# output budget, so the paper's 600 (tuned for non-thinking flash-lite) truncates
# the answer to empty. Raising the cap only rescues those — it never changes a
# response that already fit.
_MAX_OUT_OVERRIDE: int | None = None


def _max_out(cond_resp: str) -> int:
    if _MAX_OUT_OVERRIDE is not None:
        return _MAX_OUT_OVERRIDE
    return conditions.GENERATION_PARAMS["max_output_tokens"][cond_resp]

# short condition name (conditions.py) → JSONL/dir condition name
COND_TO_RESPONSE = {c: f"{c}_response" for c in conditions.CONDITIONS}


def _load_env() -> None:
    for env_file in (Path.home() / ".openai_env", REPO_ROOT / ".env"):
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def _sanitize(model_id: str) -> str:
    return model_id.replace("/", "_")


def _task_id(provider: str, cond_resp: str, unit: str, run_index: int, model_id: str) -> str:
    h = hashlib.sha256(f"{provider}|{model_id}|{cond_resp}|{unit}|{run_index}".encode()).hexdigest()[:16]
    return f"{provider}:{cond_resp}:{unit}:r{run_index}:{h}"


def _evaluable(prompts: pd.DataFrame) -> pd.DataFrame:
    return prompts[
        ((prompts["prompt_type"] == "scenario") & (prompts["is_quantitative"]))
        | (prompts["question_id"].isin([72, 73]))
    ].copy()


def _load_whisper_index() -> dict[str, str]:
    """{clip_id: transcript_text} from the shipped Whisper-1 JSONL shards."""
    idx: dict[str, str] = {}
    if not WHISPER_DIR.exists():
        return idx
    for shard in sorted(WHISPER_DIR.glob("*.jsonl")):
        with shard.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                cid, txt = r.get("clip_id"), r.get("transcript_text")
                if cid and txt is not None and cid not in idx:
                    idx[cid] = txt
    return idx


def _existing_task_ids(out_dir: Path) -> set[str]:
    done: set[str] = set()
    if not out_dir.exists():
        return done
    for shard in out_dir.glob("*.jsonl"):
        with shard.open() as fh:
            for line in fh:
                if line.strip():
                    try:
                        done.add(json.loads(line)["task_id"])
                    except (KeyError, json.JSONDecodeError):
                        pass
    return done


def _make_record(*, spec, cond, cond_resp, run_index, pid, cohort, qid, prow,
                 clip_id, audio_path, audio_path_with_suffix, text_input,
                 transcript_source, system_prompt, max_out, response_schema,
                 result_text, parsed_json, model_version, latency_s,
                 error_type, error_message, retry_count) -> dict:
    return {
        "run_id": None,
        "task_id": _task_id(spec.provider, cond_resp,
                            clip_id if clip_id else f"q{qid:02d}", run_index, spec.model_id),
        "model_provider": spec.provider,
        "model_id": spec.model_id,
        "condition": cond_resp,
        "run_index": run_index,
        "split": "strict_audit",
        "clip_id": clip_id,
        "participant_id": pid,
        "cohort": cohort,
        "question_id": qid,
        "prompt_type": prow["prompt_type"],
        "scenario": prow.get("scenario"),
        "is_quantitative": bool(prow["is_quantitative"]),
        "audio_path": audio_path,
        "audio_path_with_suffix": audio_path_with_suffix,
        "prompt_text_original": None,
        "suffix_text_appended": None,
        "prompt_text_final_for_text_condition": text_input,
        "transcript_source": transcript_source,
        "transcript_text": text_input if transcript_source else None,
        "request_summary": {"system_prompt": system_prompt,
                            "response_schema": response_schema,
                            "text_input": text_input is not None},
        "temperature": conditions.GENERATION_PARAMS["temperature"],
        "top_p": conditions.GENERATION_PARAMS["top_p"],
        "max_output_tokens": max_out,
        "seed": None,
        "response_raw": None,
        "response_text": result_text,
        "response_parsed_json": parsed_json,
        "finish_reason": None,
        "usage": None,
        "model_version_or_snapshot": model_version,
        "api_request_id": None,
        "latency_s": latency_s,
        "created_at_utc": None,
        "code_git_commit": None,
        "error_type": error_type,
        "error_message": error_message,
        "retry_count": retry_count,
    }


def _dispatch(spec, cond, *, system_prompt, audio_path, text_input, max_out, response_schema):
    """Call the right provider; return (response_text, parsed_json, model_version, latency_s)."""
    if spec.provider == "gemini":
        r = gemini.call_once(model_id=spec.model_id, condition=cond, system_prompt=system_prompt,
                             audio_path=audio_path, text_input=text_input,
                             max_output_tokens=max_out, response_schema=response_schema)
        parsed = None
        try:
            parsed = json.loads(r.response_text) if r.response_text else None
        except json.JSONDecodeError:
            parsed = None
        return r.response_text, parsed, r.model_version, r.latency_s
    if spec.provider == "openai":
        r = openai_audio.call_once(model_id=spec.model_id, system_prompt=system_prompt,
                                   audio_path=audio_path, max_output_tokens=max_out)
        return r.response_text, None, r.model_version, r.latency_s
    if spec.provider == "qwen":
        r = qwen_omni.call_once(model_id=spec.model_id, system_prompt=system_prompt,
                                audio_path=audio_path, text_input=text_input, max_output_tokens=max_out)
        return r.response_text, None, r.model_version, r.latency_s
    if spec.provider == "kimi":
        r = kimi_audio.call_once(model_id=spec.model_id, system_prompt=system_prompt,
                                 audio_path=audio_path, text_input=text_input, max_output_tokens=max_out)
        return r.response_text, None, r.model_version, r.latency_s
    raise ValueError(f"no dispatch for provider {spec.provider!r}")


def _provider_available(provider: str) -> bool:
    return {
        "gemini": gemini.is_available,
        "openai": openai_audio.is_available,
        "qwen": qwen_omni.is_available,
        "kimi": kimi_audio.is_available,
    }[provider]()


def _execute(thunks, out_handle, concurrency: int):
    """Run each thunk (returns a record or None), writing results under a lock.

    Serial when concurrency<=1; otherwise a thread pool (calls are I/O-bound for
    the API providers). Returns (n_written, n_err).
    """
    lock = threading.Lock()
    counters = {"written": 0, "err": 0}

    def _write(rec):
        if rec is None:
            return
        with lock:
            out_handle.write(json.dumps(rec) + "\n")
            counters["written"] += 1
            counters["err"] += bool(rec["error_type"])

    if concurrency <= 1:
        for t in thunks:
            _write(t())
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            for fut in as_completed([ex.submit(t) for t in thunks]):
                _write(fut.result())
    return counters["written"], counters["err"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="registry names, e.g. gemini_flash_35 gemini_pro_31 kimi_audio")
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="subset of a model's registered conditions (default: all it registers)")
    ap.add_argument("--runs", type=int, default=3, help="runs per clip (default 3)")
    ap.add_argument("--limit-participants", type=int, default=None)
    ap.add_argument("--max-retries", type=int, default=2, help="retries per failed call")
    ap.add_argument("--max-output-tokens", type=int, default=None,
                    help="override the per-condition output-token cap. Use a larger value "
                         "(e.g. 4096) for 'thinking' models like gemini-3.1-pro whose reasoning "
                         "would otherwise exhaust the default 600 and return empty.")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="parallel in-flight requests (use ~16-32 for API models like Gemini; "
                         "keep at 1 for local GPU models like Kimi)")
    ap.add_argument("--shard", default=None,
                    help="'i/N' — process only participant slice i of N (strided). Lets you "
                         "run N processes across N GPUs; each writes distinct shard files. "
                         "canonical_text (per-prompt, not per-participant) is handled by shard 0 only.")
    args = ap.parse_args()

    shard_i, shard_n = 0, 1
    if args.shard:
        shard_i, shard_n = (int(x) for x in args.shard.split("/"))

    global _MAX_OUT_OVERRIDE
    _MAX_OUT_OVERRIDE = args.max_output_tokens
    if _MAX_OUT_OVERRIDE is not None:
        print(f"max_output_tokens override: {_MAX_OUT_OVERRIDE}")

    _load_env()

    prompts      = pd.read_csv(META / "prompts.csv")
    participants = pd.read_csv(META / "participants.csv")
    splits       = pd.read_csv(META / "splits.csv")
    cohort_of    = dict(zip(participants["participant_id"], participants["cohort"]))

    ev   = _evaluable(prompts)
    pids = sorted(splits.loc[splits["in_strict_audit"], "participant_id"])
    if args.limit_participants:
        pids = pids[: args.limit_participants]
    if shard_n > 1:
        pids = pids[shard_i::shard_n]
        print(f"shard {shard_i}/{shard_n}: {len(pids)} participants")
    run_indices = tuple(range(1, args.runs + 1))
    whisper = _load_whisper_index()

    for name in args.models:
        spec = conditions.get_model(name)
        if not _provider_available(spec.provider):
            print(f"⚠ skipping {name}: provider {spec.provider!r} not available "
                  f"(missing credentials/hardware).")
            continue
        want_conds = args.conditions or list(spec.conditions)
        conds = [c for c in want_conds if c in spec.conditions]
        skipped = [c for c in want_conds if c not in spec.conditions]
        for c in skipped:
            print(f"  note: {name} does not register condition {c!r}; skipping it.")

        for cond in conds:
            cond_resp = COND_TO_RESPONSE[cond]
            out_dir = RESP_ROOT / spec.provider / _sanitize(spec.model_id) / cond_resp
            out_dir.mkdir(parents=True, exist_ok=True)
            done = _existing_task_ids(out_dir)
            print(f"\n=== {name} / {cond_resp} ===  ({len(done)} task_ids already on disk)")

            n_written = n_err = 0
            t_start = time.time()

            # canonical_text is per-prompt (not per-participant); only shard 0 runs it.
            if cond == "canonical_text" and shard_n > 1 and shard_i != 0:
                print(f"  (shard {shard_i}: skipping canonical_text — handled by shard 0)")
                continue

            # canonical_text is per-prompt (not per-participant): loop prompts×runs only.
            if cond == "canonical_text":
                thunks = []
                for run_index in run_indices:
                    for _, prow in ev.iterrows():
                        qid = int(prow["question_id"])
                        tid = _task_id(spec.provider, cond_resp, f"q{qid:02d}", run_index, spec.model_id)
                        if tid in done:
                            continue

                        def mk(run_index=run_index, prow=prow, qid=qid):
                            text_input = input_prep.canonical_text(prompts, qid)
                            sys_prompt = conditions.system_prompt(prompt_type=prow["prompt_type"], text_input=True)
                            schema = conditions.response_schema_for(prompt_type=prow["prompt_type"]) \
                                if spec.provider == "gemini" else None
                            max_out = _max_out(cond_resp)
                            return _run_one(spec, cond, cond_resp, run_index, None, None, qid, prow,
                                            clip_id=None, audio_path=None, audio_path_with_suffix=None,
                                            text_input=text_input, transcript_source=None,
                                            system_prompt=sys_prompt, max_out=max_out, response_schema=schema,
                                            max_retries=args.max_retries)
                        thunks.append(mk)
                with (out_dir / "part-0000.jsonl").open("a") as out:
                    n_written, n_err = _execute(thunks, out, args.concurrency)
                print(f"  wrote {n_written} rows ({n_err} errors) in {time.time()-t_start:.0f}s")
                continue

            # audio / transcript conditions: loop participants × prompts × runs, shard by participant.
            sh_tag = f"s{shard_i:02d}-" if shard_n > 1 else ""
            for block_i, base in enumerate(range(0, len(pids), SHARD_PARTICIPANTS)):
                shard_pids = pids[base: base + SHARD_PARTICIPANTS]
                shard_path = out_dir / f"part-{sh_tag}{block_i:04d}.jsonl"
                thunks = []
                for pid in shard_pids:
                    cohort = cohort_of.get(pid)
                    if cohort is None:
                        continue
                    for _, prow in ev.iterrows():
                        qid = int(prow["question_id"])
                        clip_id = f"{pid}_q{qid:02d}"
                        for run_index in run_indices:
                            tid = _task_id(spec.provider, cond_resp, clip_id, run_index, spec.model_id)
                            if tid in done:
                                continue

                            def mk(pid=pid, cohort=cohort, qid=qid, prow=prow, clip_id=clip_id, run_index=run_index):
                                return _prepare_and_run(
                                    spec, cond, cond_resp, run_index, pid, cohort, qid, prow, clip_id,
                                    whisper=whisper, max_retries=args.max_retries)
                            thunks.append(mk)
                with shard_path.open("a") as out:
                    nn, ne = _execute(thunks, out, args.concurrency)
                n_written += nn
                n_err += ne
            print(f"  wrote {n_written} rows ({n_err} errors) in {time.time()-t_start:.0f}s")

    print("\nall done. Re-run notebooks 06→09 to fold the new outputs into the headline.")


def _prepare_and_run(spec, cond, cond_resp, run_index, pid, cohort, qid, prow, clip_id, *,
                     whisper, max_retries):
    """Build inputs for one audio/transcript task, then call the model."""
    audio_path = f"audio/{cohort}/{pid}/{pid}_q{qid:02d}.wav"
    audio_path_with_suffix = None
    text_input = None
    transcript_source = None
    assembled_path = None

    if cond in ("direct_audio", "self_transcript"):
        try:
            assembled = input_prep.assemble_audio(
                participant_id=pid, cohort=cohort, question_id=qid, audio_root=AUDIO_ROOT,
                suffix_number=bool(prow["suffix_number"]), suffix_additional=bool(prow["suffix_additional"]),
                cache_dir=CACHE_DIR)
        except FileNotFoundError:
            return None
        assembled_path = assembled.audio_path
        if assembled_path.parent == CACHE_DIR:
            audio_path_with_suffix = str(assembled_path)

    if cond == "external_transcript":
        text_input = whisper.get(clip_id)
        if text_input is None:
            return None
        transcript_source = "openai_asr:whisper-1"

    if cond == "self_transcript":
        # Two-step: the model transcribes its own audio, then answers the transcript.
        try:
            st = gemini.transcribe_audio_self(
                model_id=spec.model_id, audio_path=assembled_path,
                max_output_tokens=conditions.GENERATION_PARAMS["max_output_tokens"]["self_transcription"],
                response_schema=conditions.SELF_TRANSCRIPTION_SCHEMA)
            parsed = json.loads(st.response_text) if st.response_text else {}
            text_input = parsed.get("transcript_text", st.response_text)
        except Exception:
            return None
        transcript_source = f"{spec.provider}:{spec.model_id}:self"

    call_audio = assembled_path if cond == "direct_audio" else None
    sys_prompt = conditions.system_prompt(prompt_type=prow["prompt_type"], text_input=(call_audio is None))
    schema = conditions.response_schema_for(prompt_type=prow["prompt_type"]) \
        if spec.provider == "gemini" else None
    max_out = _max_out(cond_resp)

    return _run_one(spec, cond, cond_resp, run_index, pid, cohort, qid, prow,
                    clip_id=clip_id, audio_path=audio_path,
                    audio_path_with_suffix=audio_path_with_suffix, text_input=text_input,
                    transcript_source=transcript_source, system_prompt=sys_prompt,
                    max_out=max_out, response_schema=schema, max_retries=max_retries)


def _run_one(spec, cond, cond_resp, run_index, pid, cohort, qid, prow, *,
             clip_id, audio_path, audio_path_with_suffix, text_input, transcript_source,
             system_prompt, max_out, response_schema, max_retries):
    """Call the model with retries; always return a schema-complete record."""
    call_audio = Path(audio_path_with_suffix) if (cond == "direct_audio" and audio_path_with_suffix) else None
    if cond == "direct_audio" and call_audio is None and cohort:
        # No suffix needed → the scenario wav itself is the direct-audio input.
        # Layout: audio/<cohort>/<pid>/<pid>_qNN.wav
        call_audio = AUDIO_ROOT / cohort / pid / f"{pid}_q{qid:02d}.wav"

    err_type = err_msg = None
    result_text, parsed_json, model_version, latency_s = "", None, None, None
    retries = 0
    for attempt in range(max_retries + 1):
        try:
            result_text, parsed_json, model_version, latency_s = _dispatch(
                spec, cond, system_prompt=system_prompt, audio_path=call_audio,
                text_input=text_input, max_out=max_out, response_schema=response_schema)
            err_type = err_msg = None
            break
        except Exception as e:  # transient or terminal — log the last one
            err_type = type(e).__name__
            err_msg = str(e)[:500]
            retries = attempt
            time.sleep(min(2 ** attempt, 8))

    return _make_record(
        spec=spec, cond=cond, cond_resp=cond_resp, run_index=run_index, pid=pid, cohort=cohort,
        qid=qid, prow=prow, clip_id=clip_id, audio_path=audio_path,
        audio_path_with_suffix=audio_path_with_suffix, text_input=text_input,
        transcript_source=transcript_source, system_prompt=system_prompt, max_out=max_out,
        response_schema=response_schema, result_text=result_text, parsed_json=parsed_json,
        model_version=model_version, latency_s=latency_s, error_type=err_type,
        error_message=err_msg, retry_count=retries)


if __name__ == "__main__":
    main()
