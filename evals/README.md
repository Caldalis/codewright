# Evaluating Codewright

Task success rate on a public benchmark, produced by a pipeline anyone can re-run.

Two rules this harness is built around:

1. **We never grade ourselves.** The rollout driver only produces patches; every
   pass/fail comes from the official SWE-bench harness.
2. **The agent never sees the tests.** The prompt contains the issue text and
   nothing else — no test names, no `FAIL_TO_PASS`, no hints.

## Requirements

- Docker. Linux x86_64 is fastest, but an arm64 Mac works: every instance image
  is x86_64, and measured on an M-series Mac the amd64 emulation costs only
  about 15% over native once images are cached. (The 20s for a cold
  `docker run` is image pull plus first-start, not steady-state emulation.)
- ~120 GB free disk for a full SWE-bench Lite sweep; a 50-instance subset is
  closer to 20 GB. Instance images of the same repo share most of their layers.
- `pip install swebench datasets` — **swebench >= 5**. It ships the instance
  image name as a dataset column and is what the CLI below assumes.
- Roughly 8 GB of memory per concurrent instance; testing threads in these repos
  is the peak. On a small Docker VM, keep `--workers` low.

## 1. Build the agent runtime (once)

SWE-bench images ship whatever Python the target repo needs (often 3.6–3.11);
codewright needs 3.12. Installing 3.12 into the instance environment would change
what the graded tests run against, so instead we bake a standalone CPython 3.12 +
codewright into a relocatable `/opt/cw` tree and drop it into each container.

```bash
./evals/build_runtime.sh              # -> evals/_runtime/cw-runtime.tgz
```

Built for `linux/amd64` by default, which is what the instance containers need —
an arm64 build would fail inside them with `exec format error`. Override with
`CW_RUNTIME_PLATFORM` only if you know your images differ.

## 2. Roll out

```bash
export CODEWRIGHT_API_KEY=...            # the container has no config.toml:
export CODEWRIGHT_BASE_URL=...           # everything comes from these env vars
                                         # (passed as `-e KEY`, so they are
                                         #  inherited, never written into argv)

python evals/swebench/run_agent.py \
  --dataset SWE-bench/SWE-bench_Lite \
  --subset 50 --seed 0 \
  --runtime evals/_runtime/cw-runtime.tgz \
  --model <model-id> \
  --max-steps 60 --timeout 1800 --workers 8 \
  --out evals/runs/lite50-$(date +%Y%m%d)
```

Start with `--subset 20` to measure cost and wall time per instance, then
extrapolate before committing to the full split. The sweep is resumable — an
instance with a result file is skipped, so a killed run continues where it left off.

Output: `preds.jsonl` plus one JSON record per instance (patch, agent summary,
token counts, failure status).

## 3. Grade with the official harness

```bash
evals/.venv/bin/swebench eval SWE-bench/SWE-bench_Lite \
  -p evals/runs/<run>/preds.jsonl \
  --run-id <run> \
  --report-dir evals/runs/<run> \
  -j 4
```

This is the swebench >= 5 CLI; the 4.x `python -m swebench.harness.run_evaluation`
form is gone. Check `swebench eval --help` against your installed version, since
the flags do move between releases. Grading pulls its own evaluation images and
runs the same containers, so budget disk and memory for it too.

The report lands as `<report-dir>/<model_name_or_path>.<run-id>.json` — pass that
file to `report.py` as `--grading`.

## 4. Report

```bash
python evals/swebench/report.py \
  --run evals/runs/<run> \
  --grading <official-report>.json
```

Prints the resolve rate with a 95% Wilson interval, plus a failure-attribution
table. **Always publish the interval on a subset** — at n=50 the interval is
roughly ±13 points, so a bare percentage is close to meaningless.

## Budget the download, not just the compute

Instance images dominate the setup cost, and the constraint is usually bandwidth
rather than CPU. Measure before committing to a subset size:

| subset | image bytes (deduped) | at 400 KB/s |
|---|---|---|
| 20 | 11.3 GB | ~8 h |
| 50 | 23.1 GB | ~17 h |

Layers are shared across instances of the same repo, so the deduplicated total is
roughly 40% of the naive sum — but SWE-bench Lite is 114 django + 77 sympy out of
300, so a random subset still spans most repos. Grading reuses the same images,
so it costs no extra download.

`docker pull` fetches up to 3 layers at once; on a bandwidth-limited link that is
*slower* than one stream, since the layers just split the same pipe. If pulls
crawl, check for another download competing before blaming the registry.

## Reporting honestly

State together: dataset + split, `n` (and the subset seed), model id, `pass@1`,
`max_steps`, date, codewright commit, and cost per instance. A number without
these is not reproducible.

Three things that would invalidate the result:

- reporting the best of several runs instead of a pre-declared one
- feeding test names or expected diffs into the prompt
- quoting a subset score without `n` and the confidence interval

## Failure buckets

`report.py` puts every non-resolved instance in exactly one bucket. This table is
worth more than the headline number, because it says what to fix next:

| bucket | meaning |
|---|---|
| `resolved` | official harness confirmed the fix |
| `tests_failed` | patch applied, tests still red — a model/reasoning miss |
| `no_patch` | agent finished without editing anything |
| `step_budget_exhausted` | hit `--max-steps` — raise it, or the agent is looping |
| `agent_timeout` | hit the wall clock |
| `agent_error` | turn aborted on an error |
| `infra_error` | container/harness problem — must be ~0, or the score is noise |

`infra_error` and `agent_timeout` measure the harness, not the agent. If they are
not near zero, fix them before quoting any number.

## Known limits of these numbers

Read these before quoting a score.

- **`--max-steps` is per agent, per turn, not a global budget.** A run that
  spawns sub-agents gets N steps *each*. The real ceiling on a runaway instance
  is `--timeout`, which is enforced by the harness, not by the agent.
- **Sub-agent tokens are not counted.** `_watch` consumes a child session's
  events on its own queue, so a child's `EvTokenCount` never reaches the run
  summary. Cost is under-reported whenever the agent spawns.
- **Token counts depend on the provider reporting usage.** The chat-completions
  adapter asks for it via `stream_options: {include_usage: true}`, an
  OpenAI-specific option many compatible gateways ignore. If yours does, every
  token field reads 0 with no error — so verify with one real call before a
  sweep. (Checked against `llm-center.modelbest.co` with `glm-5.3-flash`: usage
  is reported, `input_tokens`/`output_tokens` come back non-zero.)
- **The rollout runs with blanket approval.** `run_agent.py` passes
  `--permission-profile dangerous --full-auto`, which allows every action with
  no exceptions, so no instance can fail on a guard meant to protect a
  developer machine. **The container is the only sandbox here** — never use that
  pairing outside one.

  Plain `--full-auto` is the safe default and is what non-benchmark callers
  should use: recoverable work inside the workspace is allowed, while leaving
  the workspace root and privileged or unrecoverable commands (`sudo`, `dd`,
  `curl | bash`, `git push --force`) are denied. Two earlier holes in that deny
  list are closed and covered by tests: a hard flag hidden inside a command
  substitution (`echo $(sudo rm -rf /etc)`) is now analysed recursively, and a
  command `shlex` cannot parse now fails closed instead of resolving to allow.

  Even so, a deny list is not a sandbox. Do not rely on it alone for a workspace
  you care about.

## Notes

- `--full-auto` is required: unattended runs cannot answer approval prompts.
  It is safe here only because everything happens inside a throwaway container.
- `--no-distill` is required for a clean number. Skill distillation makes an
  extra LLM call on every red->green test transition — exactly the successful
  instances — and those tokens are not counted in the run summary. Since each
  instance gets a fresh container, distilled skills can never carry across tasks
  anyway, so leaving it on only adds cost and variance.
- Codewright writes `.codewright/audit.jsonl` into the workspace on every tool
  call. The driver deletes it before taking `git diff`, so it can never leak into
  a graded patch.
- Per-instance cost/tokens come from `codewright run --output-json`.
