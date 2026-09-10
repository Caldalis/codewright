# Evaluating Codewright

Task success rate on a public benchmark, produced by a pipeline anyone can re-run.

Two rules this harness is built around:

1. **We never grade ourselves.** The rollout driver only produces patches; every
   pass/fail comes from the official SWE-bench harness.
2. **The agent never sees the tests.** The prompt contains the issue text and
   nothing else — no test names, no `FAIL_TO_PASS`, no hints.

## Requirements

- Linux **x86_64** with Docker. Do not run this on an arm64 Mac: the official
  instance images are x86-only and emulation is too slow to be usable.
- ~120 GB free disk for SWE-bench Lite instance images.
- `pip install swebench datasets`

## 1. Build the agent runtime (once)

SWE-bench images ship whatever Python the target repo needs (often 3.6–3.11);
codewright needs 3.12. Installing 3.12 into the instance environment would change
what the graded tests run against, so instead we bake a standalone CPython 3.12 +
codewright into a relocatable `/opt/cw` tree and drop it into each container.

```bash
./evals/build_runtime.sh              # -> evals/_runtime/cw-runtime.tgz
```

Build it on the same architecture you evaluate on.

## 2. Roll out

```bash
export CODEWRIGHT_API_KEY=...            # the container has no config.toml:
export CODEWRIGHT_BASE_URL=...           # everything comes from these env vars
                                         # (passed as `-e KEY`, so they are
                                         #  inherited, never written into argv)

python evals/swebench/run_agent.py \
  --dataset princeton-nlp/SWE-bench_Lite \
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
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path evals/runs/<run>/preds.jsonl \
  --max_workers 8 \
  --run_id <run>
```

Pin the exact flags against the `swebench` version you install — the CLI does
change between releases.

## 4. Report

```bash
python evals/swebench/report.py \
  --run evals/runs/<run> \
  --grading <official-report>.json
```

Prints the resolve rate with a 95% Wilson interval, plus a failure-attribution
table. **Always publish the interval on a subset** — at n=50 the interval is
roughly ±13 points, so a bare percentage is close to meaningless.

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
  token field reads 0 with no error. Verify with one real call before a sweep.
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
