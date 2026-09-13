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
- **Disk: about 95 GB for a 50-instance subset**, and budget more for the full
  300. Two different numbers get quoted for this and only one of them is what
  you need free:

  | | 50 instances |
  |---|---|
  | downloaded over the network (compressed, shared layers counted once) | 23 GB |
  | **resident on disk after unpacking** | **94 GB** |
  | sum of what `docker images` prints per image | 246 GB |

  Images arrive compressed and are stored expanded, so the download figure
  understates the disk requirement by roughly 4x. The third number is the one to
  ignore: it counts every shared layer once per image. Grading reuses the same
  images and adds no more.
- `pip install swebench datasets` — **swebench >= 5**. It ships the instance
  image name as a dataset column and is what the CLI below assumes.
- Memory is the other limit. On a 16 GB Mac with an 8 GB Docker VM, `--workers 3`
  is comfortable; pre-pulling while containers ran was enough to get the pull
  killed by the OS memory manager.

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

| subset | bytes over the wire (deduped) | at 400 KB/s |
|---|---|---|
| 20 | 11.3 GB | ~8 h |
| 50 | 23.1 GB | ~17 h |

This is transfer, not disk — see Requirements for the ~94 GB those 50 occupy once
unpacked. Layers are shared across instances of the same repo, so the
deduplicated total is roughly 40% of the naive sum; but SWE-bench Lite is 114
django + 77 sympy out of 300, so a random subset still spans most repos. Grading
reuses the same images and downloads nothing further.

Treat the hours column as a floor. The 50 here actually took about 26 h of wall
clock, because throughput swung between 294 KB/s and 5.3 MB/s over the run and
the five matplotlib images are 2.9 GB each. A proxy helped, measured back to back
on the same link: 158 KB/s direct against 418 KB/s through it — but see the
warning below about what a proxy does to container traffic.

`docker pull` fetches up to 3 layers at once; on a bandwidth-limited link that is
*slower* than one stream, since the layers just split the same pipe. If pulls
crawl, check for another download competing before blaming the registry.

### A proxy speeds up pulls and can cut off the agent

Worth knowing before reaching for one. Docker Desktop, with a system proxy set,
transparently proxies **container** traffic as well as its own image pulls — the
containers carry no proxy environment variable, so nothing in them shows it.
Measured here with an LLM gateway on a private address:

| | image pulls | container → gateway |
|---|---|---|
| proxy off | 158 KB/s | reachable, 60 ms |
| proxy on | 418 KB/s | **fails, 5 s timeout, 3/3** |

The gateway resolved to `10.88.1.54`, and the proxy forwarded it to an exit node
where no such host exists. The host itself could still reach it, so a check run
from the shell says everything is fine; only a request from inside a container
shows the break.

So either turn the proxy off before rolling out, or give the proxy a direct rule
for the gateway and verify from inside a container, not from the host:

```bash
docker run --rm --platform linux/amd64 alpine:3.20 sh -c \
  'apk add -q curl && curl -s -o /dev/null -w "%{http_code}\n" \
   https://your-gateway/v1/models'
```

401 means reachable. `000` means the proxy ate it.

### Pulling is memory-bound too

Docker extracts layers inside its VM, and that VM's memory is the host's. A
400 MB compressed layer expands to well over a GB, and the default
`max-concurrent-downloads: 3` stacks several of those at once — enough for the
OS to kill the pull outright on a 16 GB machine. Since concurrent downloads are
also *slower* here, there is no reason not to serialize:

```json
// ~/.docker/daemon.json  (restart Docker after editing)
{ "max-concurrent-downloads": 1 }
```

Pull one image at a time, and check free memory before each one rather than
trusting a long unattended loop to survive.

### Finish the pulls before starting the sweep

Not an optimization — a correctness requirement. The same instance, run twice:

| link | model calls | outcome | wall |
|---|---|---|---|
| saturated by a concurrent `docker pull` | died after 7 | 3 attempts, all gateway errors, `infra_error` | 894 s |
| idle | 11, ran to completion | `patch_produced`, resolved | 123 s |

A gateway that proxies a model streams the response through itself. When our end
cannot drain the socket fast enough, its downstream write blocks, it stops
reading upstream, and its own idle timer fires:

    provider error 102503: passthrough stream idle timeout after 120s
    waiting for next chunk

which arrives looking like a provider fault. Pulling images while agents run will
manufacture `infra_error` at a rate that has nothing to do with either the agent
or the gateway.

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

The buckets split into two groups, and the split is the point: one group is the
model's record, the other is the harness's. Reading a harness failure as a model
failure is the easiest way to publish a number that is too low.

**The model's record** — these are the ones a better model or a better prompt
would move:

| bucket | meaning |
|---|---|
| `resolved` | the official harness ran the tests and they passed |
| `tests_failed` | the patch applied, the tests stayed red. A real miss: the agent finished, was confident, and was wrong |
| `no_patch` | the agent ran to completion and edited nothing. Usually it never found the right file, or decided nothing was wrong |
| `step_budget_exhausted` | hit `--max-steps`. Either raise it, or the agent is going in circles — check `tool_calls` to tell those apart |
| `agent_error` | the turn died on something the model caused, such as a tool call it malformed |

**The harness's record** — these say nothing about the model and must be near
zero before a score means anything:

| bucket | meaning |
|---|---|
| `infra_error` | the gateway, the network or docker gave out, and `--retries` did not recover it |
| `agent_timeout` | hit `--timeout` wall clock. Ambiguous: a slow gateway and a rambling agent look the same, so read `model_calls` |

A worked example, from the 50-instance run in this directory. Eight instances did
not resolve, and the first pass through the buckets read:

    infra_error 3 · agent_error 2 · no_patch 2 · tests_failed 1

which suggests the model failed five times. It failed once. The other four had
been cut off by the gateway:

| instance | first read as | what actually happened |
|---|---|---|
| `django-16408` | `agent_error` | 429 `concurrent request limit exceeded: 100`, after producing 51 lines |
| `matplotlib-22835` | `agent_error` | 429 rate limit, after producing 28 lines |
| `sphinx-8435` | `no_patch` | 429 concurrent limit; the agent never got to work |
| `sympy-14308` | `no_patch` | 429 rate limit on all three attempts |

`no_patch` and `agent_error` are where a reader looks for the model's mistakes,
so a throttling error landing there is not a cosmetic mislabel. The cause was
`bucket()` reading `status` alone: those four ran under a driver whose retry
markers did not yet cover throttling, so their status was decided before the
error text was ever consulted. It now reads the recorded `errors` as well, which
also keeps old run directories readable. Corrected, the same run is

    infra_error 7 · tests_failed 1

— one model miss, seven gateway failures.

The lesson generalises: **check the bucket table against the raw `errors` before
quoting either number.** A throttled gateway inflates the failure side and
deflates the score, and it does so silently.

### Retries

codewright does not retry LLM calls: one bad chunk from the provider ends the
turn. Observed against this gateway:

    provider error 102503: passthrough stream idle timeout after 120s
    waiting for next chunk

That is not the model failing the task, so `run_agent.py` re-runs the instance in
a fresh container — `--retries 2` by default. It retries **only** gateway,
transport and docker failures; a model that ran and did not solve the instance is
never retried, because retrying that is what turns pass@1 into best-of-N. The
matched patterns are `_INFRA_ERROR_MARKERS`, and every retry is kept in the
record's `infra_retries` and reported, so a run that fought the gateway all night
cannot later read as a clean one.

## A run that was actually done

`evals/runs/lite50/`, 2026-09-13, codewright at `9136b51`:

    codewright + deepseek-v4.1-flash on SWE-bench/SWE-bench_Lite
    (random subset, seed=0): 84.0% resolved
    (pass@1, n=50, 95% CI [71.5, 91.7])

    resolved 42 · infra_error 7 · tests_failed 1
    19.1M tokens · 5.6 h of container time · 6.7 min/instance

`--max-steps 60 --timeout 1800 --workers 3`, graded by `swebench eval` at 5.0.2.

The run dir itself is gitignored. The write-up of it is not — `evals/reports/`
holds the same result in four forms, two languages each:

| | English | 中文 |
|---|---|---|
| full report, opens in a browser | [`lite50.en.html`](reports/lite50.en.html) | [`lite50.zh.html`](reports/lite50.zh.html) |
| short version, reads in the repo | [`lite50.en.md`](reports/lite50.en.md) | [`lite50.zh.md`](reports/lite50.zh.md) |

The rest of this section is the same material in condensed form.

### What was checked before believing it

84% is above the range these agents usually post on Lite, so it was worth trying
to disprove before quoting:

- **Did any patch touch a test file?** None did. (The official harness restores
  test files before grading anyway, but a patch that tries says something about
  the run.)
- **Are the patches real fixes or lucky edits?** Three resolved instances were
  read against the reference patch. `astropy-12907`, `pylint-5859` and
  `sympy-24213` matched the reference character for character.
- **Does the run metadata describe the run?** It did not. A final single-instance
  pass had overwritten `config.json` with `n_instances: 1`, so the report was
  omitting the subset and seed. Restored, with a note in the file.

### Caveats that belong next to the number

- **It was not one clean sweep.** The first pass left 9 instances on gateway
  failures. The 5 that had produced no patch were re-run; the 4 that had produced
  a patch were left alone, since re-rolling a real candidate is how a pass@1
  number turns into best-of-N. Re-running recovered 4. Only infrastructure
  failures were ever re-run — an unsolved instance never was, and the tests in
  `tests/` hold that line.
- **The gateway was unhealthy throughout.** 25 of 50 instances hit at least one
  gateway failure, 48 retries in total, and `infra_error` finished at 14% rather
  than the ~0% a quotable run wants. The true score is probably *higher* than
  84% — seven instances never got a fair attempt — but the noise is real and
  belongs in any writeup.
- **n=50 is a wide interval.** ±10 points at this sample size. 84% and 74% are
  not distinguishable here.

### Reproducing it

```bash
./evals/build_runtime.sh

export CODEWRIGHT_API_KEY=... CODEWRIGHT_BASE_URL=...
evals/.venv/bin/python -u evals/swebench/run_agent.py \
  --dataset SWE-bench/SWE-bench_Lite --subset 50 --seed 0 \
  --runtime evals/_runtime/cw-runtime.tgz \
  --model deepseek-v4.1-flash \
  --max-steps 60 --timeout 1800 --workers 3 --retries 2 \
  --out evals/runs/<name>

evals/.venv/bin/swebench eval SWE-bench/SWE-bench_Lite \
  -p evals/runs/<name>/preds.jsonl --run-id <name> \
  --report-dir evals/runs/<name> -j 3

evals/.venv/bin/python evals/swebench/report.py \
  --run evals/runs/<name> \
  --grading evals/runs/<name>/codewright+<model>.<name>.json
```

`--seed 0` picks the same 50 instances on any machine: the split is sorted by
`instance_id` before sampling, so the subset does not depend on dataset order.

Pull the images first and let the pulls finish — see above for why running both
at once manufactures failures. Re-running the driver is safe: an instance with a
result file is skipped, so a killed sweep resumes where it stopped. To force one
instance to re-run, delete its file under `instances/` and pass `--instance`.

## Known limits of these numbers

Read these before quoting a score.

- **A flaky gateway is the dominant source of error, not the harness.** In the
  run above it cost 14% of the sample and 48 retries. Before quoting a score,
  check `infra_error` and the retry line; if either is high, the number measures
  the endpoint. Watch for throttling in particular — `rate limit`,
  `usage limit`, `concurrent request limit` — since those arrive wrapped in a
  429 whose outer type says `rate_limit_error` while the inner message says
  something else entirely.
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
