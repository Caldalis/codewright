# Codewright on SWE-bench Lite

[中文](lite50.zh.md) · [full report (HTML)](lite50.en.html)

**84.0% resolved — 42 of 50, 95% CI [71.5%, 91.7%]**

| | |
|---|---|
| dataset | `SWE-bench/SWE-bench_Lite` · test · n=50, random subset, seed 0 |
| agent | codewright @ `9136b51` |
| model | `deepseek-v4.1-flash` |
| settings | pass@1 · max_steps 60 · timeout 1800s · workers 3 |
| graded by | swebench 5.0.2, official harness |

The agent saw the issue text and nothing else — no test names, no reference
diff. The official harness applied the test patch and decided pass or fail.

Quotable form:

> codewright + deepseek-v4.1-flash on SWE-bench Lite (random subset, seed=0):
> **84.0% resolved** (pass@1, n=50, 95% CI [71.5, 91.7])

## Where the eight failures went

| bucket | n | |
|---|---|---|
| `resolved` | 42 | 84% |
| `infra_error` | 7 | 14% — gateway, not the model |
| `tests_failed` | 1 | 2% |

Seven of the eight were cut off by the gateway. **One genuine model miss:**
`django__django-13158` ran to completion over 34 calls, produced a 13-line
patch, and the tests stayed red.

<details>
<summary>The other seven, and the error that ended each</summary>

| instance | ended on | got as far as |
|---|---|---|
| `django__django-12284` | 3× dropped connection, then upstream 500 | 9 calls, no patch |
| `sympy__sympy-13915` | stream idle timeout / upstream 500, alternating over 4 attempts | 15 calls, no patch |
| `sympy__sympy-15346` | same two, alternating over 4 attempts | 2 calls, no patch |
| `django__django-16408` | `concurrent request limit exceeded: 100` | 17 calls, **51-line patch, truncated and graded** |
| `matplotlib__matplotlib-22835` | `Rate limit exceeded` | 19 calls, **28-line patch, truncated and graded** |
| `sympy__sympy-14308` | 500, dropped connection, then `Rate limit exceeded` | 13 calls, no patch |
| `sphinx-doc__sphinx-8435` | `concurrent request limit exceeded: 100` | 9 calls, no patch |

</details>

## By repository

No repository collapsed. The largest carry the failures because they carry the
instances — Lite is 114 django and 77 sympy out of 300.

| repository | resolved | |
|---|---|---|
| django/django | 13 / 16 | 81% |
| sympy/sympy | 11 / 14 | 79% |
| matplotlib/matplotlib | 4 / 5 | 80% |
| pytest-dev/pytest | 4 / 4 | 100% |
| sphinx-doc/sphinx | 2 / 3 | 67% |
| pylint-dev/pylint | 2 / 2 | 100% |
| scikit-learn | 2 / 2 | 100% |
| astropy · seaborn · requests · xarray | 4 / 4 | 100% |

## What it cost

18.8M input tokens · 300k output · 5.6 h of container time · 1,160 tool calls.

Per instance the median is 282k tokens, 5 minutes, 20 model calls — but the
tail is long: 14k tokens / 42 s at the cheapest, 1.56M / 22 min at the most
expensive. Budget from the tail.

## Caveats that travel with the number

- **Not one clean sweep.** Nine instances ended the first pass on gateway
  failures. The five with no patch were re-run (four recovered); the four that
  had produced a patch were left alone — re-rolling a real candidate turns
  pass@1 into best-of-N. Only infrastructure failures were ever re-run.
- **The endpoint was unhealthy throughout.** 48 retries across 25 instances.
  `infra_error` finished at 14%, where a quotable run wants roughly zero. The
  true score is plausibly *higher* than 84%, since seven instances never got a
  fair attempt — but the noise is real.
- **n=50 is ±10 points.** Quoting 84% without the interval overstates what
  fifty instances can tell you.

Before trusting the number: no patch touched a test file (all 50 checked), and
three resolved patches matched the reference diff character for character. Two
reporting bugs were found and fixed in the process — see the
[HTML report](lite50.en.html) for both.

## Reproducing it

`--seed 0` picks the same fifty instances anywhere: the split is sorted by
`instance_id` before sampling, so the subset does not depend on dataset order.

```sh
./evals/build_runtime.sh

evals/.venv/bin/python -u evals/swebench/run_agent.py \
  --dataset SWE-bench/SWE-bench_Lite --subset 50 --seed 0 \
  --runtime evals/_runtime/cw-runtime.tgz \
  --model deepseek-v4.1-flash \
  --max-steps 60 --timeout 1800 --workers 3 --retries 3 \
  --out evals/runs/<name>

evals/.venv/bin/swebench eval SWE-bench/SWE-bench_Lite \
  -p evals/runs/<name>/preds.jsonl --run-id <name> \
  --report-dir evals/runs/<name> -j 3

evals/.venv/bin/python evals/swebench/report.py \
  --run evals/runs/<name> \
  --grading evals/runs/<name>/codewright+<model>.<name>.json
```

Pull the images first and let the pulls finish — a saturated link makes the
gateway's proxy time out mid-stream and manufactures failures that look like the
agent's. Budget ~94 GB of disk for fifty instances; the 23 GB that crosses the
wire is compressed. See [`evals/README.md`](../README.md) for the full setup.

---

Per-instance records, patches and the grading report are in `evals/runs/lite50/`,
which is not tracked in git.
