#!/usr/bin/env python3
"""SWE-bench rollout driver: run codewright on each instance, collect patches.

This script only PRODUCES patches. Grading is done separately by the official
SWE-bench harness (see evals/README.md) so that the pass/fail signal never comes
from code we wrote.

Per instance:
  1. start the official instance container (repo at base_commit, env prebuilt)
  2. drop in the prebuilt /opt/cw runtime (see evals/build_runtime.sh)
  3. run one non-interactive codewright turn against /testbed
  4. take `git diff` as the candidate patch

Resumable: an instance whose result file already exists is skipped.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import random
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

# The agent is told only what a human contributor would see: the issue text.
# No test names, no FAIL_TO_PASS, no hints -- leaking those invalidates the score.
PROMPT = """You are fixing a bug in a Python repository checked out at /testbed.

<issue>
{problem_statement}
</issue>

Fix the issue by editing the source code under /testbed.

Rules:
- Modify source files only. Do NOT add, edit, or delete any test files.
- Do NOT commit. Leave your changes uncommitted in the working tree.
- You may run the project's own tests to check your work.
- Stop when the issue is fixed, and summarize the change in one or two sentences.
"""


# Every docker call gets a deadline. A wedged daemon otherwise parks a worker
# forever, and a sweep silently loses throughput with nothing in the log.
_DEFAULT_CMD_TIMEOUT = 300


def sh(cmd: list[str], timeout: int | None = _DEFAULT_CMD_TIMEOUT) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def instance_image(instance: dict[str, Any]) -> str:
    """Authoritative image name, straight from the swebench package."""
    try:
        from swebench.harness.test_spec.test_spec import make_test_spec
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "swebench is not installed -- `pip install swebench`. It is the source "
            "of truth for instance image names and is needed for grading anyway."
        ) from exc
    return make_test_spec(instance).instance_image_key


def load_instances(dataset: str, split: str, subset: int | None, seed: int) -> list[dict]:
    from datasets import load_dataset

    rows = [dict(r) for r in load_dataset(dataset, split=split)]
    rows.sort(key=lambda r: r["instance_id"])  # deterministic base order
    if subset is not None and subset < len(rows):
        # Fixed seed + sorted base order == the same subset on every machine.
        rows = random.Random(seed).sample(rows, subset)
        rows.sort(key=lambda r: r["instance_id"])
    return rows


def run_one(instance: dict[str, Any], args: argparse.Namespace, out_dir: Path) -> dict:
    iid = instance["instance_id"]
    result_path = out_dir / "instances" / f"{iid}.json"
    if result_path.exists():
        try:
            return json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            # A truncated result from a hard kill. Re-run this instance rather
            # than letting the exception escape and take the sweep with it.
            pass

    record: dict[str, Any] = {
        "instance_id": iid,
        "status": "unknown",
        "patch": "",
        "agent": None,
        "wall_s": 0.0,
        "error": None,
    }
    started = time.monotonic()
    cid = f"cw-eval-{iid.replace('__', '-').replace('/', '-')}-{uuid.uuid4().hex[:6]}"
    container_started = False

    try:
        image = instance_image(instance)
        run_cmd = ["docker", "run", "-d", "--name", cid]
        if args.platform:
            run_cmd += ["--platform", args.platform]
        for key in ("CODEWRIGHT_API_KEY", "CODEWRIGHT_BASE_URL", "CODEWRIGHT_MODEL"):
            if os.environ.get(key):
                # `-e KEY` (no value) inherits from this process. Writing
                # `-e KEY=secret` would expose the API key in `ps` output and in
                # `docker inspect` for the life of the container.
                run_cmd += ["-e", key]
        run_cmd += [image, "sleep", "infinity"]

        proc = sh(run_cmd, timeout=600)
        if proc.returncode != 0:
            record["status"] = "container_error"
            record["error"] = proc.stderr.strip()[:2000]
            return record
        container_started = True

        # Inject the prebuilt runtime. `docker cp` + untar keeps every absolute
        # path inside /opt/cw valid, and touches nothing the graded tests use.
        if sh(["docker", "cp", str(args.runtime), f"{cid}:/tmp/cw-runtime.tgz"]).returncode != 0:
            record["status"] = "container_error"
            record["error"] = "failed to copy runtime into container"
            return record
        untar = sh(["docker", "exec", cid, "tar", "xzf", "/tmp/cw-runtime.tgz", "-C", "/"], timeout=300)
        if untar.returncode != 0:
            record["status"] = "container_error"
            record["error"] = f"untar failed: {untar.stderr.strip()[:500]}"
            return record

        prompt = PROMPT.format(problem_statement=instance["problem_statement"])
        agent_cmd = [
            "docker", "exec", cid,
            "/opt/cw/venv/bin/codewright", "run", prompt,
            "--workspace", "/testbed",
            # Blanket approval. Safe only because this is a throwaway container:
            # plain --full-auto still refuses to leave the workspace root or run
            # privileged commands, and an instance must not fail on a guard that
            # exists to protect a developer's laptop.
            "--permission-profile", "dangerous",
            "--full-auto",
            "--max-steps", str(args.max_steps),
            "--no-persist",
            # Skill distillation fires an extra, unaccounted LLM call on every
            # red->green transition -- i.e. exactly on the instances that succeed.
            # Off here so cost and behavior stay reproducible.
            "--no-distill",
            "--output-json", "/tmp/cw-summary.json",
        ]
        try:
            agent = sh(agent_cmd, timeout=args.timeout)
            record["agent_exit"] = agent.returncode
            record["agent_stderr_tail"] = agent.stderr.strip()[-2000:]
        except subprocess.TimeoutExpired:
            record["status"] = "agent_timeout"
            # The timeout killed our local `docker exec` client, not the agent
            # inside the container. It is still editing files. Stop it before
            # reading anything, or a half-written edit gets graded as the
            # candidate patch.
            sh(["docker", "kill", cid], timeout=60)

        summary = sh(["docker", "exec", cid, "cat", "/tmp/cw-summary.json"])
        if summary.returncode == 0 and summary.stdout.strip():
            try:
                record["agent"] = json.loads(summary.stdout)
            except json.JSONDecodeError:
                pass

        # codewright writes .codewright/audit.jsonl into the workspace on every
        # tool call, and `git add -A` below would stage it. Remove it first so
        # nothing we added can reach a graded patch.
        sh(["docker", "exec", cid, "rm", "-rf", "/testbed/.codewright"])
        # Stage before diffing: plain `git diff` omits untracked files, so any
        # fix that adds a source file would score as no_patch. `--cached` after
        # `add -A` is what the SWE-bench reference implementations do.
        sh(["docker", "exec", cid, "git", "-C", "/testbed", "add", "-A"], timeout=120)
        diff = sh(
            ["docker", "exec", cid, "git", "-C", "/testbed", "diff", "--cached"],
            timeout=120,
        )
        record["patch"] = diff.stdout if diff.returncode == 0 else ""

        if record["status"] == "agent_timeout":
            pass
        elif not record["patch"].strip():
            record["status"] = "no_patch"
        else:
            record["status"] = "patch_produced"

    except Exception as exc:  # keep one bad instance from killing the sweep
        record["status"] = "harness_error"
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if container_started:
            sh(["docker", "rm", "-f", cid], timeout=120)
        record["wall_s"] = round(time.monotonic() - started, 1)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = result_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        tmp.replace(result_path)  # atomic: a killed sweep never leaves half a file

    return record


def main() -> None:
    ap = argparse.ArgumentParser(description="Run codewright over a SWE-bench split.")
    ap.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--subset", type=int, default=None, help="evaluate a random subset of N instances")
    ap.add_argument("--seed", type=int, default=0, help="subset seed (report it alongside the score)")
    ap.add_argument("--runtime", type=Path, required=True, help="cw-runtime.tgz from build_runtime.sh")
    ap.add_argument("--out", type=Path, required=True, help="run directory")
    ap.add_argument("--model", default=None, help="sets CODEWRIGHT_MODEL in the container")
    ap.add_argument("--base-url", default=None, help="sets CODEWRIGHT_BASE_URL in the container")
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=1800, help="per-instance wall clock, seconds")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--platform", default=None, help="e.g. linux/amd64")
    args = ap.parse_args()

    if not args.runtime.exists():
        raise SystemExit(f"runtime not found: {args.runtime} -- run evals/build_runtime.sh first")
    if shutil.which("docker") is None:
        raise SystemExit("docker not found on PATH")
    if args.model:
        os.environ["CODEWRIGHT_MODEL"] = args.model
    if args.base_url:
        os.environ["CODEWRIGHT_BASE_URL"] = args.base_url
    if not os.environ.get("CODEWRIGHT_API_KEY"):
        raise SystemExit("set CODEWRIGHT_API_KEY -- the container has no ~/.codewright/config.toml")
    if not os.environ.get("CODEWRIGHT_MODEL"):
        raise SystemExit("set --model (or CODEWRIGHT_MODEL) -- otherwise the container defaults to gpt-4o-mini")

    instances = load_instances(args.dataset, args.split, args.subset, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({
        "dataset": args.dataset, "split": args.split,
        "subset": args.subset, "seed": args.seed, "n_instances": len(instances),
        "model": os.environ["CODEWRIGHT_MODEL"], "base_url": os.environ.get("CODEWRIGHT_BASE_URL"),
        "max_steps": args.max_steps, "timeout_s": args.timeout,
        "codewright_commit": sh(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"]
        ).stdout.strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2))

    print(f">> {len(instances)} instances, {args.workers} workers -> {args.out}")
    done = 0
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_one, inst, args, args.out): inst["instance_id"] for inst in instances}
        for fut in futures.as_completed(futs):
            rec = fut.result()
            done += 1
            print(f"[{done}/{len(instances)}] {rec['instance_id']:<40} {rec['status']:<16} {rec['wall_s']}s", flush=True)

    # Predictions in the exact shape the official grader expects.
    preds = args.out / "preds.jsonl"
    with preds.open("w") as fh:
        for path in sorted((args.out / "instances").glob("*.json")):
            rec = json.loads(path.read_text())
            fh.write(json.dumps({
                "instance_id": rec["instance_id"],
                "model_name_or_path": f"codewright+{os.environ['CODEWRIGHT_MODEL']}",
                "model_patch": rec["patch"],
            }) + "\n")
    print(f">> wrote {preds}")
    print(">> now grade with the official harness (see evals/README.md)")


if __name__ == "__main__":
    main()
