# Codewright 的 SWE-bench Lite 成绩

[English](lite50.en.md) · [完整报告（HTML）](lite50.zh.html)

**解决率 84.0% —— 50 个里过了 42 个，95% 置信区间 [71.5%, 91.7%]**

| | |
|---|---|
| 数据集 | `SWE-bench/SWE-bench_Lite` · test · n=50，随机抽样，seed 0 |
| Agent | codewright @ `9136b51` |
| 模型 | `deepseek-v4.1-flash` |
| 配置 | pass@1 · max_steps 60 · timeout 1800s · workers 3 |
| 判定 | swebench 5.0.2 官方 harness |

Agent 只看得到 issue 正文，看不到测试名，也看不到参考补丁。官方 harness 打上
test patch 之后决定过与不过。

可以直接引用的写法：

> codewright + deepseek-v4.1-flash 在 SWE-bench Lite（随机子集，seed=0）上
> **解决率 84.0%**（pass@1，n=50，95% CI [71.5, 91.7]）

## 没过的 8 个去哪了

| 分类 | 个数 | |
|---|---|---|
| `resolved` | 42 | 84% |
| `infra_error` | 7 | 14% —— 网关问题，与模型无关 |
| `tests_failed` | 1 | 2% |

8 个里有 7 个是被网关掐断的。**真正算模型失手的只有 1 个：**
`django__django-13158` 完整跑完 34 次调用，产出 13 行补丁，测试没过。

<details>
<summary>另外 7 个，以及真正结束它们的那条报错</summary>

| 实例 | 结束于 | 跑到哪一步 |
|---|---|---|
| `django__django-12284` | 连断 3 次，第 4 次 upstream 500 | 9 次调用，无补丁 |
| `sympy__sympy-13915` | 流卡死 / upstream 500 交替，试了 4 次 | 15 次调用，无补丁 |
| `sympy__sympy-15346` | 同样两种交替，试了 4 次 | 2 次调用，无补丁 |
| `django__django-16408` | `concurrent request limit exceeded: 100` | 17 次调用，**已写 51 行，截断后参与判定** |
| `matplotlib__matplotlib-22835` | `Rate limit exceeded` | 19 次调用，**已写 28 行，截断后参与判定** |
| `sympy__sympy-14308` | 500、连接断开，最后 `Rate limit exceeded` | 13 次调用，无补丁 |
| `sphinx-doc__sphinx-8435` | `concurrent request limit exceeded: 100` | 9 次调用，无补丁 |

</details>

## 按仓库拆

没有哪个仓库塌方。最大的几个之所以承担了全部失败，只是因为它们本来就占了大多数
实例 —— Lite 的 300 个里 django 有 114 个，sympy 有 77 个。

| 仓库 | 解决 | |
|---|---|---|
| django/django | 13 / 16 | 81% |
| sympy/sympy | 11 / 14 | 79% |
| matplotlib/matplotlib | 4 / 5 | 80% |
| pytest-dev/pytest | 4 / 4 | 100% |
| sphinx-doc/sphinx | 2 / 3 | 67% |
| pylint-dev/pylint | 2 / 2 | 100% |
| scikit-learn | 2 / 2 | 100% |
| astropy · seaborn · requests · xarray | 4 / 4 | 100% |

## 成本

输入 1880 万 token · 输出 30 万 · 容器时间 5.6 小时 · 工具调用 1160 次。

单实例中位数是 28.2 万 token、5 分钟、20 次模型调用 —— 但尾部很长：最便宜的
1.4 万 token / 42 秒，最贵的 156 万 token / 22 分钟。做预算按尾部算。

## 必须跟着这个数一起引用的前提

- **不是一遍干净跑完的。** 第一轮结束时有 9 个实例卡在网关故障上。没产出补丁的
  5 个做了重跑（救回 4 个）；已经产出补丁的 4 个没动 —— 把真实候选重新摇一次，
  pass@1 就变成 best-of-N 了。只有基础设施故障被重跑过。
- **整个过程里网关一直不健康。** 25 个实例上重试了 48 次，`infra_error` 停在
  14%，而一个能拿出去引用的跑批应该接近 0。真实分数有理由*更高*，因为有 7 个
  实例根本没得到一次公平的尝试 —— 但这个噪声是真实存在的。
- **n=50 意味着 ±10 个百分点。** 只报 84% 而不报区间，等于夸大了 50 个实例能
  说明的东西。

相信这个数之前查过：没有任何补丁改过测试文件（50 个全查），另外抽了 3 个已解决
实例跟参考补丁逐字对照，完全一致。过程中还发现并修掉了 2 个报告侧的 bug，
细节见 [HTML 版报告](lite50.zh.html)。

## 怎么复现

`--seed 0` 在任何机器上都会选中同样的 50 个实例：抽样前先按 `instance_id` 排序，
所以子集不依赖数据集本身的顺序。

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

开跑之前先把镜像拉完，并且要等它真的拉完 —— 链路一旦被占满，网关的代理就会在流
中途超时，制造出看起来像 agent 出错的故障。50 个实例要准备约 94 GB 磁盘；过网的
那 23 GB 是压缩后的体积。完整环境准备见 [`evals/README.md`](../README.md)。

---

逐实例记录、补丁和判定报告在 `evals/runs/lite50/`，该目录不纳入 git。
