# -*- coding: utf-8 -*-
"""问题3 本地多局对照：同一套 Dog 策略，随机全向源。"""
from __future__ import annotations

import json
import os
import sys
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "q1"))
sys.path.insert(0, str(HERE.parent / "q2"))

from local_world import LocalSimulator  # noqa: E402
from run_q3 import Dog  # noqa: E402


def one_trial(seed: int, n_src=None):
    rng = np.random.default_rng(seed)
    sim = LocalSimulator(rng, n_src=n_src)
    dog = Dog(sim)
    t0 = time.time()
    with redirect_stdout(StringIO()):
        summary = dog.run()
    wall = time.time() - t0
    n_true = sim.n_true
    alive = [ch for ch, s in sim.sources.items() if s["alive"]]
    n_cleared = int(summary["n_cleared"])
    avg = (summary["virtual_time_s"] / n_cleared) if n_cleared else None
    complete = (n_cleared == n_true) and (not alive) and (not summary["failed"])
    return {
        "seed": seed,
        "n_true": n_true,
        "n_cleared": n_cleared,
        "n_empty": len(summary["empty"]),
        "failed": summary["failed"],
        "left": summary["left"],
        "alive": alive,
        "ratio": n_cleared / n_true if n_true else 0.0,
        "virtual_s": summary["virtual_time_s"],
        "avg_s": avg,
        "complete": complete,
        "wall_s": wall,
        "cleared": summary["cleared"],
    }


def main():
    n = 24
    rows = []
    t0 = time.time()
    for i in range(n):
        row = one_trial(20260911 + i)
        rows.append(row)
        print(
            f"[{i+1:02d}/{n}] n={row['n_true']:2d} cleared={row['n_cleared']:2d} "
            f"empty={row['n_empty']:2d} failed={row['failed']} left={row['left']} "
            f"T={row['virtual_s']:.0f}s avg={row['avg_s']:.0f}s  "
            f"{'OK' if row['complete'] else 'FAIL'}"
        )
    ok = [r for r in rows if r["complete"]]
    avgs = [r["avg_s"] for r in rows if r["avg_s"]]
    Ts = [r["virtual_s"] for r in rows]
    ratios = [r["ratio"] for r in rows]
    out = {
        "n_trials": n,
        "n_complete": len(ok),
        "complete_rate": len(ok) / n,
        "mean_ratio": float(np.mean(ratios)),
        "mean_avg_s": float(np.mean(avgs)),
        "median_avg_s": float(np.median(avgs)),
        "p10_avg_s": float(np.percentile(avgs, 10)),
        "p90_avg_s": float(np.percentile(avgs, 90)),
        "mean_T": float(np.mean(Ts)),
        "fail_seeds": [r["seed"] for r in rows if not r["complete"]],
        "rows": rows,
        "elapsed_wall_s": time.time() - t0,
    }
    dest = HERE / "logs" / "mc_local.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("----")
    print(
        f"complete {out['n_complete']}/{n}  mean ratio {out['mean_ratio']:.3f}  "
        f"avg {out['mean_avg_s']:.0f}s (p10 {out['p10_avg_s']:.0f}, p90 {out['p90_avg_s']:.0f})  "
        f"mean T {out['mean_T']:.0f}s"
    )
    print("wrote", dest)


if __name__ == "__main__":
    main()
