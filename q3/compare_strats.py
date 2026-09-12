# -*- coding: utf-8 -*-
"""问题3 本地策略对照：同一批随机世界，换第二点赌注 / 是否顺路复用。"""
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
sys.path.insert(0, str(HERE.parent / "paper"))

from local_world import LocalSimulator  # noqa: E402
from run_q3 import Dog  # noqa: E402
import figstyle as fs  # noqa: E402

STRATS = [
    {"id": "time_reuse", "label": "时间+复用", "bet_t": 500.0, "bet_h": 250.0, "reuse_stops": True},
    {"id": "time_bare", "label": "时间(无复用)", "bet_t": 500.0, "bet_h": 250.0, "reuse_stops": False},
    {"id": "minimax", "label": "minimax", "bet_t": 550.0, "bet_h": 498.0, "reuse_stops": True},
    {"id": "conservative", "label": "保守", "bet_t": 750.0, "bet_h": 450.0, "reuse_stops": True},
]


def one_trial(seed: int, strat: dict):
    rng = np.random.default_rng(seed)
    sim = LocalSimulator(rng)
    dog = Dog(
        sim,
        bet_t=strat["bet_t"],
        bet_h=strat["bet_h"],
        reuse_stops=strat["reuse_stops"],
    )
    t0 = time.time()
    with redirect_stdout(StringIO()):
        summary = dog.run()
    n_true = sim.n_true
    alive = [ch for ch, s in sim.sources.items() if s["alive"]]
    n_cleared = int(summary["n_cleared"])
    avg = (summary["virtual_time_s"] / n_cleared) if n_cleared else None
    complete = (n_cleared == n_true) and (not alive) and (not summary["failed"])
    return {
        "seed": seed,
        "strat": strat["id"],
        "n_true": n_true,
        "n_cleared": n_cleared,
        "n_empty": len(summary["empty"]),
        "failed": summary["failed"],
        "alive": alive,
        "virtual_s": summary["virtual_time_s"],
        "avg_s": avg,
        "complete": complete,
        "wall_s": time.time() - t0,
    }


def summarize(rows):
    ok = [r for r in rows if r["complete"]]
    avgs = [r["avg_s"] for r in rows if r["avg_s"]]
    Ts = [r["virtual_s"] for r in rows]
    return {
        "n": len(rows),
        "n_complete": len(ok),
        "complete_rate": len(ok) / len(rows) if rows else 0.0,
        "mean_avg_s": float(np.mean(avgs)) if avgs else None,
        "median_avg_s": float(np.median(avgs)) if avgs else None,
        "mean_T": float(np.mean(Ts)) if Ts else None,
        "fail_seeds": [r["seed"] for r in rows if not r["complete"]],
    }


def fig_compare(by_strat, out_png):
    fs.apply()
    fig, ax = fs.new_fig()
    labels = [s["label"] for s in STRATS]
    data = []
    means = []
    for s in STRATS:
        avgs = [r["avg_s"] for r in by_strat[s["id"]] if r["avg_s"]]
        data.append(avgs)
        means.append(float(np.mean(avgs)) if avgs else 0.0)
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55)
    for box, med in zip(bp["boxes"], bp["medians"]):
        box.set_facecolor(fs.FILL_IO)
        box.set_edgecolor(fs.BLUE)
        med.set_color(fs.ROSE)
    for w in bp["whiskers"] + bp["caps"]:
        w.set_color(fs.BLUE)
    ax.plot(range(1, len(means) + 1), means, "o", color=fs.INK, ms=5, zorder=5)
    ax.set_ylabel("平均定位清除时间 / s")
    ax.set_xlabel("策略")
    fs.style_xy(ax, title="问题三本地策略对照")
    fs.save(fig, out_png)


def main():
    n = 12
    seeds = [20260912 + i for i in range(n)]
    by_strat = {s["id"]: [] for s in STRATS}
    t0 = time.time()
    for s in STRATS:
        for i, seed in enumerate(seeds):
            row = one_trial(seed, s)
            by_strat[s["id"]].append(row)
            mark = "OK" if row["complete"] else "FAIL"
            print(
                f"{s['id']:14s} [{i+1:02d}/{n}] n={row['n_true']:2d} "
                f"cleared={row['n_cleared']:2d} avg={row['avg_s']:.0f}s {mark}"
            )
    summary = {s["id"]: summarize(by_strat[s["id"]]) for s in STRATS}
    out = {
        "n_seeds": n,
        "seeds": seeds,
        "strats": STRATS,
        "summary": summary,
        "rows": {k: v for k, v in by_strat.items()},
        "elapsed_wall_s": time.time() - t0,
    }
    dest = HERE / "logs" / "q3_strat_compare.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("----")
    for s in STRATS:
        u = summary[s["id"]]
        print(
            f"{s['id']:14s}  complete {u['n_complete']}/{n}  "
            f"mean avg {u['mean_avg_s']:.0f}s  mean T {u['mean_T']:.0f}s"
        )
    fig_compare(by_strat, str(HERE / "figs" / "fig_q3_strat.png"))
    print("wrote", dest)


if __name__ == "__main__":
    main()
