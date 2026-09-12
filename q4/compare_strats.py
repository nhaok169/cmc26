# -*- coding: utf-8 -*-
"""问题4 本地策略对照：混合随机世界 + 圆缘外向定向应力场景。"""
from __future__ import annotations

import json
import sys
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "paper"))
sys.path.insert(0, str(ROOT / "q3"))
sys.path.insert(0, str(ROOT / "q2"))
sys.path.insert(0, str(ROOT / "q1"))
sys.path.insert(0, str(HERE))

from local_world import LocalSimulator  # noqa: E402
from run_q4 import Dog  # noqa: E402
import figstyle as fs  # noqa: E402

MODES = [
    {"id": "cover7", "label": "七点覆盖"},
    {"id": "hybrid", "label": "实用环绕+早停"},
    {"id": "hybrid_full", "label": "实用环绕走满"},
    {"id": "cert26", "label": "26点环抱"},
]


def one_trial(seed: int, mode: str, stress=None, p_dir=0.5):
    rng = np.random.default_rng(seed)
    sim = LocalSimulator(rng, p_dir=p_dir, stress=stress)
    dog = Dog(sim, mode=mode)
    t0 = time.time()
    with redirect_stdout(StringIO()):
        summary = dog.run(do_enter=True)
    n_true = sim.n_true
    alive = [ch for ch, s in sim.sources.items() if s["alive"]]
    n_cleared = int(summary["n_cleared"])
    avg = (summary["virtual_time_s"] / n_cleared) if n_cleared else None
    complete = (n_cleared == n_true) and (not alive) and (not summary["failed"])
    return {
        "seed": seed,
        "mode": mode,
        "stress": stress or "mixed",
        "n_true": n_true,
        "n_dir": sim.n_dir,
        "n_cleared": n_cleared,
        "alive": alive,
        "failed": summary["failed"],
        "virtual_s": summary["virtual_time_s"],
        "avg_s": avg,
        "n_listen": summary.get("n_listen"),
        "idle_stopped": summary.get("idle_stopped"),
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
        "mean_T": float(np.mean(Ts)) if Ts else None,
        "mean_listen": float(np.mean([r["n_listen"] for r in rows])),
        "fail_seeds": [r["seed"] for r in rows if not r["complete"]],
    }


def fig_compare(out, out_png):
    fs.apply()
    fig1, ax = fs.new_fig()
    labels = [m["label"] for m in MODES]
    mixed = out["mixed"]["summary"]
    rim = out["rim_out"]["summary"]
    x = np.arange(len(MODES))
    w = 0.36
    ax.bar(x - w / 2, [100 * mixed[m["id"]]["complete_rate"] for m in MODES], w, color=fs.BLUE, label="混合随机")
    ax.bar(x + w / 2, [100 * rim[m["id"]]["complete_rate"] for m in MODES], w, color=fs.SAND, label="圆缘外向")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("全清比例 / %")
    ax.set_ylim(0, 110)
    ax.legend(loc="lower right")
    fs.style_xy(ax, title="问题四本地：全清比例")
    fs.save(fig1, str(Path(out_png).with_name("fig_q4_strat_rate.png")))

    fig2, ax = fs.new_fig()
    ax.bar(x - w / 2, [mixed[m["id"]]["mean_avg_s"] or 0 for m in MODES], w, color=fs.BLUE, label="混合随机")
    ax.bar(x + w / 2, [rim[m["id"]]["mean_avg_s"] or 0 for m in MODES], w, color=fs.SAND, label="圆缘外向")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("平均定位清除时间 / s")
    ax.legend(loc="upper left")
    fs.style_xy(ax, title="问题四本地：平均时间")
    fs.save(fig2, str(Path(out_png).with_name("fig_q4_strat_time.png")))


def run_batch(name, seeds, stress, p_dir):
    by_mode = {m["id"]: [] for m in MODES}
    for m in MODES:
        for i, seed in enumerate(seeds):
            row = one_trial(seed, m["id"], stress=stress, p_dir=p_dir)
            by_mode[m["id"]].append(row)
            mark = "OK" if row["complete"] else "MISS"
            print(
                f"{name:8s} {m['id']:12s} [{i+1:02d}/{len(seeds)}] "
                f"n={row['n_true']:2d} dir={row['n_dir']:2d} "
                f"cleared={row['n_cleared']:2d} avg={row['avg_s']:.0f}s {mark}"
            )
    return {
        "seeds": seeds,
        "summary": {m["id"]: summarize(by_mode[m["id"]]) for m in MODES},
        "rows": by_mode,
    }


def main():
    mixed_seeds = [20260912 + i for i in range(8)]
    rim_seeds = [31000 + i for i in range(6)]
    t0 = time.time()
    mixed = run_batch("mixed", mixed_seeds, None, 0.5)
    rim = run_batch("rim", rim_seeds, "rim_out", 0.5)
    out = {"mixed": mixed, "rim_out": rim, "elapsed_wall_s": time.time() - t0}
    dest = HERE / "logs" / "q4_strat_compare.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("----")
    for batch_name, batch in (("mixed", mixed), ("rim_out", rim)):
        print(f"[{batch_name}]")
        for m in MODES:
            u = batch["summary"][m["id"]]
            print(
                f"  {m['id']:12s} complete {u['n_complete']}/{u['n']}  "
                f"mean avg {u['mean_avg_s']:.0f}s  mean T {u['mean_T']:.0f}s  "
                f"listens {u['mean_listen']:.1f}"
            )
    fig_compare(out, str(HERE / "figs" / "fig_q4_strat.png"))
    print("wrote", dest)


if __name__ == "__main__":
    main()
