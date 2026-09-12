# -*- coding: utf-8 -*-
"""Q4 v3 mock 测试 (policy3: 合并巡回 + 末段 TSP).

用法:
    python q4/mock3.py 1            # 单种子详细
    python q4/mock3.py 24           # 24 种子汇总
"""
import os, sys, math, time
import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "q4"))
sys.path.insert(0, os.path.join(_root, "q2"))
import robot as R
import mock_test as MT
from policy3 import Policy3
from ledger2 import SKELETON, SKELETON_N, max_gap_all


def run(seed, verbose=False):
    srv = MT.FakeServer(problem=4, seed=seed)
    n_all = len(srv.jammers)
    MT._setup_fake(srv)
    sim = R.Sim("http://127.0.0.1:2026", "TEST", MT._DummyLog())
    sim.enter(wait_s=5)
    pol = Policy3(sim, verbose=verbose)
    n, vt = pol.run()
    cleared = sum(1 for j in srv.jammers.values() if j["cleared"])
    n_omni = sum(1 for j in srv.jammers.values() if j.get("omni", True))
    return dict(ok=cleared == n_all, vt=vt, cleared=cleared, n=n_all,
                omni=n_omni, dir=n_all - n_omni, move=pol.mv,
                measure=sim.measure_cnt, switch=sim.switch_cnt,
                clear_fail=sim.clear_fail, phases=pol.phase_stats,
                tour=pol._order_len, n_clear=pol.n_clear, n_fail=pol.n_fail,
                ch_meas=pol.ch_meas,
                residual=len(pol.ledger.unknown_channels()))


def detail(seed):
    r = run(seed, verbose=True)
    print(f"\nseed={seed} cleared={r['cleared']}/{r['n']} vt={r['vt']:.0f} "
          f"avg={r['vt']/max(1,r['cleared']):.0f}  order={r['tour']:.0f}m  "
          f"omni={r['omni']} dir={r['dir']}")
    print(f"{'phase':10s} {'vt':>8s} {'move_m':>8s} {'move_s':>7s} {'meas':>5s} {'clear':>6s}")
    for nm, (vt, mv, nme, ncl) in r['phases'].items():
        print(f"{nm:10s} {vt:8.0f} {mv:8.0f} {mv/5:7.0f} {nme:5d} {ncl:6d}")
    print(f"total move={r['move']:.0f}m ({r['move']/5:.0f}s)  meas={r['measure']} "
          f"switch={r['switch']} clear_ok={r['cleared']} clear_attempts={r['n_clear']} "
          f"clear_fail={r['clear_fail']}")
    mc = sorted(r['ch_meas'].items(), key=lambda x: -x[1])
    print("measure per channel:", mc)


def main():
    args = sys.argv[1:]
    n_seeds = int(args[0]) if args else 24
    if n_seeds == 1:
        detail(1)
        return
    rows = []
    for seed in range(1, n_seeds + 1):
        try:
            r = run(seed)
        except Exception as e:
            print(f"seed={seed} ERROR {e}")
            import traceback; traceback.print_exc()
            continue
        rows.append(r)
        avg = r["vt"] / r["cleared"] if r["cleared"] else float("inf")
        ph = r["phases"]
        print(f"seed={seed:2d} {r['cleared']:2d}/{r['n']:2d} n={r['n']:2d} "
              f"vt={r['vt']:6.0f} avg={avg:6.1f} move={r['move']/5:6.0f}s "
              f"ini={ph.get('initial',(0,))[0]:4.0f} mrg={ph.get('merged',(0,))[0]:6.0f} "
              f"fin={ph.get('final',(0,))[0]:5.0f} meas={r['measure']:3d} "
              f"fail={r['clear_fail']:2d} resid={r['residual']:2d} "
              f"{'OK' if r['ok'] else 'FAIL'}", flush=True)

    ok = [r for r in rows if r["ok"]]
    avgs = [r["vt"] / r["cleared"] for r in rows if r["cleared"]]
    print("\n" + "=" * 78)
    print(f"Q4 v3  {len(rows)} seeds  Clear={len(ok)}/{len(rows)}  "
          f"T_avg={np.mean(avgs):.1f}  Std={np.std(avgs):.1f}")
    if rows:
        print(f"平均移动={np.mean([r['move'] for r in rows])/5:.0f}s  "
              f"平均测量数={np.mean([r['measure'] for r in rows]):.0f}  "
              f"平均源数={np.mean([r['n'] for r in rows]):.1f}  "
              f"平均清除失败={np.mean([r['clear_fail'] for r in rows]):.1f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
