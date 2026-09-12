# -*- coding: utf-8 -*-
"""核验：圆域上探针最大方位间隙。环抱证书要求 maxgap ≤ 180°。"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from run_q4 import cover_points, encircle_skeleton, listen_net

HERE = Path(__file__).resolve().parent


def maxgap_deg(pts, G, radius=1000.0) -> float:
    angs = []
    G = np.asarray(G, float)
    for p in pts:
        p = np.asarray(p, float)
        if float(np.linalg.norm(p - G)) <= radius + 1e-9:
            angs.append(math.atan2(p[1] - G[1], p[0] - G[0]))
    if len(angs) < 2:
        return 360.0
    angs.sort()
    gaps = [angs[i + 1] - angs[i] for i in range(len(angs) - 1)]
    gaps.append(angs[0] + 2.0 * math.pi - angs[-1])
    return max(gaps) * 180.0 / math.pi


def grid_points(arena=1800.0, n_r=19, n_ang=96):
    pts = [np.zeros(2)]
    for i in range(1, n_r + 1):
        r = arena * i / n_r
        n = n_ang if r > 1 else 1
        for k in range(n):
            a = 2.0 * math.pi * k / n
            pts.append(np.array([r * math.cos(a), r * math.sin(a)]))
    return pts


def eval_net(name, pts, grid):
    worst = 0.0
    worst_g = grid[0]
    n_fail = 0
    for G in grid:
        g = maxgap_deg(pts, G)
        if g > worst:
            worst, worst_g = g, G
        if g > 180.0 + 1e-6:
            n_fail += 1
    rim = np.array([1800.0, 0.0])
    return {
        "name": name,
        "n_pts": len(pts),
        "worst_maxgap_deg": round(worst, 2),
        "worst_xy": [round(float(worst_g[0]), 1), round(float(worst_g[1]), 1)],
        "n_fail_grid": n_fail,
        "n_grid": len(grid),
        "pass_grid": n_fail == 0,
        "rim_out_maxgap_deg": round(maxgap_deg(pts, rim), 2),
        "tour_m": round(sum(float(np.linalg.norm(pts[i] - pts[i - 1])) for i in range(1, len(pts))), 1),
    }


def main():
    grid = grid_points()
    nets = [
        ("cover7", cover_points()),
        ("listen25", listen_net()),
        ("cert26", encircle_skeleton()),
        ("cert26+O", [np.zeros(2)] + encircle_skeleton()),
    ]
    rows = [eval_net(name, pts, grid) for name, pts in nets]
    dest = HERE / "logs" / "q4_cert_check.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in rows:
        print(
            f"{r['name']:10s} n={r['n_pts']:2d} worst={r['worst_maxgap_deg']:6.1f}° "
            f"at {r['worst_xy']}  rim(1800,0)={r['rim_out_maxgap_deg']:6.1f}°  "
            f"fail={r['n_fail_grid']}/{r['n_grid']}  tour≈{r['tour_m']:.0f}m"
        )
    print("wrote", dest)


if __name__ == "__main__":
    main()
