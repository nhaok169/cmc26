# -*- coding: utf-8 -*-
"""v3.1 -> v3.2 四项数值实验:
实验1  κ = MEC半径/(直径/2) 分布 + minimax赌注(550,498)的MEC级保证核验(含误差)
实验2  E[T] 蒙特卡洛 (e_i ≠ 0): 四流派优势是否维持
实验3  rho 自适应规则扫描
实验4  D_far 楔形边缘算例
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from geometry import (DELTA, wedge_hp, region, diam_and_axis, mec, k_expr, episode, expected_time)

# ================= 实验1: κ 分布 =================
print("=" * 74)
print("实验1a: κ = MEC半径/(直径/2) 分布  (e=0, 23t×4h×30D)")
print("=" * 74)
kaps = []
mecs_when_40 = []
for t in np.arange(300, 1401, 50):
    for h in (250, 400, 550, 700):
        for D in np.arange(50, 1501, 50):
            G = np.array([D, 0.0])
            S2 = np.array([t, h])
            th1 = 0.0
            th2 = np.arctan2(G[1] - S2[1], G[0] - S2[0])
            v = region(wedge_hp(np.array([0., 0.]), th1) + wedge_hp(S2, th2))
            if len(v) < 3:
                continue
            d2, _ = diam_and_axis(v)
            if d2 > 3000:
                continue
            r, _ = mec(v)
            kaps.append(r / (d2 / 2))
            if d2 <= 40:
                mecs_when_40.append(r)
kaps = np.array(kaps)
print(f"样本数 {len(kaps)}")
print(f"κ: mean={kaps.mean():.4f}  p95={np.percentile(kaps,95):.4f}  p99={np.percentile(kaps,99):.4f}  max={kaps.max():.4f}")
print(f"直径≤40 的样本数 {len(mecs_when_40)}, 其中最大 MEC 半径 = {max(mecs_when_40):.2f} m  (阈值20)")
print(f"直径≤40 且 MEC>20 的比例 = {sum(1 for r in mecs_when_40 if r>20)/max(1,len(mecs_when_40))*100:.1f}%")

print()
print("=" * 74)
print("实验1b: minimax赌注(550,498) 在 D∈[5,734] 的 MEC 级保证 (e1,e2∈{-δ,0,+δ}全组合)")
print("=" * 74)
worst_r = 0; worst_case = None
for D in np.arange(5, 735, 25):
    for e1 in (-DELTA, 0.0, DELTA):
        G = np.array([D * np.cos(e1), D * np.sin(e1)])
        S2 = np.array([550., 498.])
        th2t = np.arctan2(G[1] - S2[1], G[0] - S2[0])
        for e2 in (-DELTA, 0.0, DELTA):
            v = region(wedge_hp(np.array([0., 0.]), 0.0) + wedge_hp(S2, th2t + e2))
            if len(v) < 3:
                continue
            r, _ = mec(v)
            if r > worst_r:
                worst_r = r; worst_case = (D, np.rad2deg(e1), np.rad2deg(e2))
print(f"最坏 MEC 半径 = {worst_r:.2f} m  (D={worst_case[0]}, e1={worst_case[1]}°, e2={worst_case[2]}°)")
print(f"=> {'MEC级保证成立(≤20)' if worst_r<=20 else 'MEC级保证被打破, K-条件确非充要'}")

# ================= 实验2: E[T] 蒙特卡洛 =================
print()
print("=" * 74)
print("实验2: E[T]|成功 蒙特卡洛, e_i~U[-1°,1°] iid, 30 draws, f_D∝D, rho=max(50,0.5L)")
print("=" * 74)
Ds = np.linspace(5, 1500, 40)
w = Ds.copy(); w /= w.sum()
rho_ad = lambda L: max(50.0, 0.5 * L)
schools = [(550, 498, "minimax保证"), (675, 476, "P最优(α=1)"),
           (750, 450, "保守流派E[T]"), (500, 250, "时间流派E[T]")]
for t, h, name in schools:
    et0, p0, en0 = expected_time(t, h, Ds, w=w, rho_rule=rho_ad)
    ETs, Ps = [], []
    rng = np.random.default_rng(2026)
    for k in range(30):
        etk, pk, _ = expected_time(t, h, Ds, w=w, rng=rng, rho_rule=rho_ad)
        ETs.append(etk); Ps.append(pk)
    ETs, Ps = np.array(ETs), np.array(Ps)
    print(f"{name:12s} (t,h)=({t},{h}):  E[T](e=0)={et0:5.1f}s P={p0:.3f}  "
          f"E[T](MC)={ETs.mean():5.1f}±{ETs.std():4.1f}s  P(MC)={Ps.mean():.3f}  "
          f"E[n]={en0:.2f}")

# ================= 实验3: rho 规则 =================
print()
print("=" * 74)
print("实验3a: 固定rho扫描 (e=0), 场景(赌注, D)")
print("=" * 74)
scen = [((700, 462), 1500, "远目标/保守赌注"), ((500, 250), 1500, "远目标/时间赌注"),
        ((500, 250), 900, "中目标/时间赌注")]
for (t, h), D, name in scen:
    row = []
    for rho in (100, 150, 200, 250, 300, 400, 500, 600):
        T, n, ok = episode(t, h, D, rho_rule=lambda L, r=rho: r)
        row.append(f"rho={rho}:{T if ok else float('inf'):4.0f}s/{n}meas/{'ok' if ok else 'fail'}")
    print(f"{name} (t,h)=({t},{h}), D={D}:  " + "  ".join(row))

print()
print("实验3b: 自适应 rho = max(50, c·L长轴) vs 固定300 (e=0, f_D∝D；仅成功样本计时)")
for (t, h, name) in [(500, 250, "时间流派"), (750, 450, "保守流派")]:
    for c in (0.5, 0.75, 1.0, 1.5, 2.0):
        et, p, _ = expected_time(t, h, Ds, w=w, rho_rule=lambda L, c=c: max(50.0, c * L))
        print(f"{name}(t,h)=({t},{h}) rho={c:.2f}·L:  E[T]|ok={et:5.1f}s  P={p:.3f}")
    et, p, _ = expected_time(t, h, Ds, w=w, rho_rule=lambda L: 300.0)
    print(f"{name}(t,h)=({t},{h}) rho=300固定: E[T]|ok={et:5.1f}s  P={p:.3f}")

# ================= 实验4: D_far 楔形边缘算例 =================
print()
print("=" * 74)
print("实验4: D_far = min(1500, 出圆距离) 的楔形边缘差异算例")
print("=" * 74)
def exit_dist(S1, psi, R=1800.0):
    e = np.array([np.cos(psi), np.sin(psi)])
    a = 1.0                       # |S1 + l e|^2 = R^2 -> l^2 + 2(S1·e)l + |S1|^2-R^2 = 0
    b = 2 * (S1 @ e)
    c = S1 @ S1 - R * R
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    l = (-b + np.sqrt(disc)) / 2
    return l if l > 0 else None
for S1xy, th in [((1200, 900), np.deg2rad(180)), ((1700, 100), np.deg2rad(120)),
                 ((1790, 0), np.deg2rad(95)), ((0, 0), np.deg2rad(37))]:
    S1 = np.array(S1xy, float)
    th = th % (2 * np.pi)
    ex = [exit_dist(S1, th + s * DELTA) for s in (-1, 0, 1)]
    ex = [None if e is None else min(1500.0, e) for e in ex]
    print(f"S1={S1xy}, θ1={np.rad2deg(th):5.1f}°:  出圆距离(θ-1°,θ,θ+1°) = "
          f"{['%.1f' % e if e is not None else '∞>1500' for e in ex]}  "
          f"→ D_far(稳健)=min(1500, max)={[f'{e:.1f}' for e in ex if e]}")
