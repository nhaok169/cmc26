# -*- coding: utf-8 -*-
"""问题2 设计图: (a) t-D平面定位直径热力图(h=杠杆下界) + 40m等值线 + 四流派成功区间包络
             (b) E[T] vs 赌注t (两流派, 自适应rho=0.5L)
输出: figs/design_chart.png
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False
from geometry import DELTA, k_expr, episode

K = 40.0 / (2 * DELTA)
D_NEAR, D_FAR = 5.0, 1500.0
EPS5 = np.linspace(-DELTA, DELTA, 5)
c_lever = 0.5 / np.sqrt(1 - 0.25)          # s_min=0.5
def lever_h(t): return c_lever * max(D_FAR - t, t - D_NEAR)

# ---------- Panel 1: heatmap ----------
ts = np.arange(150, 1451, 15.0)
Ds = np.linspace(5, 1500, 145)
Z = np.zeros((len(Ds), len(ts)))
for i, D in enumerate(Ds):
    for j, t in enumerate(ts):
        h = lever_h(t)
        Z[i, j] = 2 * DELTA * np.sqrt(min(k_expr(t, h, D, e) for e in EPS5))
Zc = np.clip(Z, 0, 150)                     # 截断到150m便于显示

schools = [(550, 498, "minimax (550,498)"),
           (675, 476, "概率最优 (675,476)"),
           (750, 450, "保守 (750,450)"),
           (500, 250, "时间最优 (500,250)")]
# 各流派用自身h的成功区间
def interval(t, h):
    m = np.array([max(k_expr(t, h, D, e) for e in EPS5) <= K * K for D in Ds])
    idx = np.where(m)[0]
    if len(idx) == 0: return None
    return Ds[idx[0]], Ds[idx[-1]]

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
ax = axes[0]
pm = ax.pcolormesh(ts, Ds, Zc, cmap="viridis_r", shading="auto")
cs = ax.contour(ts, Ds, Z, levels=[40], colors="red", linewidths=2)
ax.clabel(cs, fmt="40 m", fontsize=9)
colors = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00"]
for (t, h, name), col in zip(schools, colors):
    iv = interval(t, h)
    if iv:
        ax.plot([t, t], [iv[0], iv[1]], color=col, lw=3, alpha=0.85)
        ax.annotate(f"{name}\n[{iv[0]:.0f},{iv[1]:.0f}]m", (t, iv[1] + 40),
                    color=col, fontsize=7.5, ha="center")
ax.plot([5, 734], [734, 734], "--", color="#e41a1c", lw=1, alpha=0.5)
ax.set_xlabel("赌注 t / m　　（侧偏 h 取杠杆下界）")
ax.set_ylabel("真实距离 D / m")
ax.set_title("(a) (t, D) 平面定位直径　　红线：40 m 两步成功等值线")
fig.colorbar(pm, ax=ax, label="定位区域直径 / m（显示截断于 150）")

# ---------- Panel 2: E[T] vs t ----------
ax = axes[1]
rho_ad = lambda L: max(50.0, 0.5 * L)
w = Ds.copy(); w /= w.sum()
tt = np.arange(350, 1001, 50.0)
ET_time, ET_cons = [], []
for t in tt:
    T1 = np.array([episode(t, 250.0, D, rho_rule=rho_ad)[0] for D in Ds]); ET_time.append(T1 @ w)
    T2 = np.array([episode(t, lever_h(t), D, rho_rule=rho_ad)[0] for D in Ds]); ET_cons.append(T2 @ w)
ax.plot(tt, ET_time, "o-", color="#ff7f00", label="时间流派（h = 250）")
ax.plot(tt, ET_cons, "s-", color="#4daf4a", label="保守流派（h = 杠杆下界）")
for (t, h, name), col in zip(schools, colors):
    if 350 <= t <= 1000:
        ax.annotate(name.split(" ")[0], (t, 268 if "时间" in name else 319),
                    color=col, fontsize=8, ha="center")
ax.set_xlabel("赌注 t / m")
ax.set_ylabel("期望总时间 E[T] / s")
ax.set_title("(b) E[T]–赌注　　自适应 ρ = 0.5 L，先验 f_D ∝ D")
ax.legend(); ax.grid(alpha=0.3)

os.makedirs(os.path.join(os.path.dirname(__file__), "figs"), exist_ok=True)
out = os.path.join(os.path.dirname(__file__), "figs", "design_chart.png")
fig.tight_layout()
fig.savefig(out, dpi=150)
print("saved:", out, os.path.getsize(out), "bytes")
print("E[T] time school  min at t=", tt[int(np.argmin(ET_time))], "value", min(ET_time))
print("E[T] conserv school min at t=", tt[int(np.argmin(ET_cons))], "value", min(ET_cons))
for t, h, name in schools:
    print(f"interval {name}: {interval(t,h)}")

