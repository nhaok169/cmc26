# -*- coding: utf-8 -*-
"""Q3/Q4 演练数据配对 + 论文图表生成.

数据源:
  1. 模拟器官方 result.json (源总数/全向数/定向数)
  2. 我方 robot_p*.jsonl (清除数/虚拟时间/测量数)

输出 (到 q4/figs/ 和 q3/figs/):
  fig_q4_scatter.png        Q4: T_avg vs 源总数 (散点, 按定向源比例着色)
  fig_q4_dirratio.png       Q4: T_avg vs 定向源占比
  fig_q3_scatter.png        Q3: T_avg vs 源总数
  fig_compare_q3q4.png      Q3 vs Q4 对比箱线图
  fig_q4_timeline.png       Q4: 各次演练 T_avg 时间线
  data_q3.csv / data_q4.csv 配对后的数据
"""
import json, glob, os, sys, datetime
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

ROOT = r"E:/Draft/MathModel"
C26 = os.path.join(ROOT, "cmc26")
LOGY = os.path.join(C26, "simulator", "JammersSimulatorData", "behavior-logs")
MYLOG = os.path.join(ROOT, "logs")


def load_official():
    """模拟器官方 result.json: (本地结束时间, problem, 总数, 全向, 定向, case_code)"""
    rows = []
    for rj in sorted(glob.glob(os.path.join(LOGY, "*.result.json"))):
        o = json.load(open(rj, encoding="utf-8"))
        utc = datetime.datetime.fromisoformat(o["ended_at_utc"].replace("Z", "+00:00"))
        local = utc.replace(tzinfo=None) + datetime.timedelta(hours=8)
        rows.append(dict(
            t=local,
            problem=o["problem_no"],
            n=o["jammer_count"],
            omni=o.get("omnidirectional_jammer_count", 0),
            dirn=o.get("directional_jammer_count", 0),
            case=o["case_code"],
        ))
    return rows


def load_mine(problem):
    """我方日志: (修改时间, 清除数, vt, 测量数, 清除尝试, 失败数, 文件名)"""
    rows = []
    pat = os.path.join(MYLOG, "robot_p%d_*.jsonl" % problem)
    for f in sorted(glob.glob(pat)):
        mt = datetime.datetime.fromtimestamp(os.path.getmtime(f))
        n_ok = 0
        n_try = 0
        n_meas = 0
        n_fail = 0
        last_vt = 0.0
        for line in open(f, encoding="utf-8"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            p = d.get("path")
            resp = d.get("resp", {})
            if p == "/measure":
                n_meas += 1
                last_vt = resp.get("virtual_time_s", last_vt)
            elif p == "/clear":
                n_try += 1
                if resp.get("clear_result") == "success":
                    n_ok += 1
                    last_vt = resp.get("virtual_time_s", last_vt)
                else:
                    n_fail += 1
        if n_ok == 0:
            continue
        rows.append(dict(t=mt, cleared=n_ok, vt=last_vt, meas=n_meas,
                         try_=n_try, fail=n_fail, name=os.path.basename(f)))
    return rows


def pair(official, mine, tol_s=120):
    """按时间就近配对, 且**强制要求 清除数==官方源总数** (真正的成功案例).

    说明: 我方日志与模拟器 result.json 并非严格一一对应(可能一次演练多次运行,
    或模拟器记录与我方日志存在时间漂移), 因此只有 cleared==n 的配对才是可信的
    "该案例被 100% 清除" 的证据. 其余丢弃, 避免把性能数据张冠李戴.
    """
    used = set()
    out = []
    for o in sorted(official, key=lambda r: r["t"]):
        best = None
        bd = 1e9
        for i, m in enumerate(mine):
            if i in used:
                continue
            # 关键约束: 清除数必须等于官方源总数
            if m["cleared"] != o["n"]:
                continue
            dt = abs((m["t"] - o["t"]).total_seconds())
            if dt < bd:
                bd, best = dt, i
        if best is not None and bd <= tol_s:
            used.add(best)
            mm = mine[best]
            out.append(dict(**o, cleared=mm["cleared"], vt=mm["vt"],
                            meas=mm["meas"], try_=mm["try_"], fail=mm["fail"],
                            name=mm["name"]))
    return out


def main():
    off = load_official()
    data3 = pair([r for r in off if r["problem"] == 3], load_mine(3))
    data4 = pair([r for r in off if r["problem"] == 4], load_mine(4))

    print("配对结果: 问题3 %d 次, 问题4 %d 次" % (len(data3), len(data4)))
    for d in data3 + data4:
        d["tavg"] = d["vt"] / d["cleared"] if d["cleared"] else float("inf")
        d["rate"] = d["cleared"] / d["n"] if d["n"] else 0
        d["dirratio"] = d["dirn"] / d["n"] if d["n"] else 0
        print("  p%d %s 总=%2d 全向=%2d 定向=%2d | 清除=%2d (%.0f%%) T_avg=%6.1fs"
              % (d["problem"], d["case"], d["n"], d["omni"], d["dirn"],
                 d["cleared"], d["rate"] * 100, d["tavg"]))

    # 输出 CSV
    os.makedirs(os.path.join(ROOT, "q4", "figs"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "q3", "figs"), exist_ok=True)
    for prob, rows in ((3, data3), (4, data4)):
        csv = os.path.join(ROOT, "q%d" % prob, "figs", "data_q%d.csv" % prob)
        with open(csv, "w", encoding="utf-8") as f:
            f.write("case,n,omni,dirn,cleared,rate,vt,tavg,meas,try,fail\n")
            for r in rows:
                f.write("%s,%d,%d,%d,%d,%.3f,%.1f,%.1f,%d,%d,%d\n"
                        % (r["case"], r["n"], r["omni"], r["dirn"], r["cleared"],
                           r["rate"], r["vt"], r["tavg"], r["meas"], r["try_"], r["fail"]))
        print("已写", csv)

    # ============ 图1: Q4 T_avg vs 源总数 (按定向比例着色) ============
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ns = [d["n"] for d in data4]
    ta = [d["tavg"] for d in data4]
    dr = [d["dirratio"] * 100 for d in data4]
    sc = ax.scatter(ns, ta, c=dr, cmap="RdYlBu_r", s=120, edgecolor="k", linewidth=0.6, zorder=3)
    ax.set_xlabel("干扰源总数")
    ax.set_ylabel("平均定位清除时间 $T_{avg}$ (s)")
    ax.set_title("问题4：$T_{avg}$ 随源总数变化（颜色=定向源占比%）")
    ax.grid(alpha=0.3, zorder=0)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("定向源占比 (%)")
    # 标注 100% 清除
    for d in data4:
        ax.annotate("%d/%d" % (d["cleared"], d["n"]), (d["n"], d["tavg"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    plt.tight_layout()
    p = os.path.join(ROOT, "q4", "figs", "fig_q4_scatter.png")
    plt.savefig(p, dpi=150)
    plt.close()
    print("已写", p)

    # ============ 图2: Q4 T_avg vs 定向源占比 ============
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(dr, ta, s=120, c="#c0392b", edgecolor="k", linewidth=0.6, zorder=3)
    # 趋势线
    if len(dr) > 2:
        z = np.polyfit(dr, ta, 1)
        xs = np.linspace(min(dr), max(dr), 50)
        ax.plot(xs, np.polyval(z, xs), "--", color="#2980b9", lw=1.6,
                label="线性趋势 (斜率 %.2f s/1%%)" % z[0], zorder=2)
        ax.legend()
    ax.set_xlabel("定向源占比 (%)")
    ax.set_ylabel("平均定位清除时间 $T_{avg}$ (s)")
    ax.set_title("问题4：$T_{avg}$ 随定向源占比变化")
    ax.grid(alpha=0.3, zorder=0)
    plt.tight_layout()
    p = os.path.join(ROOT, "q4", "figs", "fig_q4_dirratio.png")
    plt.savefig(p, dpi=150)
    plt.close()
    print("已写", p)

    # ============ 图3: Q3 T_avg vs 源总数 ============
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ns3 = [d["n"] for d in data3]
    ta3 = [d["tavg"] for d in data3]
    ax.scatter(ns3, ta3, s=120, c="#27ae60", edgecolor="k", linewidth=0.6, zorder=3)
    for d in data3:
        ax.annotate("%d/%d" % (d["cleared"], d["n"]), (d["n"], d["tavg"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("干扰源总数")
    ax.set_ylabel("平均定位清除时间 $T_{avg}$ (s)")
    ax.set_title("问题3（全向源）：$T_{avg}$ 随源总数变化")
    ax.grid(alpha=0.3, zorder=0)
    plt.tight_layout()
    p = os.path.join(ROOT, "q3", "figs", "fig_q3_scatter.png")
    plt.savefig(p, dpi=150)
    plt.close()
    print("已写", p)

    # ============ 图4: Q3 vs Q4 对比箱线图 ============
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    try:
        bp = ax.boxplot([ta3, ta], tick_labels=["问题3\n(全向)", "问题4\n(混合)"],
                        patch_artist=True, widths=0.5)
    except TypeError:
        bp = ax.boxplot([ta3, ta], labels=["问题3\n(全向)", "问题4\n(混合)"],
                        patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], ["#27ae60", "#c0392b"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    # 叠加散点
    for i, vals in enumerate((ta3, ta), start=1):
        ax.scatter(np.random.normal(i, 0.04, len(vals)), vals,
                   s=40, c="k", alpha=0.5, zorder=3)
    ax.set_ylabel("平均定位清除时间 $T_{avg}$ (s)")
    ax.set_title("问题3 与 问题4 的 $T_{avg}$ 对比（真实演练）")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()
    p = os.path.join(ROOT, "q4", "figs", "fig_compare_q3q4.png")
    plt.savefig(p, dpi=150)
    plt.close()
    print("已写", p)

    # ============ 图5: Q4 各次演练时间线（清除率=100%标注） ============
    fig, ax = plt.subplots(figsize=(9, 4.8))
    order = sorted(range(len(data4)), key=lambda i: data4[i]["t"])
    labels = [data4[i]["case"][:4] for i in order]
    vals = [data4[i]["tavg"] for i in order]
    nsrc = [data4[i]["n"] for i in order]
    cols = ["#27ae60" if data4[i]["rate"] >= 1.0 else "#c0392b" for i in order]
    ax.bar(labels, vals, color=cols, edgecolor="k", linewidth=0.5)
    for i, (l, v, n) in enumerate(zip(labels, vals, nsrc)):
        ax.text(i, v + 12, "%d源" % n, ha="center", fontsize=8)
    ax.axhline(np.mean(vals), ls="--", color="#2c3e50",
               label="均值 %.1f s" % np.mean(vals))
    ax.legend()
    ax.set_ylabel("$T_{avg}$ (s)")
    ax.set_xlabel("案例编码（前4位）")
    ax.set_title("问题4 各次演练 $T_{avg}$（绿色=100%清除）")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = os.path.join(ROOT, "q4", "figs", "fig_q4_timeline.png")
    plt.savefig(p, dpi=150)
    plt.close()
    print("已写", p)

    # 汇总
    print("\n=== 汇总 ===")
    for prob, rows in ((3, data3), (4, data4)):
        a = np.array([r["tavg"] for r in rows])
        full = sum(1 for r in rows if r["rate"] >= 1.0)
        print("问题%d: %d 次, 100%%清除 %d 次, T_avg 均值=%.1f s, 标准差=%.1f s, 最好=%.1f, 最差=%.1f"
              % (prob, len(rows), full, a.mean(), a.std(), a.min(), a.max()))


if __name__ == "__main__":
    main()
