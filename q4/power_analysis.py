# -*- coding: utf-8 -*-
"""样本量与统计功效分析: 当前轮次够不够? 再跑 50 轮值不值?

核心问题:
  1. 当前样本能检测出多小的性能差异? (最小可检测效应 MDE)
  2. 论文要报告 T_avg 的置信区间, 当前宽度是多少?
  3. 追加轮次的边际收益如何? (MDE 随 n 下降曲线)

方法:
  - 两独立样本 t 检验功效: n = 2*(z_a/2 + z_b)^2 * (sigma / delta)^2
  - 以各指标实测 Std 作为 sigma
  - Q4 强烈建议按【官方源数分层/协变量】比较, 因为 T_avg = vt/n 的
    方差主要来自场景源数随机, 而非策略本身
"""
import os, csv, math
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Z_A, Z_B = 1.959964, 0.841621   # alpha=0.05 双尾, power=0.80


def mde(sigma, n_per_group):
    """给定每组样本量, 可检测的最小绝对差异 (80% 功效)"""
    return (Z_A + Z_B) * sigma * math.sqrt(2.0 / n_per_group)


def n_needed(sigma, delta):
    """给定要检测的绝对差异, 每组需要多少样本"""
    return 2 * (Z_A + Z_B) ** 2 * (sigma / delta) ** 2


def ci95(vals):
    if len(vals) < 2:
        return float("nan"), float("nan")
    se = st.stdev(vals) / math.sqrt(len(vals))
    m = st.mean(vals)
    return m - 1.96 * se, m + 1.96 * se


def load_q4_all():
    p = os.path.join(ROOT, "logs_q4_compare.csv")
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    for r in rows:
        r["truth_n"] = int(r["truth_n"])
        r["vt_s"] = float(r["vt_s"])
        r["cleared"] = int(r["cleared"])
        r["T_avg"] = r["vt_s"] / max(r["cleared"], 1)
    return rows


def load_q3():
    p = os.path.join(ROOT, "logs_q3_batch", "stats_csv", "每轮明细.csv")
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    for r in rows:
        r["vt"] = float(r["vt_total_s"])
        r["T_avg"] = float(r["avg_per_source_s"])
        r["n"] = int(r["cleared"])
    return rows


def report(name, tavg, vt, extra=""):
    print("\n" + "=" * 78)
    print("【%s】 N = %d 轮%s" % (name, len(tavg), extra))
    print("=" * 78)
    s_t, s_v = st.stdev(tavg), st.stdev(vt)
    m_t, m_v = st.mean(tavg), st.mean(vt)
    lo, hi = ci95(tavg)
    print("T_avg   = %.1f ± %.1f  (CV %.1f%%)   95%%CI [%.1f, %.1f]  半宽 ±%.1f (%.1f%%)"
          % (m_t, s_t, s_t / m_t * 100, lo, hi, (hi - lo) / 2, (hi - lo) / 2 / m_t * 100))
    lo2, hi2 = ci95(vt)
    print("vt      = %.0f ± %.0f  (CV %.1f%%)   95%%CI [%.0f, %.0f]  半宽 ±%.0f (%.1f%%)"
          % (m_v, s_v, s_v / m_v * 100, lo2, hi2, (hi2 - lo2) / 2, (hi2 - lo2) / 2 / m_v * 100))

    print("\n-- 当前样本量能检测到的最小差异 (MDE, 80%% 功效, 两独立组) --")
    print("  指标      当前MDE       要检测5%%    10%%     15%%")
    for lbl, s, m in [("T_avg", s_t, m_t), ("vt", s_v, m_v)]:
        cur = mde(s, len(tavg))
        row = "%-8s  %7.1f (%.1f%%)" % (lbl, cur, cur / m * 100)
        for pct in (5, 10, 15):
            need = n_needed(s, m * pct / 100)
            row += "   %6.0f" % math.ceil(need)
        print(row)

    print("\n-- 追加轮次的边际收益 (每组样本量 -> T_avg 可检测差异) --")
    for nn in (10, 20, 30, 50, 80, 100, 150):
        d = mde(s_t, nn)
        print("   n=%-4d  MDE = %6.1f s/个 (%.1f%%)" % (nn, d, d / m_t * 100))
    return s_t, m_t


def main():
    q3 = load_q3()
    q4 = load_q4_all()

    print("#" * 78)
    print("# Q3 / Q4 样本量与功效分析")
    print("#" * 78)

    s3, m3 = report("Q3  (10 轮, v6 自适应追踪)",
                    [r["T_avg"] for r in q3], [r["vt"] for r in q3])

    s4, m4 = report("Q4  (三批合并 28 轮, 同一份代码)",
                    [r["T_avg"] for r in q4], [r["vt_s"] for r in q4])

    # Q4 按官方源数分层后, 残差 Std 能降多少
    print("\n" + "=" * 78)
    print("【Q4 关键: 按官方源数分层能降低多少方差?】")
    print("=" * 78)
    xs = [r["truth_n"] for r in q4]
    ys = [r["vt_s"] for r in q4]
    mx, my = st.mean(xs), st.mean(ys)
    b1 = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    b0 = my - b1 * mx
    resid = [y - (b0 + b1 * x) for x, y in zip(xs, ys)]
    ss_t = sum((y - my) ** 2 for y in ys)
    ss_r = sum(e ** 2 for e in resid)
    print("vt 对源数回归: vt = %.1f %+.1f*n,  R2 = %.3f" % (b0, b1, 1 - ss_r / ss_t))
    print("vt 原始 Std = %.0f  ->  控制源数后残差 Std = %.0f  (降 %.0f%%)"
          % (st.stdev(ys), st.stdev(resid), (1 - st.stdev(resid) / st.stdev(ys)) * 100))
    print("⇒ 等效样本量放大 %.1f 倍" % (st.stdev(ys) / st.stdev(resid)) ** 2)

    # T_avg 对源数回归
    yt = [r["T_avg"] for r in q4]
    mt = st.mean(yt)
    b1t = sum((x - mx) * (y - mt) for x, y in zip(xs, yt)) / sum((x - mx) ** 2 for x in xs)
    b0t = mt - b1t * mx
    residt = [y - (b0t + b1t * x) for x, y in zip(xs, yt)]
    ss_tt = sum((y - mt) ** 2 for y in yt)
    ss_rt = sum(e ** 2 for e in residt)
    print("\nT_avg 对源数回归: T_avg = %.1f %+.1f*n,  R2 = %.3f" % (
        b0t, b1t, 1 - ss_rt / ss_tt))
    print("T_avg 原始 Std = %.0f  ->  控制源数后残差 Std = %.0f  (降 %.0f%%)"
          % (st.stdev(yt), st.stdev(residt), (1 - st.stdev(residt) / st.stdev(yt)) * 100))
    print("⇒ 等效样本量放大 %.1f 倍" % (st.stdev(yt) / st.stdev(residt)) ** 2)

    print("\n" + "=" * 78)
    print("【结论阈值】")
    print("=" * 78)
    for lbl, s, m in [("Q3 T_avg", s3, m3), ("Q4 T_avg", s4, m4)]:
        for pct in (5, 10):
            print("  %-10s 检测 %2d%% 差异需每组 %3.0f 轮" % (
                lbl, pct, math.ceil(n_needed(s, m * pct / 100))))
    print("\n  注: Q4 的 T_avg 方差主要来自【场景源数随机 10~16】, 不是策略;")
    print("      因此比较策略时必须按官方 jammer_count 分层或作协变量,")
    print("      否则堆轮数的收益会被场景噪声吃掉 (需 150+ 轮才达 5% 精度)。")


if __name__ == "__main__":
    main()
