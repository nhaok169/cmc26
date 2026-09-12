# -*- coding: utf-8 -*-
"""Q4 两批次严格对照: 历史 600s 批次 (17:35-17:39) vs 本批次 (22:39+).

目的: 判定 T_avg 从 ~550 抬到 ~654 是"策略退化"还是"场景源数差异".
对照原则:
  1. 用模拟器官方 result.json 的 jammer_count 作为真值源数 (不依赖策略自检数)
  2. 分离两个因子: vt(总时间, 反映策略效率) 与 n(源数, 反映场景难度)
  3. T_avg = vt / n, 因此分别检验 vt 与 n 的差异显著性
"""
import os, sys, json, glob, math, time, csv
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM = os.path.join(ROOT, "simulator", "JammersSimulatorData", "behavior-logs")

BATCHES = {
    "历史600s批次(17:35)": os.path.join(ROOT, "logs_q4_hist600"),
    "本批次(22:39+)": os.path.join(ROOT, "logs_q4_batch"),
    "老代码复跑(23:00)": os.path.join(ROOT, "logs_q4_600wt"),
    "修复版验证(23:24+)": os.path.join(ROOT, "logs_q4_fix"),
}


def load_sim_truth():
    """读 result.json -> {mtime: (total, omni, dir)}"""
    out = []
    for f in glob.glob(os.path.join(SIM, "*.result.json")):
        d = json.load(open(f, encoding="utf-8"))
        if d.get("problem_no") != 4:
            continue
        out.append((
            os.path.getmtime(f),
            d["jammer_count"],
            d["omnidirectional_jammer_count"],
            d["directional_jammer_count"],
            d.get("case_code", ""),
        ))
    out.sort()
    return out


def parse_run(path):
    """解析单轮日志 -> vt, cleared, detected, n_measure"""
    recs = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            recs.append(json.loads(line))
    seq = [r for r in recs if "virtual_time_s" in r.get("resp", {})]
    if not seq:
        return None
    vt = float(seq[-1]["resp"]["virtual_time_s"])
    cleared = meas = 0
    ch_tried = set()   # 策略尝试清除过的频道 (detected 的代理: 策略认为存在的源)
    for r in seq:
        p = r.get("path", "")
        resp = r.get("resp", {})
        if p == "/measure":
            meas += 1
        elif p == "/clear":
            # 注意: Q4 的成功字段是 "success", 不是 "cleared"
            if resp.get("clear_result") == "success":
                cleared += 1
            ch = r.get("req", {}).get("channel")
            if ch is not None:
                ch_tried.add(ch)
    det = len(ch_tried)
    return {"vt": vt, "cleared": cleared, "detected": det,
            "n_measure": meas, "file": os.path.basename(path),
            "mtime": os.path.getmtime(path)}


def welch_t(a, b):
    """Welch t 检验 (不等方差), 返回 t 值与近似 p 值(正态近似)"""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan"), float("nan")
    ma, mb = st.mean(a), st.mean(b)
    va, vb = st.variance(a), st.variance(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return float("nan"), float("nan")
    t = (ma - mb) / se
    # 正态近似 p (双尾)
    p = math.erfc(abs(t) / math.sqrt(2))
    return t, p


def main():
    truth = load_sim_truth()
    print("=" * 96)
    print("Q4 两批次严格对照 (模拟器官方 jammer_count 为真值)")
    print("=" * 96)

    allres = {}
    for name, d in BATCHES.items():
        rows = []
        for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            r = parse_run(f)
            if r:
                rows.append(r)
        # 与 result.json 按时间就近匹配 (贪心一对一, 避免多日志匹配同一场景)
        used = set()
        for r in sorted(rows, key=lambda x: x["mtime"]):
            bi, best = min(
                ((i, t) for i, t in enumerate(truth) if i not in used),
                key=lambda it: abs(it[1][0] - r["mtime"]))
            used.add(bi)
            r["truth_n"] = best[1]
            r["truth_omni"] = best[2]
            r["truth_dir"] = best[3]
            r["case"] = best[4]
        allres[name] = rows

        print(f"\n### {name}  ({len(rows)} 轮)")
        print("%-32s %5s %5s %6s %6s %6s %8s %8s %8s" % (
            "日志", "真值n", "全向", "定向", "清除", "检测", "vt", "T_avg", "vt/n真值"))
        for r in rows:
            print("%-32s %5d %5d %6d %6d %6d %8.1f %8.1f %8.1f" % (
                r["file"][:32], r["truth_n"], r["truth_omni"], r["truth_dir"],
                r["cleared"], r["detected"], r["vt"],
                r["vt"] / max(r["cleared"], 1), r["vt"] / max(r["truth_n"], 1)))

    names = list(BATCHES)

    print("\n" + "=" * 100)
    print("三批汇总 (三者策略代码完全相同, 差异只能来自场景)")
    print("=" * 100)
    print("%-22s %5s %8s %9s %9s %8s" % ("批次", "轮数", "真值n", "vt(s)", "T_avg", "定向%"))
    for nm in names:
        rs = allres[nm]
        tv = [r["vt"] / max(r["cleared"], 1) for r in rs]
        dr = [r["truth_dir"] / max(r["truth_n"], 1) * 100 for r in rs]
        print("%-22s %5d %8.2f %9.1f %9.1f %8.1f" % (
            nm, len(rs), st.mean([r["truth_n"] for r in rs]),
            st.mean([r["vt"] for r in rs]), st.mean(tv), st.mean(dr)))

    print("\n" + "=" * 100)
    print("显著性检验 (基准 = 本批次 22:39+)")
    print("=" * 100)
    BASE = names[1]
    B_ = allres[BASE]
    for nm in names:
        if nm == BASE:
            continue
        A_ = allres[nm]
        print("\n--- %s  vs  %s ---" % (BASE, nm))
        print("%-22s %10s %10s %10s %10s" % ("指标", BASE[:9], nm[:9], "差异%", "p值"))
        for key, label in [("vt", "vt 总时间(s)"), ("truth_n", "真值源数 n"),
                           ("cleared", "清除数"), ("n_measure", "测向次数")]:
            a = [r[key] for r in B_]
            b = [r[key] for r in A_]
            t, p = welch_t(a, b)
            ma, mb = st.mean(a), st.mean(b)
            print("%-22s %10.1f %10.1f %+9.1f%% %10.4f" % (
                label, ma, mb, (mb - ma) / ma * 100, p))
        ta = [r["vt"] / max(r["cleared"], 1) for r in B_]
        tb = [r["vt"] / max(r["cleared"], 1) for r in A_]
        t, p = welch_t(ta, tb)
        print("%-22s %10.1f %10.1f %+9.1f%% %10.4f" % (
            "T_avg (s/个)", st.mean(ta), st.mean(tb),
            (st.mean(tb) - st.mean(ta)) / st.mean(ta) * 100, p))

    # 后面归一化反事实沿用前两批
    A, B = allres[names[0]], allres[names[1]]

    # 归一化: 若本批次源数与历史批次相同, T_avg 会是多少
    nA = st.mean([r["truth_n"] for r in A])
    vtB = st.mean([r["vt"] for r in B])
    hist_tavg = st.mean([r["vt"] / max(r["cleared"], 1) for r in A])   # 历史批次实测
    print("\n--- 归一化反事实 ---")
    print("历史批次平均真值源数 n = %.2f" % nA)
    print("本批次平均 vt        = %.1f s" % vtB)
    print("若本批次也遇到 n=%.2f 的场景, T_avg = %.1f / %.2f = %.1f s/个"
          % (nA, vtB, nA, vtB / nA))
    print("历史实测 T_avg       = %.1f s/个" % hist_tavg)
    print("残留差异             = %+.1f%%" % ((vtB / nA - hist_tavg) / hist_tavg * 100))

    # 定向源比例
    ra = [r["truth_dir"] / max(r["truth_n"], 1) for r in A]
    rb = [r["truth_dir"] / max(r["truth_n"], 1) for r in B]
    t, p = welch_t(ra, rb)
    print("\n定向源占比: 历史 %.1f%%  本批 %.1f%%  (p=%.3f)"
          % (st.mean(ra) * 100, st.mean(rb) * 100, p))

    # T_avg 对 n 的回归 (合并两批)
    xs = [r["truth_n"] for r in A + B]
    ys = [r["vt"] / max(r["cleared"], 1) for r in A + B]
    n = len(xs)
    mx, my = st.mean(xs), st.mean(ys)
    b1 = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    b0 = my - b1 * mx
    ss_t = sum((y - my) ** 2 for y in ys)
    ss_r = sum((y - (b0 + b1 * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_r / ss_t if ss_t else 0
    print("\n合并回归 T_avg = %.1f %+.1f * n   (R2=%.3f, N=%d)" % (b0, b1, r2, n))

    # 导出对照 csv
    out = os.path.join(ROOT, "logs_q4_compare.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["batch", "log", "truth_n", "truth_omni", "truth_dir",
                    "cleared", "detected", "vt_s", "T_avg_s", "n_measure", "case"])
        for name in names:
            for r in allres[name]:
                w.writerow([name, r["file"], r["truth_n"], r["truth_omni"],
                            r["truth_dir"], r["cleared"], r["detected"],
                            round(r["vt"], 1), round(r["vt"] / max(r["cleared"], 1), 1),
                            r["n_measure"], r["case"]])
    print("\n对照明细 ->", out)


if __name__ == "__main__":
    main()
