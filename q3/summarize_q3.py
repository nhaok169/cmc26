# -*- coding: utf-8 -*-
"""把 Q3 批量测试的 jsonl 日志汇总成 xlsx (+ 同名 CSV), 供后续作图分析.

用法:
    python cmc26/q3/summarize_q3.py
    python cmc26/q3/summarize_q3.py --indir E:/Draft/MathModel/cmc26/logs_q3_batch
    python cmc26/q3/summarize_q3.py --indir E:/Draft/MathModel/logs   # 也兼容旧日志目录

输出 (默认写在 --indir 下):
    q3_stats.xlsx   多 sheet: 汇总统计 / 每轮明细 / 每频道明细 / 阶段分解 / 操作分解 / 事件序列
    stats_csv/*.csv 同样内容的 CSV (方便其它 AI / 脚本直接读取)
"""
import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

SPEED = 5.0
MEASURE_COST = 5.0
SWITCH_COST = 1.0
CLEAR_OK_COST = 5.0     # 光学定位 3s + 激光清除 2s
CLEAR_FAIL_COST = 3.0
ARENA_R = 1800.0

# 补网阶段使用的骨架环点 (原点 + 1150m 正六边形)
RING_PTS = [(1150.0 * math.cos(i * math.pi / 3), 1150.0 * math.sin(i * math.pi / 3))
            for i in range(6)]


def is_ring_point(p, tol=1.0):
    return any(math.hypot(p[0] - q[0], p[1] - q[1]) <= tol for q in RING_PTS)


def parse_log(path, problem=3):
    """解析单轮日志 -> (run_row, channel_rows, phase_rows, op_rows, event_rows)

    problem=3: 阶段按 Q3 语义切分 (初始扫描 / 追踪清除 / 补网扫描),
               补网起点 = 首次到达 1150m 骨架环点。
    problem=4: Q4 是"证书巡回+定位+清除"合并的一条龙, 不再有独立补网阶段,
               故只切两段 (初始扫描 / 证书巡回)。
    """
    name = os.path.basename(path)
    m = re.match(r"r(\d+)_", name)
    run_id = int(m.group(1)) if m else 0

    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not recs:
        return None

    seq = [r for r in recs if isinstance(r.get("resp"), dict)
           and "virtual_time_s" in r["resp"]]
    acts = [r for r in seq if r.get("path") in ("/measure", "/clear")]

    # ---------- 事件序列 ----------
    event_rows = []
    prev_vt = 0.0
    for i, r in enumerate(seq):
        req = r.get("req", {}) or {}
        pos = req.get("position", {}) or {}
        vt = float(r["resp"]["virtual_time_s"])
        event_rows.append({
            "run": run_id, "log_file": name, "idx": i,
            "vt_s": round(vt, 3), "dt_s": round(vt - prev_vt, 3),
            "action": r["path"].lstrip("/"),
            "channel": req.get("channel", ""),
            "x": pos.get("x", ""), "y": pos.get("y", ""),
            "result": r["resp"].get("measure_result") or r["resp"].get("clear_result") or "",
        })
        prev_vt = vt
    vt_total = prev_vt

    measures = [r for r in acts if r["path"] == "/measure"]
    clears = [r for r in acts if r["path"] == "/clear"]

    # ---------- 每频道明细 ----------
    ch_det = {}          # ch -> first detect vt
    ch_bearings = defaultdict(int)
    ch_clear_try = defaultdict(int)
    ch_clear_fail = defaultdict(int)
    ch_cleared = {}      # ch -> clear vt
    ch_det_pos = {}

    for r in measures:
        ch = r["req"]["channel"]
        if r["resp"].get("measure_result") == "direction":
            ch_bearings[ch] += 1
            if ch not in ch_det:
                ch_det[ch] = float(r["resp"]["virtual_time_s"])
                p = r["req"]["position"]
                ch_det_pos[ch] = math.hypot(p["x"], p["y"])
    for r in clears:
        ch = r["req"]["channel"]
        ch_clear_try[ch] += 1
        if r["resp"].get("clear_result") == "success":
            if ch not in ch_cleared:
                ch_cleared[ch] = float(r["resp"]["virtual_time_s"])
        else:
            ch_clear_fail[ch] += 1

    channel_rows = []
    for ch in range(1, 21):
        det = ch in ch_det
        cl = ch in ch_cleared
        channel_rows.append({
            "run": run_id, "log_file": name, "channel": ch,
            "detected": int(det),
            "first_detect_vt_s": round(ch_det[ch], 3) if det else "",
            "first_detect_radius_m": round(ch_det_pos[ch], 1) if det else "",
            "n_bearings": ch_bearings.get(ch, 0),
            "cleared": int(cl),
            "clear_vt_s": round(ch_cleared[ch], 3) if cl else "",
            "locate_clear_time_s": round(ch_cleared[ch] - ch_det[ch], 3) if (det and cl) else "",
            "n_clear_attempts": ch_clear_try.get(ch, 0),
            "n_clear_fail": ch_clear_fail.get(ch, 0),
            "clear_success_rate": (round(1 - ch_clear_fail.get(ch, 0) / ch_clear_try[ch], 3)
                                   if ch_clear_try.get(ch, 0) else ""),
        })

    # ---------- 阶段分解 ----------
    # 初始扫描: 原点 20 频道; 补网: 首次到达骨架环点(1150m)之后
    origin_measures = [r for r in measures
                       if abs(r["req"]["position"]["x"]) < 1e-6
                       and abs(r["req"]["position"]["y"]) < 1e-6]
    init_vt = float(origin_measures[-1]["resp"]["virtual_time_s"]) if origin_measures else 0.0
    if problem == 4:
        phase_rows = [
            {"run": run_id, "log_file": name, "phase": "初始扫描(原点20频道)",
             "seconds": round(init_vt, 3)},
            {"run": run_id, "log_file": name,
             "phase": "证书巡回(26点环抱+定位清除)",
             "seconds": round(max(vt_total - init_vt, 0.0), 3)},
        ]
        hunt_vt, ring_vt = vt_total - init_vt, 0.0
    else:
        ring_start = None
        for r in measures:
            p = r["req"]["position"]
            if is_ring_point((p["x"], p["y"])):
                ring_start = float(r["resp"]["virtual_time_s"])
                break
        if ring_start is None:
            hunt_vt, ring_vt = vt_total - init_vt, 0.0
        else:
            hunt_vt, ring_vt = ring_start - init_vt, vt_total - ring_start
        phase_rows = [
            {"run": run_id, "log_file": name, "phase": "初始扫描(原点20频道)",
             "seconds": round(init_vt, 3)},
            {"run": run_id, "log_file": name, "phase": "追踪清除(hunt)",
             "seconds": round(max(hunt_vt, 0.0), 3)},
            {"run": run_id, "log_file": name, "phase": "补网扫描(ring)",
             "seconds": round(max(ring_vt, 0.0), 3)},
        ]

    # ---------- 操作分解 ----------
    # 注意: 必须遍历全部动作(measure + clear)累加相邻位移,
    # 只遍历 measures 会漏掉"测向点 -> 清除点"的位移(实测低估 ~2000m/轮)。
    prev_pos = (0.0, 0.0)
    prev_ch = 1
    move_d = 0.0
    t_move = t_measure = t_switch = 0.0
    for r in acts:
        p = r["req"]["position"]
        pos = (p["x"], p["y"])
        ch = r["req"]["channel"]
        d = math.hypot(pos[0] - prev_pos[0], pos[1] - prev_pos[1])
        move_d += d
        t_move += d / SPEED
        if r["path"] == "/measure":
            t_measure += MEASURE_COST
        if ch != prev_ch:
            t_switch += SWITCH_COST
        prev_pos, prev_ch = pos, ch

    n_clear_ok = sum(1 for r in clears if r["resp"].get("clear_result") == "success")
    n_clear_fail = len(clears) - n_clear_ok
    t_clear_ok = n_clear_ok * CLEAR_OK_COST
    t_clear_fail = n_clear_fail * CLEAR_FAIL_COST
    # clear 动作本身还包含移动到清除点的耗时
    t_move += 0.0
    other = vt_total - t_move - t_measure - t_switch - t_clear_ok - t_clear_fail
    op_rows = [
        {"run": run_id, "log_file": name, "category": "移动", "seconds": round(t_move, 3)},
        {"run": run_id, "log_file": name, "category": "测向", "seconds": round(t_measure, 3)},
        {"run": run_id, "log_file": name, "category": "换频", "seconds": round(t_switch, 3)},
        {"run": run_id, "log_file": name, "category": "清除(成功)", "seconds": round(t_clear_ok, 3)},
        {"run": run_id, "log_file": name, "category": "清除(失败)", "seconds": round(t_clear_fail, 3)},
        {"run": run_id, "log_file": name, "category": "未解释/余量", "seconds": round(other, 3)},
    ]

    # ---------- 每轮汇总 ----------
    n_cleared = len(ch_cleared)
    n_detected = len(ch_det)
    ts = [r["t"] for r in recs if r.get("t")]
    real_elapsed = ""
    if len(ts) >= 2:
        try:
            from datetime import datetime
            t0 = datetime.fromisoformat(ts[0])
            t1 = datetime.fromisoformat(ts[-1])
            real_elapsed = round((t1 - t0).total_seconds(), 2)
        except Exception:
            pass

    run_row = {
        "run": run_id, "log_file": name,
        "start_time": ts[0] if ts else "",
        "cleared": n_cleared,
        "detected": n_detected,
        "missed": max(0, n_detected - n_cleared),
        "vt_total_s": round(vt_total, 3),
        "vt_total_h": round(vt_total / 3600.0, 4),
        "avg_per_source_s": round(vt_total / n_cleared, 3) if n_cleared else "",
        "avg_locate_clear_s": round(
            np.mean([c["locate_clear_time_s"] for c in channel_rows
                     if c["locate_clear_time_s"] != ""]), 3) if n_cleared else "",
        "n_measure": len(measures),
        "n_clear": len(clears),
        "n_clear_fail": n_clear_fail,
        "clear_success_rate": round(n_clear_ok / len(clears), 3) if clears else "",
        "avg_bearings_per_source": round(
            sum(ch_bearings[c] for c in ch_cleared) / n_cleared, 2) if n_cleared else "",
        "n_switch": int(t_switch / SWITCH_COST),
        "move_distance_m": round(move_d, 1),
        "move_time_s": round(t_move, 1),
        "measure_time_s": round(t_measure, 1),
        "switch_time_s": round(t_switch, 1),
        "clear_time_s": round(t_clear_ok + t_clear_fail, 1),
        "other_time_s": round(other, 1),
        "init_time_s": round(init_vt, 1),
        "hunt_time_s": round(max(hunt_vt, 0.0), 1),
        "ring_time_s": round(max(ring_vt, 0.0), 1),
        "real_elapsed_s": real_elapsed,
        "n_records": len(recs),
    }
    return run_row, channel_rows, phase_rows, op_rows, event_rows


def describe(df, metrics):
    rows = []
    for c in metrics:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append({
            "指标": c, "样本数": int(s.size), "均值": round(float(s.mean()), 3),
            "标准差": round(float(s.std(ddof=1)), 3) if s.size > 1 else 0.0,
            "最小值": round(float(s.min()), 3),
            "25分位": round(float(s.quantile(0.25)), 3),
            "中位数": round(float(s.median()), 3),
            "75分位": round(float(s.quantile(0.75)), 3),
            "最大值": round(float(s.max()), 3),
        })
    return pd.DataFrame(rows)


def autosize(writer, sheet, df):
    ws = writer.sheets[sheet]
    for i, col in enumerate(df.columns, start=1):
        try:
            w = max(len(str(col)) * 2 + 2,
                    int(df[col].astype(str).str.len().quantile(0.9)) + 4)
        except Exception:
            w = 14
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = min(max(w, 10), 34)
    ws.freeze_panes = "A2"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default=os.path.join(root, "logs_q3_batch"))
    ap.add_argument("--pattern", default="*.jsonl")
    ap.add_argument("--problem", type=int, default=3, choices=[3, 4],
                    help="3=Q3(阶段按 hunt/ring 切分), 4=Q4(合并巡回, 阶段只切两段)")
    ap.add_argument("--out", default="", help="xlsx 输出路径, 默认 <indir>/q{3|4}_stats.xlsx")
    args = ap.parse_args()

    indir = os.path.abspath(args.indir)
    files = sorted(glob.glob(os.path.join(indir, args.pattern)))
    if not files:
        print(f"[WARN] {indir} 下没有 {args.pattern}")
        return

    runs, chans, phases, ops, events = [], [], [], [], []
    for f in files:
        got = parse_log(f, problem=args.problem)
        if not got:
            continue
        r, c, p, o, e = got
        runs.append(r)
        chans.extend(c)
        phases.extend(p)
        ops.extend(o)
        events.extend(e)
        print(f"  {r['log_file']}: 清除 {r['cleared']}/{r['detected']}, "
              f"vt={r['vt_total_s']}s, 平均 {r['avg_per_source_s']}s/个")

    df_run = pd.DataFrame(runs).sort_values("run").reset_index(drop=True)
    df_ch = pd.DataFrame(chans)
    df_ph = pd.DataFrame(phases)
    df_op = pd.DataFrame(ops)
    df_ev = pd.DataFrame(events)

    tot = max(df_ph["seconds"].sum(), 1e-9)
    df_ph["pct_of_run"] = df_ph.groupby("run")["seconds"].transform(
        lambda s: (s / s.sum() * 100).round(2))
    df_op["pct_of_run"] = df_op.groupby("run")["seconds"].transform(
        lambda s: (s / s.sum() * 100).round(2))

    metrics = ["cleared", "detected", "missed", "vt_total_s", "vt_total_h",
               "avg_per_source_s", "avg_locate_clear_s", "n_measure", "n_clear",
               "n_clear_fail", "clear_success_rate", "avg_bearings_per_source",
               "n_switch", "move_distance_m", "move_time_s", "measure_time_s",
               "switch_time_s", "clear_time_s", "other_time_s",
               "init_time_s", "hunt_time_s", "ring_time_s", "real_elapsed_s"]
    df_sum = describe(df_run, metrics)

    out = args.out or os.path.join(indir, f"q{args.problem}_stats.xlsx")
    stats_dir = os.path.join(indir, "stats_csv")
    os.makedirs(stats_dir, exist_ok=True)

    with pd.ExcelWriter(out, engine="openpyxl") as w:
        for name, df in [("汇总统计", df_sum), ("每轮明细", df_run),
                         ("每频道明细", df_ch), ("阶段分解", df_ph),
                         ("操作分解", df_op), ("事件序列", df_ev)]:
            df.to_excel(w, sheet_name=name, index=False)
            autosize(w, name, df)
        for name, df in [("汇总统计", df_sum), ("每轮明细", df_run),
                         ("每频道明细", df_ch), ("阶段分解", df_ph),
                         ("操作分解", df_op), ("事件序列", df_ev)]:
            df.to_csv(os.path.join(stats_dir, f"{name}.csv"),
                      index=False, encoding="utf-8-sig")

    print(f"\n共解析 {len(df_run)} 轮")
    print(f"xlsx : {out}")
    print(f"csv  : {stats_dir}")
    if not df_sum.empty:
        key = df_sum[df_sum["指标"].isin(
            ["cleared", "vt_total_s", "avg_per_source_s", "avg_locate_clear_s",
             "n_measure", "clear_success_rate"])]
        print("\n--- 关键指标 ---")
        print(key.to_string(index=False))


if __name__ == "__main__":
    main()
