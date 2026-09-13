# -*- coding: utf-8 -*-
"""监控正式测试结果并自动汇总。

正式测试的权威结果由模拟器自动写入 formal-statistics-queue.sqlite3 的
statistics_tasks 表（每个正式测试一局一行）。本脚本轮询该表，一旦出现
新记录就打印并保存汇总，用于后续填入论文 tab:formal_test_q3。

用法:
    python watch_formal_q3.py [--poll 5] [--once]

输出:
    formal_result_q3.json   正式测试逐局明细 + 汇总
    formal_result_q3.md     人类可读汇总
"""
import argparse
import json
import os
import sqlite3
import sys
import time

DB = r"E:/Draft/MathModel/cmc26/simulator/JammersSimulatorData/formal-statistics-queue.sqlite3"
OUT_JSON = r"E:/Draft/MathModel/cmc26/formal_result_q3.json"
OUT_MD = r"E:/Draft/MathModel/cmc26/formal_result_q3.md"


def read_tasks():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    rows = [dict(r) for r in c.execute(
        "SELECT * FROM statistics_tasks ORDER BY id")]
    c.close()
    return rows


def summarize(rows):
    out = []
    for r in rows:
        vt_us = r.get("virtual_time_us") or 0
        n = r.get("cleared_jammer_count") or 0
        avg = (vt_us / 1e6 / n) if n else 0.0
        dur_ms = r.get("program_run_duration_ms") or 0
        out.append({
            "id": r.get("id"),
            "team_no": r.get("team_no"),
            "entered": r.get("entered"),
            "end_reason": r.get("end_reason"),
            "cleared_jammer_count": n,
            "virtual_time_us": vt_us,
            "avg_clear_time_s": round(avg, 2),
            "program_run_duration_ms": dur_ms,
            "measure_accepted_count": r.get("measure_accepted_count"),
            "channel_switch_count": r.get("channel_switch_count"),
            "clear_failure_count": r.get("clear_failure_count"),
            "state": r.get("state"),
            "activation_ticket_sha256": (r.get("activation_ticket_sha256") or "")[:16],
        })
    return out


def write_outputs(detail):
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(detail, f, ensure_ascii=False, indent=2)
    # md 汇总
    lines = ["# 问题三 正式测试结果（自动汇总）", ""]
    if not detail:
        lines.append("（尚无正式测试记录）")
    else:
        lines.append("| # | 案例编码 | 清除源数 | 平均定位清除时间(s) | 程序运行时间(ms) |")
        lines.append("|---|---|---|---|---|")
        for i, r in enumerate(detail, 1):
            code = r.get("id")  # statistics_tasks 无 case_code, 用 id 占位
            lines.append(f"| {i} | {code} | {r['cleared_jammer_count']} | "
                         f"{r['avg_clear_time_s']} | {r['program_run_duration_ms']} |")
        # 汇总
        ns = [r["cleared_jammer_count"] for r in detail]
        avgs = [r["avg_clear_time_s"] for r in detail if r["avg_clear_time_s"] > 0]
        lines.append("")
        lines.append(f"- 全清率: {sum(1 for r in detail if r['cleared_jammer_count']>0)}/{len(detail)}")
        if avgs:
            lines.append(f"- 平均定位清除时间: {sum(avgs)/len(avgs):.2f} s/个")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll", type=int, default=5, help="轮询间隔秒数")
    ap.add_argument("--once", action="store_true", help="只读一次不循环")
    args = ap.parse_args()

    seen = set()
    print(f"监控正式测试队列: {DB}")
    print("等待正式测试记录写入 statistics_tasks ... (Ctrl+C 停止)")
    while True:
        rows = read_tasks()
        detail = summarize(rows)
        write_outputs(detail)
        if len(rows) != len(seen):
            print(f"\n[{time.strftime('%H:%M:%S')}] 正式测试记录数: {len(rows)}")
            for d in detail:
                if d["id"] not in seen:
                    print(f"  #{d['id']} 清除={d['cleared_jammer_count']} "
                          f"avg={d['avg_clear_time_s']}s/个 "
                          f"程序运行={d['program_run_duration_ms']}ms "
                          f"end={d['end_reason']} state={d['state']}")
            seen = {r["id"] for r in rows}
            if len(rows) >= 3:
                print("\n=== 已收集到 3 次正式测试，结果已保存到 formal_result_q3.json / .md ===")
        if args.once:
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
