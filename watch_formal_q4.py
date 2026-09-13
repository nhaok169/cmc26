# -*- coding: utf-8 -*-
"""监控 Q4 正式测试结果并自动汇总。

Q4 正式测试权威结果：upload_tasks 表有 problem_no 字段区分问题，
statistics_tasks 表记录清除数/时间等指标。Q3 正式测试已占 statistics_tasks.id=1,2,3，
Q4 从 id=4 开始。本脚本用 upload_tasks.problem_no=4 定位 Q4 的 formal_index，
再关联 statistics_tasks 取指标。

用法:
    python watch_formal_q4.py [--poll 3]
"""
import argparse
import json
import sqlite3
import time

DB_STAT = r"E:/Draft/MathModel/cmc26/simulator/JammersSimulatorData/formal-statistics-queue.sqlite3"
DB_UPLOAD = r"E:/Draft/MathModel/cmc26/simulator/JammersSimulatorData/upload-queue.sqlite3"
OUT_JSON = r"E:/Draft/MathModel/cmc26/formal_result_q4.json"
OUT_MD = r"E:/Draft/MathModel/cmc26/formal_result_q4.md"


def read_q4_tasks():
    """读取 Q4 的正式测试：upload_tasks.problem_no=4 的 formal_index 对应 statistics_tasks.id"""
    cu = sqlite3.connect(DB_UPLOAD)
    cu.row_factory = sqlite3.Row
    q4_uploads = [dict(r) for r in cu.execute(
        "SELECT * FROM upload_tasks WHERE problem_no=4 ORDER BY formal_index")]
    cu.close()

    cs = sqlite3.connect(DB_STAT)
    cs.row_factory = sqlite3.Row
    stats = {r["id"]: dict(r) for r in cs.execute(
        "SELECT * FROM statistics_tasks ORDER BY id")}
    cs.close()

    result = []
    for up in q4_uploads:
        idx = up.get("formal_index")
        s = stats.get(idx)
        if s is None:
            continue
        result.append({
            "formal_index": idx,
            "case_code": up.get("case_code"),
            "cleared_jammer_count": s.get("cleared_jammer_count"),
            "virtual_time_us": s.get("virtual_time_us"),
            "avg_clear_time_s": round(((s.get("virtual_time_us") or 0) / 1e6 /
                                       (s.get("cleared_jammer_count") or 1)), 2),
            "program_run_duration_ms": s.get("program_run_duration_ms"),
            "measure_accepted_count": s.get("measure_accepted_count"),
            "channel_switch_count": s.get("channel_switch_count"),
            "clear_failure_count": s.get("clear_failure_count"),
            "end_reason": s.get("end_reason"),
            "state": s.get("state"),
            "formal_upload_state": up.get("formal_upload_state"),
        })
    return result


def write_outputs(detail):
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(detail, f, ensure_ascii=False, indent=2)
    lines = ["# 问题四 正式测试结果（自动汇总）", ""]
    if not detail:
        lines.append("（尚无 Q4 正式测试记录）")
    else:
        lines.append("| # | 案例编码 | 清除源数 | 平均定位清除时间(s) | 程序运行时间(ms) |")
        lines.append("|---|---|---|---|---|")
        for r in detail:
            lines.append(f"| {r['formal_index']} | {r['case_code']} | "
                         f"{r['cleared_jammer_count']} | {r['avg_clear_time_s']} | "
                         f"{r['program_run_duration_ms']} |")
        avgs = [r["avg_clear_time_s"] for r in detail if r["avg_clear_time_s"] > 0]
        lines.append("")
        lines.append(f"- 全清率: {sum(1 for r in detail if r['cleared_jammer_count']>0)}/{len(detail)}")
        if avgs:
            lines.append(f"- 平均定位清除时间: {sum(avgs)/len(avgs):.2f} s/个")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll", type=int, default=3)
    args = ap.parse_args()
    seen = set()
    print("监控 Q4 正式测试队列 (upload_tasks.problem_no=4 关联 statistics_tasks)")
    print("等待 Q4 正式测试记录写入 ... (Ctrl+C 停止)")
    while True:
        detail = read_q4_tasks()
        write_outputs(detail)
        ids = {r["formal_index"] for r in detail}
        if ids != seen:
            print(f"\n[{time.strftime('%H:%M:%S')}] Q4 正式测试记录数: {len(detail)}")
            for d in detail:
                if d["formal_index"] not in seen:
                    print(f"  #{d['formal_index']} {d['case_code']} 清除={d['cleared_jammer_count']} "
                          f"avg={d['avg_clear_time_s']}s/个 "
                          f"程序运行={d['program_run_duration_ms']}ms "
                          f"cfail={d['clear_failure_count']} state={d['state']} "
                          f"upload={d['formal_upload_state']}")
            seen = ids
            if len(detail) >= 3:
                print("\n=== 已收集到 3 次 Q4 正式测试，结果保存到 formal_result_q4.json / .md ===")
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
