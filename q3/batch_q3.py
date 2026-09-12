# -*- coding: utf-8 -*-
"""Q3 批量测试 runner.

连续跑 N 轮真实模拟器演练, 每轮:
    /enter -> Policy(自适应追踪) -> /exit
产物:
    1) 每轮一条 jsonl 原始日志 (指定目录, 默认 cmc26/logs_q3_batch/)
    2) 每轮一条 console 文本日志 (.log)
    3) 逐轮追加的 summary.csv (崩溃也保留已完成轮次)
    4) 结束后由 summarize_q3.py 生成 xlsx 统计

用法:
    python cmc26/q3/batch_q3.py --rounds 10
    python cmc26/q3/batch_q3.py --rounds 20 --gap 5 --wait 600
"""
import argparse
import csv
import datetime
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
for _ in range(4):                       # 向上最多找 4 层, 定位含 robot.py 的仓库根
    if os.path.isfile(os.path.join(_ROOT, "robot.py")):
        break
    _ROOT = os.path.dirname(_ROOT)
for _p in (_ROOT, os.path.join(_ROOT, "q3"), os.path.join(_ROOT, "q2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import requests                          # noqa: E402
import robot as R                        # noqa: E402
from policy import Policy                # noqa: E402

DEFAULT_OUTDIR = os.path.join(_ROOT, "logs_q3_batch")
CSV_FIELDS = [
    "round", "log_file", "start_time", "end_time", "enter_ok", "exit_ok",
    "cleared", "vt_total_s", "vt_total_h", "avg_per_source_s",
    "measure_cnt", "switch_cnt", "clear_attempts", "clear_fail",
    "channels_detected", "real_elapsed_s", "records", "error",
]


class Tee(object):
    """同时写控制台和文件."""

    def __init__(self, stream, fobj):
        self.stream = stream
        self.fobj = fobj

    def write(self, data):
        self.stream.write(data)
        self.fobj.write(data)
        self.fobj.flush()

    def flush(self):
        self.stream.flush()
        self.fobj.flush()


def enter_round(sim, wait_s, rid, log):
    """带自定义 request_id 的 /enter 轮询 (原 Sim.enter 固定用 enter-1)."""
    body = {"arena_id": "default", "robot_id": sim.team, "request_id": rid}
    t_end = time.time() + wait_s
    tried = 0
    while True:
        try:
            r = requests.post(sim.base + "/enter", json=body, timeout=5,
                              headers={"Content-Type": "application/json; charset=utf-8"})
            code, resp = r.status_code, None
            try:
                resp = r.json()
            except ValueError:
                pass
        except requests.RequestException as e:
            code, resp = 0, repr(e)
        tried += 1
        if code == 200 and resp and resp.get("accepted"):
            sim.t0 = time.time()
            sim.vt = resp["virtual_time_s"]
            log.write(__import__("json").dumps(
                {"t": datetime.datetime.now().isoformat(), "rid": rid, "path": "/enter",
                 "http": code, "resp": resp}, ensure_ascii=False) + "\n")
            log.flush()
            print(f"  [ENTER] ok, 剩余现实时间 {resp.get('remaining_real_duration_s')}s")
            return resp
        if time.time() > t_end:
            raise RuntimeError(f"等待测试窗口超时({wait_s}s, 尝试{tried}次): HTTP {code} {resp}")
        if tried % 15 == 0:
            print(f"  ...等待窗口开启 ({int(t_end - time.time())}s 后放弃)")
        time.sleep(1.0)


def write_csv_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def run_one_round(idx, args, outdir, csv_path):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"r{idx:02d}_{stamp}"
    log_path = os.path.join(outdir, f"{tag}.jsonl")
    con_path = os.path.join(outdir, f"{tag}.console.log")

    print(f"\n{'='*66}\n[轮次 {idx}/{args.rounds}] 开始 {stamp}\n  日志: {log_path}\n{'='*66}")

    row = {k: "" for k in CSV_FIELDS}
    row.update({"round": idx, "log_file": os.path.basename(log_path),
                "start_time": stamp, "enter_ok": 0, "exit_ok": 0})

    logf = open(log_path, "w", encoding="utf-8")
    conf = open(con_path, "w", encoding="utf-8")
    old_stdout = sys.stdout
    sys.stdout = Tee(old_stdout, conf)
    try:
        sim = R.Sim(args.base, args.team, logf)
        try:
            enter_round(sim, wait_s=args.wait, rid=f"enter-{tag}", log=logf)
            row["enter_ok"] = 1
            print("[ENTER] 成功进入, 开始 v6 自适应追踪策略")
        except Exception as e:
            row["error"] = f"enter: {e}"
            print(f"[ERROR] 进入失败: {e}")
            return row

        try:
            policy = Policy(sim, alpha=args.alpha, theta=args.theta)
            n, vt = policy.run()
        except Exception as e:
            print(f"[ERROR] 策略异常: {e}")
            n, vt = len(sim.cleared), sim.vt
            row["error"] = f"policy: {e}"

        try:
            sim.exit()
            row["exit_ok"] = 1
            print("[EXIT] 已退出")
        except Exception as e:
            print(f"[EXIT] 失败: {e}")
            row["error"] = (row["error"] + " | " if row["error"] else "") + f"exit: {e}"

        row.update({
            "cleared": len(sim.cleared),
            "vt_total_s": round(sim.vt, 3),
            "vt_total_h": round(sim.vt / 3600.0, 4),
            "avg_per_source_s": round(sim.vt / n, 3) if n else "",
            "measure_cnt": sim.measure_cnt,
            "switch_cnt": sim.switch_cnt,
            "clear_fail": sim.clear_fail,
            "channels_detected": len(sim.cleared),
            "real_elapsed_s": round(sim.elapsed_real(), 2),
        })
        print(f">>> 轮次 {idx} 结果: 清除 {len(sim.cleared)} 个, "
              f"虚拟时间 {sim.vt:.1f}s, 平均 {row['avg_per_source_s']} s/个")
    finally:
        sys.stdout = old_stdout
        try:
            logf.close()
        except Exception:
            pass
        try:
            conf.close()
        except Exception:
            pass
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                row["records"] = sum(1 for _ in f if _.strip())
        except Exception:
            pass
        write_csv_row(csv_path, row)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10, help="批跑轮数")
    ap.add_argument("--team", default="202601006115")
    ap.add_argument("--port", type=int, default=2026)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--wait", type=int, default=600, help="每轮等待窗口开启的秒数")
    ap.add_argument("--gap", type=float, default=3.0, help="两轮之间的现实间隔秒")
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--theta", type=float, default=2.05)
    args = ap.parse_args()

    args.base = f"http://127.0.0.1:{args.port}"
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "summary.csv")

    print(f"robot_id={args.team}  问题3  接口 {args.base}")
    print(f"轮数={args.rounds}  日志目录={outdir}")
    print(f"汇总CSV={csv_path}")

    t_all = time.time()
    done = []
    for i in range(1, args.rounds + 1):
        try:
            done.append(run_one_round(i, args, outdir, csv_path))
        except Exception as e:
            print(f"[FATAL] 轮次 {i} 未捕获异常: {e}")
            import traceback
            traceback.print_exc()
        if i < args.rounds:
            print(f"-- 间隔 {args.gap}s 后开始下一轮 (若模拟器需手动开始, 请现在点击) --")
            time.sleep(args.gap)

    ok = [r for r in done if r.get("cleared") != ""]
    print(f"\n{'='*66}\n批量测试结束, 共 {len(done)} 轮, 耗时 {time.time()-t_all:.1f}s")
    if ok:
        vt = [r["vt_total_s"] for r in ok]
        cl = [r["cleared"] for r in ok]
        avg = [r["avg_per_source_s"] for r in ok if r["avg_per_source_s"] != ""]
        print(f"清除数: min={min(cl)} max={max(cl)} 平均={sum(cl)/len(cl):.2f}")
        print(f"虚拟总时间: min={min(vt):.1f}s max={max(vt):.1f}s 平均={sum(vt)/len(vt):.1f}s")
        if avg:
            print(f"平均定位清除时间: min={min(avg):.1f}s max={max(avg):.1f}s 平均={sum(avg)/len(avg):.1f}s")
    print(f"下一步: python {os.path.join(_HERE, 'summarize_q3.py')} --indir \"{outdir}\"")


if __name__ == "__main__":
    main()
