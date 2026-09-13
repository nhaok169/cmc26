# -*- coding: utf-8 -*-
"""启动 Q4 环抱证书策略 (v3 三段式合并巡回) 连接真实测试服务器.

可从任意工作目录运行: python q4/run_q4.py  或  cd q4 && python run_q4.py
    python q4/run_q4.py --watch
"""
import argparse
import datetime
import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, "q2"))
import robot as R
from policy import Policy3

TEAM = "202601006115"
BASE = "http://127.0.0.1:2026"
LOGDIR = os.path.join(_root, "logs")


def make_policy(sim):
    return Policy3(sim)


def run_once(wait_s=600):
    os.makedirs(LOGDIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOGDIR, f"robot_p4_{stamp}.jsonl")
    log = open(log_path, "w", encoding="utf-8")
    print(f"robot_id={TEAM}, 问题4, 接口 {BASE}")
    print(f"日志: {log.name}")
    print("等待测试窗口开启...")
    sim = R.Sim(BASE, TEAM, log)
    sim.enter(wait_s=wait_s)
    print("[ENTER] 成功进入, 开始 Q4 v3 策略")
    make_policy(sim).run()
    try:
        sim.exit()
        print("[EXIT] 已退出")
    except Exception as e:
        print(f"[EXIT] 失败: {e}")
    log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="持续监听测试窗口, 一局结束再等下一局")
    ap.add_argument("--wait", type=int, default=600, help="单局等待窗口秒数(仅非 --watch)")
    args = ap.parse_args()
    if args.watch:
        R.watch_loop(BASE, TEAM, 4, make_policy, "Q4 v3")
        return
    run_once(wait_s=args.wait)


if __name__ == "__main__":
    main()
