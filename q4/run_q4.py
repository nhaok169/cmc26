# -*- coding: utf-8 -*-
"""启动 Q4 环抱证书策略 (v3 三段式合并巡回) 连接真实测试服务器.

可从任意工作目录运行: python cmc26/q4/run_q4.py  或  cd cmc26/q4 && python run_q4.py
"""
import sys, os, datetime

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)                         # 仓库根 (robot.py)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # q4/ (policy, ledger)
sys.path.insert(0, os.path.join(_root, "q2"))     # q2/ (geometry.py)
import robot as R
from policy import Policy3

TEAM = "202601006115"
BASE = "http://127.0.0.1:2026"

# 日志统一写到仓库根的 logs/ (与 Q3 一致), 而非当前工作目录
_logdir = os.path.join(_root, "logs")
os.makedirs(_logdir, exist_ok=True)
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log = open(os.path.join(_logdir, f"robot_p4_{stamp}.jsonl"), "w", encoding="utf-8")

print(f"robot_id={TEAM}, 问题4, 接口 {BASE}")
print(f"日志: {log.name}")
print("等待测试窗口开启...")

sim = R.Sim(BASE, TEAM, log)
sim.enter(wait_s=600)
print("[ENTER] 成功进入, 开始 Q4 v3 策略")

policy = Policy3(sim)
n, vt = policy.run()

try:
    sim.exit()
    print("[EXIT] 已退出")
except Exception as e:
    print(f"[EXIT] 失败: {e}")

log.close()
