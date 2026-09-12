# -*- coding: utf-8 -*-
"""启动 v6 自适应追踪策略连接真实测试服务器.

可从任意工作目录运行: python cmc26/q3/run_q3.py  或  cd cmc26/q3 && python run_q3.py
需要的三个路径:
  仓库根 (robot.py)  /  q3 (policy, ledger, scheduler, score)  /  q2 (geometry)
"""
import sys, os, datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
for _ in range(4):                       # 向上最多找 4 层
    if os.path.isfile(os.path.join(_ROOT, "robot.py")):
        break
    _ROOT = os.path.dirname(_ROOT)
for _p in (_ROOT, os.path.join(_ROOT, "q3"), os.path.join(_ROOT, "q2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import robot as R
from policy import Policy

TEAM = "202601006115"
BASE = "http://127.0.0.1:2026"

# 日志统一写到仓库根的 logs/ (与 Q4 一致), 而非当前工作目录
_logdir = os.path.join(_ROOT, "logs")
os.makedirs(_logdir, exist_ok=True)
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log = open(os.path.join(_logdir, f"robot_p3_{stamp}.jsonl"), "w", encoding="utf-8")

print(f"robot_id={TEAM}, 问题3, 接口 {BASE}")
print(f"日志: {log.name}")
print("等待测试窗口开启...")

sim = R.Sim(BASE, TEAM, log)
sim.enter(wait_s=600)
print("[ENTER] 成功进入, 开始 v6 策略")

policy = Policy(sim, alpha=2.0, theta=2.05)
n, vt = policy.run()

try:
    sim.exit()
    print("[EXIT] 已退出")
except Exception as e:
    print(f"[EXIT] 失败: {e}")

log.close()
