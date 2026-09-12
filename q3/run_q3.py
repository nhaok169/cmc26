# -*- coding: utf-8 -*-
"""启动 v6 自适应追踪策略连接真实测试服务器."""
import sys, os, datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "q3"))
import robot as R
from policy import Policy

TEAM = "202601006115"
BASE = "http://127.0.0.1:2026"

os.makedirs("logs", exist_ok=True)
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log = open(f"logs/robot_p3_{stamp}.jsonl", "w", encoding="utf-8")

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
