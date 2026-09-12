# -*- coding: utf-8 -*-
"""分析真机测试日志."""
import json, sys

log_file = sys.argv[1] if len(sys.argv) > 1 else "logs/robot_p4_20260912_141821.jsonl"
lines = open(log_file, encoding="utf-8").readlines()
events = [json.loads(l) for l in lines if l.strip()]

measures = [e for e in events if e.get("path") == "/measure"]
clears = [e for e in events if e.get("path") == "/clear"]
dirs = [e for e in measures if e["resp"].get("measure_result") == "direction"]
no_sig = [e for e in measures if e["resp"].get("measure_result") == "no_signal"]
near = [e for e in measures if e["resp"].get("measure_result") == "near"]
clear_ok = [e for e in clears if e["resp"].get("clear_result") == "success"]
clear_fail = [e for e in clears if e["resp"].get("clear_result") == "no_target_in_range"]

ch_dirs = {}
for e in dirs:
    ch = e["req"]["channel"]
    ch_dirs[ch] = ch_dirs.get(ch, 0) + 1

first_clear_vt = min(e["resp"]["virtual_time_s"] for e in clear_ok)
last_clear_vt = max(e["resp"]["virtual_time_s"] for e in clear_ok)

grid_start_vt = None
for e in measures:
    pos = e["req"]["position"]
    d = (pos["x"]**2 + pos["y"]**2)**0.5
    if d > 2000:
        grid_start_vt = e["resp"]["virtual_time_s"]
        break

last_vt = events[-1]["resp"]["virtual_time_s"]

print(f"总测量: {len(measures)} (direction={len(dirs)}, no_signal={len(no_sig)}, near={len(near)})")
print(f"清除: {len(clear_ok)} 成功, {len(clear_fail)} 失败 (浪费 {len(clear_fail)*5}s)")
print(f"检出方向的频道: {ch_dirs}")
print(f"首次清除 VT: {first_clear_vt:.0f}s")
print(f"末次清除 VT: {last_clear_vt:.0f}s")
if grid_start_vt:
    print(f"外圈格点开始 VT: {grid_start_vt:.0f}s")
    print(f"grid_cover 外圈阶段: {grid_start_vt:.0f}s - {last_vt:.0f}s ({last_vt-grid_start_vt:.0f}s)")
else:
    print("无外圈格点")
print(f"总 VT: {last_vt:.0f}s, 清除率: {len(clear_ok)}/15")

# Per-channel clear failures
ch_fails = {}
for e in clear_fail:
    ch = e["req"]["channel"]
    ch_fails[ch] = ch_fails.get(ch, 0) + 1
print(f"按频道清除失败: {ch_fails}")

# Movement analysis
total_move = 0
for i in range(1, len(measures)):
    p0 = measures[i-1]["req"]["position"]
    p1 = measures[i]["req"]["position"]
    total_move += ((p1["x"]-p0["x"])**2 + (p1["y"]-p0["y"])**2)**0.5
print(f"总移动距离: {total_move:.0f}m ({total_move/5:.0f}s)")
