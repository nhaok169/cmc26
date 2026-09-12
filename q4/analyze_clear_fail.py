# -*- coding: utf-8 -*-
"""分析清除失败的频道详情."""
import json, sys, math

log_file = sys.argv[1] if len(sys.argv) > 1 else "logs/robot_p4_20260912_141821.jsonl"
lines = open(log_file, encoding="utf-8").readlines()
events = [json.loads(l) for l in lines if l.strip()]

# 按频道收集所有测量和清除事件
ch_data = {}
for e in events:
    if "req" not in e:
        continue
    ch = e["req"].get("channel")
    if ch is None:
        continue
    if ch not in ch_data:
        ch_data[ch] = {"measures": [], "clears": []}
    if e["path"] == "/measure":
        ch_data[ch]["measures"].append(e)
    elif e["path"] == "/clear":
        ch_data[ch]["clears"].append(e)

# 分析每个有清除失败的频道
for ch in sorted(ch_data.keys()):
    data = ch_data[ch]
    fails = [c for c in data["clears"] if c["resp"].get("clear_result") == "no_target_in_range"]
    oks = [c for c in data["clears"] if c["resp"].get("clear_result") == "success"]
    if not fails and not oks:
        continue

    dirs = [m for m in data["measures"] if m["resp"].get("measure_result") == "direction"]
    nears = [m for m in data["measures"] if m["resp"].get("measure_result") == "near"]

    print(f"\n=== 频道 {ch} ===")
    print(f"  测量: {len(data['measures'])} 次 (direction={len(dirs)}, near={len(nears)})")
    print(f"  清除: {len(oks)} 成功, {len(fails)} 失败")

    if dirs:
        print(f"  检出方向 (前5次):")
        for d in dirs[:5]:
            pos = d["req"]["position"]
            deg = d["resp"]["svd_deg"]
            vt = d["resp"]["virtual_time_s"]
            print(f"    VT={vt:.0f}s pos=({pos['x']:.0f},{pos['y']:.0f}) bearing={deg}")

    if fails:
        print(f"  清除失败详情 (前5次):")
        for f in fails[:5]:
            pos = f["req"]["position"]
            vt = f["resp"]["virtual_time_s"]
            print(f"    VT={vt:.0f}s pos=({pos['x']:.0f},{pos['y']:.0f})")

    if oks:
        print(f"  清除成功:")
        for o in oks:
            pos = o["req"]["position"]
            vt = o["resp"]["virtual_time_s"]
            print(f"    VT={vt:.0f}s pos=({pos['x']:.0f},{pos['y']:.0f})")

    # 如果有>=2条方位线, 计算LSQ估计
    if len(dirs) >= 2:
        bearings = []
        for d in dirs:
            pos = d["req"]["position"]
            deg = d["resp"]["svd_deg"]
            bearings.append((pos, deg))

        # LSQ
        Sxx = Sxy = Syy = sx = sy = 0.0
        for pos, deg in bearings:
            a = math.radians(deg)
            n0, n1 = -math.sin(a), math.cos(a)
            p = pos
            b = n0 * p["x"] + n1 * p["y"]
            Sxx += n0 * n0; Sxy += n0 * n1; Syy += n1 * n1
            sx += n0 * b; sy += n1 * b
        det = Sxx * Syy - Sxy * Sxy
        if abs(det) > 1e-9:
            est_x = (sx * Syy - sy * Sxy) / det
            est_y = (Sxx * sy - Sxy * sx) / det
            print(f"  LSQ估计位置: ({est_x:.0f}, {est_y:.0f})")
            if oks:
                ok_pos = oks[0]["req"]["position"]
                d = math.hypot(est_x - ok_pos["x"], est_y - ok_pos["y"])
                print(f"  LSQ vs 实际清除点距离: {d:.0f}m")
