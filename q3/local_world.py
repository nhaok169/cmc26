# -*- coding: utf-8 -*-
"""问题3 本地对照环境：规则与附件一致，用于多局策略检验（非正式测试）。"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np


class LocalSimulator:
    def __init__(self, rng: np.random.Generator, n_src: Optional[int] = None):
        self.rng = rng
        n = int(n_src if n_src is not None else rng.integers(10, 17))
        channels = rng.choice(np.arange(1, 21), size=n, replace=False)
        self.sources: Dict[int, dict] = {}
        for ch in channels:
            r = 1800.0 * math.sqrt(float(rng.random()))
            a = float(rng.random() * 2.0 * math.pi)
            G = np.array([r * math.cos(a), r * math.sin(a)], dtype=float)
            R = float(rng.uniform(1000.0, 1500.0))
            self.sources[int(ch)] = {"G": G, "R": R, "alive": True, "err": {}}
        self.n_true = n
        self.xy = np.zeros(2)
        self.channel = 1
        self.virtual_time_s = 0.0
        self.log_rows = []

    def _move_dt(self, x: float, y: float) -> float:
        return float(math.hypot(x - self.xy[0], y - self.xy[1]) / 5.0)

    def _bearing(self, x: float, y: float, ch: int) -> Tuple[str, Optional[float]]:
        src = self.sources.get(ch)
        if src is None or not src["alive"]:
            return "no_signal", None
        G = src["G"]
        d = float(math.hypot(G[0] - x, G[1] - y))
        if d > src["R"] + 1e-9:
            return "no_signal", None
        if d <= 5.0 + 1e-9:
            return "near", None
        key = (round(x, 2), round(y, 2), ch)
        if key not in src["err"]:
            src["err"][key] = float(self.rng.uniform(-1.0, 1.0))
        true = math.degrees(math.atan2(G[1] - y, G[0] - x)) % 360.0
        svd = (true + src["err"][key]) % 360.0
        return "direction", round(svd, 2)

    def enter(self) -> dict:
        self.xy = np.zeros(2)
        self.channel = 1
        self.virtual_time_s = 0.0
        body = {
            "accepted": True,
            "virtual_time_s": 0.0,
            "remaining_real_duration_s": 1200.0,
            "max_real_duration_s": 1200.0,
            "max_virtual_duration_s": 360000.0,
        }
        self.log_rows.append({"path": "/enter", "payload": {}, "response": body})
        return body

    def measure(self, x: float, y: float, channel: int) -> dict:
        dt = self._move_dt(x, y)
        sw = 1.0 if int(channel) != self.channel else 0.0
        self.virtual_time_s += dt + sw + 5.0
        self.xy = np.array([x, y], dtype=float)
        self.channel = int(channel)
        kind, svd = self._bearing(x, y, int(channel))
        body = {"accepted": True, "virtual_time_s": self.virtual_time_s, "measure_result": kind}
        if svd is not None:
            body["svd_deg"] = svd
        self.log_rows.append(
            {
                "path": "/measure",
                "payload": {"position": {"x": x, "y": y}, "channel": int(channel)},
                "response": body,
            }
        )
        return body

    def clear(self, x: float, y: float, channel: int) -> dict:
        dt = self._move_dt(x, y)
        src = self.sources.get(int(channel))
        hit = (
            src is not None
            and src["alive"]
            and float(math.hypot(src["G"][0] - x, src["G"][1] - y)) <= 20.0 + 1e-9
        )
        if hit:
            src["alive"] = False
            self.virtual_time_s += dt + 5.0
            result = "success"
        else:
            self.virtual_time_s += dt + 3.0
            result = "no_target_in_range"
        self.xy = np.array([x, y], dtype=float)
        body = {"accepted": True, "virtual_time_s": self.virtual_time_s, "clear_result": result}
        self.log_rows.append(
            {
                "path": "/clear",
                "payload": {"position": {"x": x, "y": y}, "channel": int(channel)},
                "response": body,
            }
        )
        return body

    def exit(self) -> dict:
        body = {"accepted": True, "virtual_time_s": self.virtual_time_s, "exit_reason": "user_exit"}
        self.log_rows.append({"path": "/exit", "payload": {}, "response": body})
        return body
