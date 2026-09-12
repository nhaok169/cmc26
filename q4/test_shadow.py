# -*- coding: utf-8 -*-
"""复现官方局里那种：沿示向度走进定向阴影后应就地搜线段，而不是两点打转。"""
from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from local_world import LocalSimulator
from run_q4 import Dog


class ScriptedSim(LocalSimulator):
    def __init__(self):
        rng = np.random.default_rng(0)
        super().__init__(rng, n_src=1, p_dir=0.0)
        G = np.array([960.0, 400.0])
        u = np.array([1900.0, 0.0]) - G
        u = u / float(np.linalg.norm(u))
        ch = next(iter(self.sources))
        self.sources[ch] = {
            "G": G,
            "R": 1400.0,
            "alive": True,
            "err": {},
            "directional": True,
            "u": u,
        }
        self.n_true = 1
        self.n_dir = 1
        self.ch = ch


def main():
    sim = ScriptedSim()
    dog = Dog(sim, mode="cert26")
    sim.enter()
    # 外环点亮，再沿示向度走进阴影
    body = dog.measure_at(np.array([1900.0, 0.0]), sim.ch)
    assert body.get("measure_result") == "direction", body
    buf = StringIO()
    with redirect_stdout(buf):
        ok = dog.hunt(sim.ch, last_ditch=False)
    text = buf.getvalue()
    print(text[-1500:])
    print(
        "ok", ok, "cleared", sim.ch in dog.book.cleared,
        "T", round(sim.virtual_time_s),
        "n_clear_calls",
        sum(1 for r in sim.log_rows if r["path"] == "/clear"),
        "fails",
        sum(
            1
            for r in sim.log_rows
            if r["path"] == "/clear" and r["response"].get("clear_result") != "success"
        ),
        "shadow" , "shadow-search" in text,
        "alive", sim.sources[sim.ch]["alive"],
    )
    assert ok and not sim.sources[sim.ch]["alive"]
    assert sim.virtual_time_s < 2500, sim.virtual_time_s
    print("shadow-hunt regression OK")


if __name__ == "__main__":
    main()
