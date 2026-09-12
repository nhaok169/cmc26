# -*- coding: utf-8 -*-
"""模拟器 HTTP 封装：串行、新 request_id、检查 accepted；本机请求关掉代理。"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

for _k in list(os.environ):
    if "proxy" in _k.lower():
        os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

BASE_URL = "http://127.0.0.1:2026"
ARENA_ID = "default"


class ApiError(RuntimeError):
    pass


class Simulator:
    def __init__(self, robot_id: str, base_url: str = BASE_URL, timeout: float = 30.0):
        self.robot_id = str(robot_id)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.seq = 0
        self.virtual_time_s = 0.0
        self.log_rows = []

    def _new_id(self, prefix: str) -> str:
        self.seq += 1
        return f"{prefix}-{os.getpid()}-{self.seq:04d}"

    def _post(self, path: str, payload: Dict[str, Any], retries: int = 4) -> Dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        last_err: Optional[Exception] = None
        for attempt in range(retries):
            req = urllib.request.Request(
                self.base_url + path,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8")
                    body = json.loads(raw)
                    self.log_rows.append({"path": path, "payload": payload, "response": body})
                    if body.get("accepted") is True and "virtual_time_s" in body:
                        self.virtual_time_s = float(body["virtual_time_s"])
                    return body
            except urllib.error.HTTPError as e:
                raw = e.read().decode("utf-8", "replace")
                try:
                    body = json.loads(raw)
                except Exception:
                    raise ApiError(f"HTTP {e.code} {path}: {raw[:400]}") from e
                self.log_rows.append({"path": path, "payload": payload, "response": body, "http": e.code})
                if e.code == 409:
                    raise ApiError(f"HTTP 409 幂等冲突 {path}: {body}") from e
                if e.code >= 500:
                    last_err = e
                    time.sleep(0.4 * (attempt + 1))
                    continue
                return body
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last_err = e
                time.sleep(0.5 * (attempt + 1))
        raise ApiError(f"{path} 连接失败: {last_err}")

    def enter(self) -> Dict[str, Any]:
        last = None
        for k in range(6):
            try:
                body = self._post(
                    "/enter",
                    {"arena_id": ARENA_ID, "robot_id": self.robot_id, "request_id": self._new_id("enter")},
                )
                if body.get("accepted") is True:
                    return body
                last = ApiError(f"/enter 被拒绝: {body}")
            except ApiError as e:
                last = e
            time.sleep(0.8 * (k + 1))
        raise last if last else ApiError("/enter 失败")

    def measure(self, x: float, y: float, channel: int) -> Dict[str, Any]:
        body = self._post(
            "/measure",
            {
                "arena_id": ARENA_ID,
                "robot_id": self.robot_id,
                "request_id": self._new_id("meas"),
                "position": {"x": float(x), "y": float(y)},
                "channel": int(channel),
            },
        )
        if body.get("accepted") is not True:
            raise ApiError(f"/measure 被拒绝: {body}")
        return body

    def clear(self, x: float, y: float, channel: int) -> Dict[str, Any]:
        body = self._post(
            "/clear",
            {
                "arena_id": ARENA_ID,
                "robot_id": self.robot_id,
                "request_id": self._new_id("clr"),
                "position": {"x": float(x), "y": float(y)},
                "channel": int(channel),
            },
        )
        if body.get("accepted") is not True:
            raise ApiError(f"/clear 被拒绝: {body}")
        return body

    def exit(self) -> Dict[str, Any]:
        body = self._post(
            "/exit",
            {"arena_id": ARENA_ID, "robot_id": self.robot_id, "request_id": self._new_id("exit")},
        )
        return body
