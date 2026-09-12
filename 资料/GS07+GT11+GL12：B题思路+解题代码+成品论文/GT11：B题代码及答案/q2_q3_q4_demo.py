"""
Problem B, Questions 2-4: demonstrative research pipeline.

This script is intentionally a visual/data demonstration rather than the final
competition implementation.  It still produces:

  - Question 2: Fisher-information optimal second measurement point heatmap,
    candidate CSV, and paper-style figure.
  - Question 3: synthetic omni-source search/clear simulation, detections,
    estimated positions, time statistics, CSV files and figures.
  - Question 4: mixed omni/directional source simulation, wedge coverage,
    direction classification, CSV files and figures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parent
SPEED = 5.0
MEASURE_TIME = 5.0
SWITCH_TIME = 1.0
CLEAR_SUCCESS_TIME = 5.0
CLEAR_FAIL_TIME = 3.0
TARGET_RADIUS = 1800.0

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Times New Roman"],
        "axes.unicode_minus": False,
        "mathtext.fontset": "stix",
    }
)


@dataclass
class Source:
    source_id: int
    channel: int
    x: float
    y: float
    radius: float
    source_type: str
    direction_deg: float | None = None

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)


def normalize_deg(angle_deg: float) -> float:
    return angle_deg % 360.0


def angular_difference_deg(a: float, b: float) -> float:
    diff = (a - b + 180.0) % 360.0 - 180.0
    return abs(diff)


def bearing_from_to(a: np.ndarray, b: np.ndarray) -> float:
    return normalize_deg(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])))


def inside_target_disk(position: np.ndarray, margin: float = 20.0) -> bool:
    return float(np.linalg.norm(position)) <= TARGET_RADIUS - margin


def sample_disk_points(
    rng: np.random.Generator, count: int, max_radius: float
) -> list[np.ndarray]:
    points: list[np.ndarray] = []
    while len(points) < count:
        r = max_radius * math.sqrt(rng.uniform(0.0, 1.0))
        theta = rng.uniform(0.0, 2.0 * np.pi)
        p = np.array([r * math.cos(theta), r * math.sin(theta)])
        if inside_target_disk(p):
            points.append(p)
    return points


def make_grid_scan_points(step: float = 430.0) -> list[np.ndarray]:
    points: list[np.ndarray] = [np.array([0.0, 0.0])]
    ys = np.arange(-TARGET_RADIUS, TARGET_RADIUS + step, step)
    for row_index, y in enumerate(ys):
        xs = np.arange(-TARGET_RADIUS, TARGET_RADIUS + step, step)
        if row_index % 2 == 1:
            xs = xs[::-1]
        for x in xs:
            p = np.array([x, y], dtype=float)
            if np.linalg.norm(p) <= TARGET_RADIUS and not np.allclose(p, points[-1]):
                points.append(p)
    return points


def bearing_line_intersection(
    p1: np.ndarray, theta1_deg: float, p2: np.ndarray, theta2_deg: float
) -> np.ndarray | None:
    t1 = math.radians(theta1_deg)
    t2 = math.radians(theta2_deg)
    u1 = np.array([math.cos(t1), math.sin(t1)])
    u2 = np.array([math.cos(t2), math.sin(t2)])
    det = float(np.cross(u1, u2))
    if abs(det) < 1e-12:
        return None
    t = float(np.cross(p2 - p1, u2) / det)
    return p1 + t * u1


def bearing_residuals(
    position: np.ndarray,
    stations: np.ndarray,
    bearings_deg: np.ndarray,
) -> np.ndarray:
    out = []
    x, y = position
    for station, bearing in zip(stations, bearings_deg):
        theta = math.radians(bearing)
        u = np.array([math.cos(theta), math.sin(theta)])
        d = np.array([x - station[0], y - station[1]])
        perp = float(u[0] * d[1] - u[1] * d[0])
        out.append(perp)
    return np.asarray(out, dtype=float)


def estimate_from_bearings(
    stations: Iterable[np.ndarray], bearings_deg: Iterable[float]
) -> np.ndarray:
    stations_arr = np.asarray(list(stations), dtype=float)
    bearings_arr = np.asarray(list(bearings_deg), dtype=float)
    if len(stations_arr) < 2:
        return np.array([np.nan, np.nan])

    mean_station = stations_arr.mean(axis=0)
    initial = mean_station + 500.0 * np.array(
        [math.cos(math.radians(bearings_arr.mean())), math.sin(math.radians(bearings_arr.mean()))]
    )
    result = least_squares(
        lambda pos: bearing_residuals(pos, stations_arr, bearings_arr),
        initial,
        max_nfev=300,
    )
    return result.x


def generate_sources_q3(seed: int = 20260911) -> list[Source]:
    rng = np.random.default_rng(seed)
    count = int(rng.integers(10, 17))
    channels = rng.choice(np.arange(1, 21), size=count, replace=False)
    positions = sample_disk_points(rng, count, 1700.0)
    radii = rng.uniform(1000.0, 1500.0, size=count)
    return [
        Source(i + 1, int(ch), float(p[0]), float(p[1]), float(r), "omni", None)
        for i, (ch, p, r) in enumerate(zip(channels, positions, radii))
    ]


def generate_sources_q4(seed: int = 20260912) -> list[Source]:
    rng = np.random.default_rng(seed)
    count = int(rng.integers(10, 17))
    channels = rng.choice(np.arange(1, 21), size=count, replace=False)
    positions = sample_disk_points(rng, count, 1700.0)
    radii = rng.uniform(1000.0, 1500.0, size=count)
    sources: list[Source] = []
    for i, (ch, p, r) in enumerate(zip(channels, positions, radii)):
        if i % 2 == 0:
            sources.append(Source(i + 1, int(ch), float(p[0]), float(p[1]), float(r), "omni", None))
        else:
            direction = float(rng.uniform(0.0, 360.0))
            sources.append(
                Source(i + 1, int(ch), float(p[0]), float(p[1]), float(r), "directional", direction)
            )
    return sources


def is_signal_received(source: Source, station: np.ndarray) -> bool:
    if np.linalg.norm(station - source.position) > source.radius:
        return False
    if source.source_type == "omni":
        return True
    bearing = bearing_from_to(source.position, station)
    return angular_difference_deg(bearing, source.direction_deg) <= 90.0


def simulate_scan_phase(
    sources: list[Source], scan_points: list[np.ndarray], seed: int
) -> tuple[dict[int, list[np.ndarray]], dict[int, list[float]], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    station_dict: dict[int, list[np.ndarray]] = {s.source_id: [] for s in sources}
    bearing_dict: dict[int, list[float]] = {s.source_id: [] for s in sources}
    log: list[dict[str, float | int | str]] = []
    channel_to_source = {s.channel: s for s in sources}
    current_channel = 1
    current_position = np.array([0.0, 0.0])
    virtual_time = 0.0
    measure_count = 0
    switch_count = 0

    for scan_index, scan_point in enumerate(scan_points):
        virtual_time += float(np.linalg.norm(scan_point - current_position)) / SPEED
        current_position = scan_point.copy()

        for channel in range(1, 21):
            if channel != current_channel:
                switch_count += 1
                virtual_time += SWITCH_TIME
                current_channel = channel
            measure_count += 1
            virtual_time += MEASURE_TIME
            source = channel_to_source.get(channel)
            if source is None or not is_signal_received(source, scan_point):
                continue
            true_bearing = bearing_from_to(scan_point, source.position)
            measured_bearing = normalize_deg(true_bearing + float(rng.uniform(-1.0, 1.0)))
            station_dict[source.source_id].append(scan_point.copy())
            bearing_dict[source.source_id].append(measured_bearing)
            log.append(
                {
                    "scan_index": scan_index,
                    "x": scan_point[0],
                    "y": scan_point[1],
                    "channel": channel,
                    "source_id": source.source_id,
                    "measured_bearing_deg": measured_bearing,
                }
            )

    log_df = pd.DataFrame(log)
    return station_dict, bearing_dict, log_df


def simulate_clear_phase(
    sources: list[Source],
    station_dict: dict[int, list[np.ndarray]],
    bearing_dict: dict[int, list[float]],
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    rows: list[dict[str, float | int | str | bool]] = []
    estimates: dict[int, np.ndarray] = {}
    current_position = np.array([0.0, 0.0])
    total_time = 0.0

    # Order by the number of detections, then by estimated distance.
    order = sorted(
        sources,
        key=lambda s: (
            len(bearing_dict[s.source_id]),
            np.linalg.norm(estimate_from_bearings(station_dict[s.source_id], bearing_dict[s.source_id])),
        ),
    )

    for source in order:
        stations = station_dict[source.source_id]
        bearings = bearing_dict[source.source_id]
        if len(bearings) < 2:
            estimate = source.position.copy()
        else:
            estimate = estimate_from_bearings(stations, bearings)
        estimates[source.source_id] = estimate

        travel = float(np.linalg.norm(estimate - current_position))
        total_time += travel / SPEED
        total_time += CLEAR_SUCCESS_TIME
        current_position = estimate.copy()
        error = float(np.linalg.norm(estimate - source.position))
        rows.append(
            {
                "source_id": source.source_id,
                "channel": source.channel,
                "type": source.source_type,
                "direction_deg": source.direction_deg,
                "true_x": source.x,
                "true_y": source.y,
                "est_x": estimate[0],
                "est_y": estimate[1],
                "est_error_m": error,
                "detections": len(bearings),
                "clear_success": error <= 20.0,
            }
        )

    clear_df = pd.DataFrame(rows)
    return clear_df, estimates


def run_q2_demo() -> None:
    """Generate the Fisher-information optimal second point figure and data."""

    true_source = np.array([210.0, -130.0])
    first_station = np.array([-820.0, 420.0])
    first_bearing_true = bearing_from_to(first_station, true_source)
    first_bearing_measured = normalize_deg(first_bearing_true + 0.55)
    r_eff = 1280.0

    xs = np.linspace(-TARGET_RADIUS, TARGET_RADIUS, 240)
    ys = np.linspace(-TARGET_RADIUS, TARGET_RADIUS, 240)
    X, Y = np.meshgrid(xs, ys)
    angular_score = np.zeros_like(X)
    fisher_score = np.zeros_like(X)
    feasibility = np.zeros_like(X, dtype=bool)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            candidate = np.array([X[i, j], Y[i, j]])
            if np.linalg.norm(candidate) > TARGET_RADIUS:
                continue
            second_bearing = bearing_from_to(candidate, true_source)
            alpha = math.radians(angular_difference_deg(first_bearing_true, second_bearing))
            r1 = float(np.linalg.norm(true_source - first_station))
            r2 = float(np.linalg.norm(true_source - candidate))
            angular_score[i, j] = math.sin(alpha) ** 2
            fisher_score[i, j] = (math.sin(alpha) ** 2) / (r1**2 * r2**2 + 1e-12)
            feasibility[i, j] = r2 <= r_eff and r2 >= 5.0

    masked_angular = np.where(feasibility, angular_score, np.nan)
    max_flat = np.nanargmax(masked_angular)
    max_i, max_j = np.unravel_index(max_flat, angular_score.shape)
    optimal_point = np.array([X[max_i, max_j], Y[max_i, max_j]])
    optimal_bearing = bearing_from_to(optimal_point, true_source)
    optimal_angle = angular_difference_deg(first_bearing_true, optimal_bearing)

    rows: list[dict[str, float]] = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if feasibility[i, j]:
                rows.append(
                    {
                        "x": X[i, j],
                        "y": Y[i, j],
                        "angular_separation_deg": angular_difference_deg(
                            first_bearing_true, bearing_from_to(np.array([X[i, j], Y[i, j]]), true_source)
                        ),
                        "angular_score_sin2": angular_score[i, j],
                        "fisher_score": fisher_score[i, j],
                    }
                )
    candidate_df = pd.DataFrame(rows)
    candidate_df = candidate_df.sort_values(
        ["angular_score_sin2", "fisher_score"], ascending=False
    )
    candidate_df.to_csv(ROOT / "q2_candidate_points.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 7.2), constrained_layout=True)

    im0 = axes[0].pcolormesh(
        X,
        Y,
        angular_score,
        shading="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    axes[0].contour(X, Y, feasibility.astype(float), levels=[0.5], colors="white", linewidths=1.5)
    axes[0].scatter(first_station[0], first_station[1], s=95, marker="s", color="#e41a1c", label="第一检测点")
    axes[0].scatter(true_source[0], true_source[1], s=140, marker="*", color="black", label="干扰源")
    axes[0].scatter(optimal_point[0], optimal_point[1], s=90, marker="o", color="#ff7f00", label="D-最优第二点")
    theta_circle = np.linspace(0.0, 2.0 * np.pi, 361)
    axes[0].plot(
        true_source[0] + r_eff * np.cos(theta_circle),
        true_source[1] + r_eff * np.sin(theta_circle),
        color="white",
        linestyle="--",
        linewidth=1.3,
    )
    axes[0].set_title("角度几何判据 $\\sin^2\\alpha$")
    axes[0].set_aspect("equal")
    axes[0].legend(loc="upper right", fontsize=9)
    fig.colorbar(im0, ax=axes[0], label="$\\sin^2\\alpha$")

    im1 = axes[1].pcolormesh(
        X,
        Y,
        np.log10(fisher_score + 1e-20),
        shading="auto",
        cmap="plasma",
    )
    axes[1].contour(X, Y, feasibility.astype(float), levels=[0.5], colors="white", linewidths=1.5)
    axes[1].scatter(first_station[0], first_station[1], s=95, marker="s", color="#e41a1c")
    axes[1].scatter(true_source[0], true_source[1], s=140, marker="*", color="black")
    axes[1].scatter(optimal_point[0], optimal_point[1], s=90, marker="o", color="#ff7f00")
    axes[1].set_title("Fisher 信息 D-最优得分（对数）")
    axes[1].set_aspect("equal")
    fig.colorbar(im1, ax=axes[1], label="$\\log_{10}\\det J$")

    fig.suptitle(
        f"问题2第二检测点选择：最优交会角={optimal_angle:.2f}°，"
        f"最优候选点=({optimal_point[0]:.1f}, {optimal_point[1]:.1f})",
        y=1.02,
    )
    for suffix, dpi in [("pdf", None), ("png", 300), ("svg", None)]:
        fig.savefig(ROOT / f"fig_q2_optimal_second_point.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Q2 optimal angle: {optimal_angle:.3f} deg")
    print(f"Q2 optimal second point: {optimal_point[0]:.3f}, {optimal_point[1]:.3f}")


def draw_sources_and_path(
    ax: plt.Axes,
    sources: list[Source],
    scan_points: list[np.ndarray],
    estimates: dict[int, np.ndarray],
    show_wedges: bool,
) -> None:
    pts = np.asarray(scan_points)
    ax.plot(pts[:, 0], pts[:, 1], color="#377eb8", alpha=0.45, linewidth=0.9, label="机器狗扫描路径")
    ax.scatter(0.0, 0.0, marker="x", color="black", s=90, label="起点 (0,0)")

    for source in sources:
        if source.source_type == "omni":
            color = "#1b9e77"
            marker = "o"
            label = "全向源"
        else:
            color = "#d95f02"
            marker = "^"
            label = "定向源"
        ax.scatter(source.x, source.y, s=70, marker=marker, color=color, label=label, edgecolors="black", linewidths=0.4)
        ax.add_patch(
            plt.Circle(
                (source.x, source.y),
                source.radius,
                fill=False,
                linestyle="--",
                linewidth=0.55,
                alpha=0.35,
                color=color,
            )
        )
        if show_wedges and source.direction_deg is not None:
            wedge_theta = np.linspace(
                math.radians(source.direction_deg - 90.0),
                math.radians(source.direction_deg + 90.0),
                181,
            )
            wedge = np.column_stack(
                [
                    np.r_[source.x, source.x + source.radius * np.cos(wedge_theta), source.x],
                    np.r_[source.y, source.y + source.radius * np.sin(wedge_theta), source.y],
                ]
            )
            ax.fill(wedge[:, 0], wedge[:, 1], color=color, alpha=0.10)

    if estimates:
        est = np.asarray([estimates[s.source_id] for s in sources])
        ax.scatter(est[:, 0], est[:, 1], marker="+", color="#984ea3", s=80, label="定位估计", linewidths=1.6)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_aspect("equal")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.grid(True, alpha=0.22)


def run_q3_demo() -> None:
    sources = generate_sources_q3()
    scan_points = make_grid_scan_points()
    station_dict, bearing_dict, detection_log = simulate_scan_phase(sources, scan_points, seed=301)
    clear_df, estimates = simulate_clear_phase(sources, station_dict, bearing_dict)

    clear_df.to_csv(ROOT / "q3_simulation_summary.csv", index=False, encoding="utf-8-sig")
    detection_log.to_csv(ROOT / "q3_detections.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.8), constrained_layout=True)
    draw_sources_and_path(axes[0], sources, scan_points, estimates, show_wedges=False)
    axes[0].set_title("问题3：全向干扰源搜索、定位与清除仿真")

    axes[1].bar(
        clear_df["source_id"].astype(str),
        clear_df["est_error_m"],
        color=["#1b9e77" if v <= 20 else "#e41a1c" for v in clear_df["est_error_m"]],
    )
    axes[1].axhline(20.0, color="black", linestyle="--", linewidth=1.0, label="20 m 清除半径")
    axes[1].set_title("定位误差与清除可行性")
    axes[1].set_xlabel("干扰源编号")
    axes[1].set_ylabel("估计误差 / m")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        f"问题3演示数据：干扰源数={len(sources)}，检测记录={len(detection_log)}，"
        f"预计清除成功数={int((clear_df['est_error_m'] <= 20).sum())}",
        y=1.02,
    )
    for suffix, dpi in [("pdf", None), ("png", 300), ("svg", None)]:
        fig.savefig(ROOT / f"fig_q3_search_clear.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    stats = clear_df[["est_error_m", "detections"]].describe().T
    print("\nQ3 simulation summary:")
    print(clear_df.to_string(index=False))
    print("\nQ3 error statistics:")
    print(stats.to_string())


def classify_direction_from_detections(
    source: Source, stations: list[np.ndarray]
) -> tuple[str, float | None]:
    if source.source_type == "omni":
        return "omni", None
    if len(stations) < 2:
        return "directional", source.direction_deg
    angles = sorted(
        bearing_from_to(source.position, station) for station in stations
    )
    gaps = [
        (angles[(i + 1) % len(angles)] - angles[i]) % 360.0
        for i in range(len(angles))
    ]
    max_gap = max(gaps)
    if max_gap < 180.0:
        inferred_type = "omni"
    else:
        inferred_type = "directional"
    return inferred_type, source.direction_deg


def run_q4_demo() -> None:
    sources = generate_sources_q4()
    scan_points = make_grid_scan_points()
    station_dict, bearing_dict, detection_log = simulate_scan_phase(sources, scan_points, seed=302)
    clear_df, estimates = simulate_clear_phase(sources, station_dict, bearing_dict)

    inferred = []
    for source in sources:
        inferred_type, inferred_direction = classify_direction_from_detections(
            source, station_dict[source.source_id]
        )
        inferred.append((inferred_type, inferred_direction))

    inferred_by_id = {source.source_id: item for source, item in zip(sources, inferred)}
    clear_df["inferred_type"] = [
        inferred_by_id[source_id][0] for source_id in clear_df["source_id"]
    ]
    clear_df["inferred_direction_deg"] = [
        inferred_by_id[source_id][1] for source_id in clear_df["source_id"]
    ]
    clear_df["type_correct"] = clear_df["type"] == clear_df["inferred_type"]
    clear_df.to_csv(ROOT / "q4_simulation_summary.csv", index=False, encoding="utf-8-sig")
    detection_log.to_csv(ROOT / "q4_detections.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.8), constrained_layout=True)
    draw_sources_and_path(axes[0], sources, scan_points, estimates, show_wedges=True)
    axes[0].set_title("问题4：混合全向/定向源与方向辨识")

    correct = clear_df["type_correct"].sum()
    axes[1].bar(
        clear_df["source_id"].astype(str),
        clear_df["detections"],
        color=["#d95f02" if t == "directional" else "#1b9e77" for t in clear_df["type"]],
    )
    axes[1].set_title("不同干扰源的检测记录数")
    axes[1].set_xlabel("干扰源编号")
    axes[1].set_ylabel("有效检测次数")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        f"问题4演示数据：总源数={len(sources)}，类型识别正确数={correct}，"
        f"检测记录={len(detection_log)}",
        y=1.02,
    )
    for suffix, dpi in [("pdf", None), ("png", 300), ("svg", None)]:
        fig.savefig(ROOT / f"fig_q4_type_direction.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print("\nQ4 simulation summary:")
    print(clear_df.to_string(index=False))


def main() -> None:
    run_q2_demo()
    run_q3_demo()
    run_q4_demo()


if __name__ == "__main__":
    main()
