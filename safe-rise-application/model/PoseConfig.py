"""
Pose / keypoint configuration
This module centralizes COCO-17 keypoint naming so the rest of the codebase can
refer to points by name instead of hard-coded indices.

COCO-17 (MS COCO keypoints) index order used by most pose models:
0 nose
1 left_eye
2 right_eye
3 left_ear
4 right_ear
5 left_shoulder
6 right_shoulder
7 left_elbow
8 right_elbow
9 left_wrist
10 right_wrist
11 left_hip
12 right_hip
13 left_knee
14 right_knee
15 left_ankle
16 right_ankle
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple


# Ordered names by COCO index (0..16)
COCO17_KEYPOINT_NAMES: List[str] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

def map_keypoints_xy_conf(
    xy, conf=None, *, names: List[str] = COCO17_KEYPOINT_NAMES
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    k = min(len(names), len(xy))

    for i in range(k):
        name = names[i]
        x = float(xy[i][0])
        y = float(xy[i][1])
        if conf is None:
            out[name] = {"x": x, "y": y}
        else:
            out[name] = {"x": x, "y": y, "conf": float(conf[i])}

    return out


# ----------------------------
# Posture / pose classification (rule-based)
# ----------------------------

POSTURE_STAND = "stand"
POSTURE_STAND_UP = "stand_up"
POSTURE_SIT = "sit"
POSTURE_SIT_DOWN = "sit_down"
POSTURE_LEAN = "lean"
POSTURE_LIE = "lie" # Sleep

POSTURE_LABELS = [
    POSTURE_STAND,
    POSTURE_STAND_UP,
    POSTURE_SIT,
    POSTURE_SIT_DOWN,
    POSTURE_LEAN,
    POSTURE_LIE,
]


def _get_conf(keypoint: Dict[str, Dict[str, float]], name: str) -> float:
    return float(keypoint.get(name, {}).get("conf", 1.0))


def _has(keypoint: Dict[str, Dict[str, float]], name: str, min_conf: float) -> bool:
    return (name in keypoint) and (_get_conf(keypoint, name) >= min_conf)


def _point_xy(
    keypoint: Dict[str, Dict[str, float]],
    name: str,
    min_conf: float,
) -> Optional[Tuple[float, float]]:
    if not _has(keypoint, name, min_conf):
        return None
    return float(keypoint[name]["x"]), float(keypoint[name]["y"])


def _pair_center_xy(
    key_point: Dict[str, Dict[str, float]],
    left_name: str,
    right_name: str,
    min_conf: float,
) -> Tuple[Optional[Tuple[float, float]], float]:
    """Return pair center and reliability (0..2 points available)."""
    left_point = _point_xy(key_point, left_name, min_conf)
    right_point = _point_xy(key_point, right_name, min_conf)

    if left_point is not None and right_point is not None:
        left_w = max(min_conf, _get_conf(key_point, left_name))
        right_w = max(min_conf, _get_conf(key_point, right_name))
        w_sum = left_w + right_w
        cx = (left_point[0] * left_w + right_point[0] * right_w) / w_sum
        cy = (left_point[1] * left_w + right_point[1] * right_w) / w_sum
        return (cx, cy), 2.0

    if left_point is not None:
        return left_point, 1.0
    if right_point is not None:
        return right_point, 1.0

    return None, 0.0


def _group_center_xy(
    keypoint: Dict[str, Dict[str, float]],
    names: List[str],
    min_conf: float,
) -> Tuple[Optional[Tuple[float, float]], float]:
    weighted_x = 0.0
    weighted_y = 0.0
    total_w = 0.0
    count = 0

    for n in names:
        pt = _point_xy(keypoint, n, min_conf)
        if pt is None:
            continue
        w = max(min_conf, _get_conf(keypoint, n))
        weighted_x += pt[0] * w
        weighted_y += pt[1] * w
        total_w += w
        count += 1

    if count == 0 or total_w <= 1e-6:
        return None, 0.0

    return (weighted_x / total_w, weighted_y / total_w), float(count)


def _weighted_mean(values: List[float], weights: Optional[List[float]] = None) -> Optional[float]:
    if not values:
        return None
    if not weights or len(weights) != len(values):
        return float(sum(values) / len(values))

    w_sum = 0.0
    wx_sum = 0.0
    for v, w in zip(values, weights):
        ww = max(1e-6, float(w))
        w_sum += ww
        wx_sum += float(v) * ww

    if w_sum <= 1e-6:
        return float(sum(values) / len(values))
    return float(wx_sum / w_sum)


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _angle_degrees_from_vertical(dx: float, dy: float) -> float:
    """Return absolute angle (0..90) of a vector from vertical axis.

    - 0 deg  => perfectly vertical
    - 90 deg => perfectly horizontal

    Note: image coordinate has y increasing downward; we use abs values here.
    """
    # avoid division by zero
    adx = abs(float(dx))
    ady = abs(float(dy))
    if adx < 1e-6 and ady < 1e-6:
        return 0.0
    if ady < 1e-6:
        return 90.0

    # tan(theta) = |dx| / |dy| where theta from vertical
    theta = math.degrees(math.atan(adx / ady))
    if theta < 0:
        theta = -theta
    if theta > 90:
        theta = 90.0
    return float(theta)


def _knee_bend_score(hip_y: float, knee_y: float, ankle_y: float) -> float:
    """A simple 0..1-ish knee bend score using vertical distances.

    Standing legs: hip->knee and knee->ankle both large, knee around the middle.
    Sitting legs: hip close to knee (small hip->knee) and knee high relative to ankle.

    We return a score where:
      - near 0 => straight / standing-like
      - near 1 => bent / sitting-like

    This is intentionally heuristic (no trig) to be robust and lightweight.
    """
    upper = max(1e-6, float(knee_y - hip_y))      # hip -> knee (downward)
    lower = max(1e-6, float(ankle_y - knee_y))    # knee -> ankle (downward)
    total = upper + lower

    # if knee is very close to hip, likely bent
    upper_ratio = upper / total

    # upper_ratio ~0.5 in standing, smaller when bent (hip closer to knee)
    bend = 1.0 - (upper_ratio * 2.0)  # 0 when ~0.5, higher when upper_ratio small
    if bend < 0.0:
        bend = 0.0
    if bend > 1.0:
        bend = 1.0
    return float(bend)


def _knee_bend_score_from_points(
    hip: Tuple[float, float],
    knee: Tuple[float, float],
    ankle: Tuple[float, float],
) -> float:
    """Blend geometric knee angle and vertical heuristic into a 0..1 bend score."""
    hx, hy = float(hip[0]), float(hip[1])
    kx, ky = float(knee[0]), float(knee[1])
    ax, ay = float(ankle[0]), float(ankle[1])

    v1x, v1y = hx - kx, hy - ky
    v2x, v2y = ax - kx, ay - ky

    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)

    if n1 < 1e-6 or n2 < 1e-6:
        return _knee_bend_score(hip_y=hy, knee_y=ky, ankle_y=ay)

    cos_angle = (v1x * v2x + v1y * v2y) / (n1 * n2)
    if cos_angle > 1.0:
        cos_angle = 1.0
    if cos_angle < -1.0:
        cos_angle = -1.0

    knee_angle_deg = math.degrees(math.acos(cos_angle))

    # 165+ deg => almost straight (bend~0), ~70 deg => deep bend (bend~1)
    bend_from_angle = _clamp01((165.0 - knee_angle_deg) / 95.0)
    bend_from_vertical = _knee_bend_score(hip_y=hy, knee_y=ky, ankle_y=ay)

    return float((0.70 * bend_from_angle) + (0.30 * bend_from_vertical))


def classify_posture_coco17(
    keypoints_by_name: Dict[str, Dict[str, float]],
    *,
    min_keypointt_conf: float = 0.30,
) -> str:
    keypoint = keypoints_by_name

    # ---------- 1) Get important centers ----------
    shoulder_center, shoulder_pts = _pair_center_xy(keypoint, "left_shoulder", "right_shoulder", min_keypointt_conf)
    hip_center, hip_pts = _pair_center_xy(keypoint, "left_hip", "right_hip", min_keypointt_conf)
    knee_center, knee_pts = _pair_center_xy(keypoint, "left_knee", "right_knee", min_keypointt_conf)
    ankle_center, ankle_pts = _pair_center_xy(keypoint, "left_ankle", "right_ankle", min_keypointt_conf)
    head_center, head_pts = _group_center_xy(
        keypoint,
        ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
        min_keypointt_conf,
    )
    print(f"shoulder_center: {shoulder_center}, shoulder_pts: {shoulder_pts}")

    # ---------- 2) Person Body angle (0=vertical, 90=horizontal) ----------
    torso_angle_from_vertical: Optional[float] = None
    if shoulder_center is not None and hip_center is not None:
        torso_angle_from_vertical = _angle_degrees_from_vertical(
            shoulder_center[0] - hip_center[0],
            shoulder_center[1] - hip_center[1],
        )
    elif head_center is not None and hip_center is not None:
        torso_angle_from_vertical = _angle_degrees_from_vertical(
            head_center[0] - hip_center[0],
            head_center[1] - hip_center[1],
        )

    # ---------- 3) Knee bend (0=straight, 1=bent) ----------
    knee_bend_values: List[float] = []
    for side in ["left", "right"]:
        hip_pt = _point_xy(keypoint, f"{side}_hip", min_keypointt_conf)
        knee_pt = _point_xy(keypoint, f"{side}_knee", min_keypointt_conf)
        ankle_pt = _point_xy(keypoint, f"{side}_ankle", min_keypointt_conf)
        if hip_pt is None or knee_pt is None or ankle_pt is None:
            continue
        knee_bend_values.append(_knee_bend_score_from_points(hip_pt, knee_pt, ankle_pt))

    if not knee_bend_values and hip_center is not None and knee_center is not None and ankle_center is not None:
        knee_bend_values.append(_knee_bend_score_from_points(hip_center, knee_center, ankle_center))

    knee_bend = _weighted_mean(knee_bend_values)

    # ---------- 4) If lower body is missing, classify mostly from torso ----------
    upper_visible = (head_pts > 0.0) or (shoulder_pts > 0.0) or (hip_pts > 0.0)
    lower_visible = (knee_pts > 0.0) or (ankle_pts > 0.0)
    if upper_visible and not lower_visible:
        if torso_angle_from_vertical is None:
            return POSTURE_LEAN
        if torso_angle_from_vertical >= 75.0:
            return POSTURE_LIE
        if torso_angle_from_vertical >= 35.0:
            return POSTURE_LEAN
        return POSTURE_STAND

    # ---------- 5) Strong torso rules first ----------
    if torso_angle_from_vertical is not None:
        if torso_angle_from_vertical >= 75.0:
            return POSTURE_LIE
        if torso_angle_from_vertical >= 45.0:
            return POSTURE_LEAN

    # ---------- 6) Knee-based sitting / standing ----------
    if knee_bend is None:
        # Not enough lower-body info, use torso fallback
        if torso_angle_from_vertical is None:
            return POSTURE_LEAN
        return POSTURE_STAND if torso_angle_from_vertical < 35.0 else POSTURE_LEAN

    # Clear sit / stand
    if knee_bend >= 0.65:
        return POSTURE_SIT
    if knee_bend <= 0.25:
        return POSTURE_STAND

    # ---------- 7) Transitional posture heuristic ----------
    # Middle knee bend range => likely transitioning.
    # Use torso angle to split sit_down vs stand_up a bit.
    if torso_angle_from_vertical is None:
        return POSTURE_SIT_DOWN if knee_bend >= 0.45 else POSTURE_STAND_UP

    if torso_angle_from_vertical < 20.0:
        # body still upright -> more likely standing up
        return POSTURE_STAND_UP

    if torso_angle_from_vertical < 40.0:
        # moderate torso tilt + bent knee -> likely sitting down
        return POSTURE_SIT_DOWN

    return POSTURE_LEAN
