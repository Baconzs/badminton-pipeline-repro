"""Conservative, evidence-led stroke coaching for one badminton player.

The tracking pipeline supplies confirmed hit events, a calibrated court and
two-dimensional pose.  That is enough to describe *candidate* shot families
and visible posture signals, but it cannot see a racket face, grip, spin,
three-dimensional shuttle height, or tactical intent.  This module keeps that
boundary explicit: every named technique is a candidate and every rule is
driven by structured evidence supplied by the caller.

The module is deliberately pure.  Video/YOLO/CSV handling lives in
``web_app.coaching`` so these rules can be regression-tested without models.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40
NET_Y_M = COURT_LENGTH_M / 2.0
FRONT_COURT_M = 2.20
MID_COURT_M = 4.40
ZONE_BUFFER_M = 0.25

_VALID_HANDS = {"left", "right", "auto"}
_VALID_ZONES = {"front", "mid", "back"}
_FAMILY_LABELS = {
    "serve": "发球",
    "push": "推球",
    "lift": "挑球",
    "cross_net": "勾对角",
    "net": "搓放网",
    "net_kill": "扑球",
    "clear": "高远球",
    "drive_clear": "平高球",
    "drop": "吊球",
    "smash": "杀球",
    "defensive_drive": "后场平抽",
    "front_unknown": "前场出球",
    "back_unknown": "后场出球",
    "unknown": "球路信息不足",
}
_OVERHEAD_FAMILIES = {"clear", "drive_clear", "drop", "smash"}
_FRONT_FAMILIES = {"push", "lift", "cross_net", "net", "net_kill"}

# Only map BST classes whose coaching semantics are clear enough to select a
# family-specific prompt.  ``擋小球``/``平球``/``點扣`` and generic ``Drive``
# labels intentionally stay unmapped: forcing them into an existing family
# would make a confident model label trigger the wrong technical advice.
_BST_FAMILY_ALIASES = {
    "放小球": "net",
    "搓放网": "net",
    "net-shot": "net",
    "net shot": "net",
    "网前球": "net",
    "殺球": "smash",
    "杀球": "smash",
    "smash": "smash",
    "挑球": "lift",
    "防守回挑": "lift",
    "net-lift": "lift",
    "net lift": "lift",
    "長球": "clear",
    "高远球": "clear",
    "clear": "clear",
    "切球": "drop",
    "過渡切球": "drop",
    "吊球": "drop",
    "dropshot": "drop",
    "drop-shot": "drop",
    "drop shot": "drop",
    "推球": "push",
    "push": "push",
    "撲球": "net_kill",
    "扑球": "net_kill",
    "net-kill": "net_kill",
    "net kill": "net_kill",
    "勾球": "cross_net",
    "勾对角": "cross_net",
    "cross-net": "cross_net",
    "cross net": "cross_net",
    "發短球": "serve",
    "發長球": "serve",
    "短发球": "serve",
    "长发球": "serve",
    "发球": "serve",
    "serve": "serve",
    "後場抽平球": "defensive_drive",
    "后场平抽": "defensive_drive",
    "防守回抽": "defensive_drive",
}
_BST_UNKNOWN_LABELS = {"未知球种", "未知球種", "unknown", "待复核", ""}
_BST_MIN_CONFIDENCE = 60.0
_BST_MIN_COVERAGE = 0.08
# The looser thresholds above keep the research type report backwards
# compatible.  Motion Coach has a different cost profile: a wrong technical
# prompt is worse than omitting one, so a BST type may influence coaching only
# after this stricter, independently auditable gate passes.
_BST_COACH_MIN_CONFIDENCE = 85.0
_BST_COACH_MIN_MARGIN = 25.0
_BST_COACH_MIN_POSE_COVERAGE = 0.85
_BST_COACH_MIN_BALL_COVERAGE = 0.55
_STROKE_UNKNOWN_FAMILIES = {"unknown", "front_unknown", "back_unknown"}

_MIN_RELIABLE_STROKE_CONFIDENCE = 72
_MIN_RELIABLE_CONTACT_SAMPLES = 3
_MIN_RELIABLE_POSE_CONFIDENCE = 0.80
_MIN_RELIABLE_REAL_POINTS = 6
_MIN_REPEATED_HITS = 3

_EXPLICITLY_BLOCKED_CLASSIFICATION_STATUSES = {
    "conflict",
    "side_mismatch",
    "unsupported",
}

# These are product copy, not free-form model output.  ``problem`` describes
# the repeated visible signal and ``action`` gives exactly one drill cue.
_RECOMMENDATION_TEMPLATES = {
    "preparation_support": {
        "title": "准备支撑",
        "problem": "多次击球前的支撑姿态偏直或偏窄。",
        "action": "来球过网前完成轻微屈膝和分腿准备，再启动最后一步。",
        "metric": "准备阶段支撑二维信号",
    },
    "serve_preparation": {
        "title": "发球支撑",
        "problem": "发球前支撑连续偏直或偏窄。",
        "action": "双脚先落稳，轻微屈膝后出手。",
        "metric": "发球支撑二维信号",
    },
    "serve_first_step": {
        "title": "发球后第一步",
        "problem": "出手后连续未见明显启动。",
        "action": "把出手后的准备步连成完整发球动作。",
        "metric": "发球后移动二维信号",
    },
    "front_support": {
        "title": "前场支撑",
        "problem": "最后一步支撑连续偏直或偏窄。",
        "action": "最后一步落稳并轻微屈膝，再完成出拍。",
        "metric": "前场支撑二维信号",
    },
    "front_contact_space": {
        "title": "前场触球空间",
        "problem": "触球点连续靠近躯干。",
        "action": "提前启动，在身体前方留出触球空间。",
        "metric": "前场触球空间二维信号",
    },
    "overhead_extension": {
        "title": "上手伸展",
        "problem": "持拍臂连续未充分伸展。",
        "action": "提前到位，在身体前上方拉开触球空间。",
        "metric": "上手伸展二维信号",
    },
    "overhead_preparation": {
        "title": "上手准备",
        "problem": "多次上手击球直到触球前才抬起持拍手。",
        "action": "提前完成架拍，让触球前最后一段只负责加速。",
        "metric": "上手准备时序二维信号",
    },
    "back_support": {
        "title": "后场支撑",
        "problem": "后场击球时支撑腿连续偏直。",
        "action": "更早到位并可控蹬地，落地后立即衔接第一步。",
        "metric": "后场支撑二维信号",
    },
    "recovery_stability": {
        "title": "击球恢复",
        "problem": "多次击球后持拍臂和下肢恢复稳定偏慢。",
        "action": "随挥结束立即收拍回到身前，并衔接回位第一步。",
        "metric": "击球后恢复时序二维信号",
    },
}

# Generic visible mechanics should aggregate across compatible stroke types.
# Exact type remains important for type-specific prompts, but a clear/drive/
# smash split must not prevent three independent overhead-extension samples
# from supporting the same concise training cue.
_SIGNAL_COACH_GROUPS = {
    "preparation_support": ("preparation", "准备衔接"),
    "serve_preparation": ("serve", "发球"),
    "serve_first_step": ("serve", "发球"),
    "front_support": ("frontcourt", "前场球"),
    "front_contact_space": ("frontcourt", "前场球"),
    "overhead_extension": ("overhead", "上手球"),
    "overhead_preparation": ("overhead", "上手球"),
    "back_support": ("backcourt", "后场球"),
    "recovery_stability": ("recovery", "击球恢复"),
}
_TYPE_AGNOSTIC_SIGNALS = {
    "preparation_support",
    "overhead_preparation",
    "recovery_stability",
}


def _number(value: object, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _point(value: object, minimum_confidence: float = 0.40) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x, y = _number(value[0]), _number(value[1])
    if x is None or y is None:
        return None
    if len(value) >= 3:
        confidence = _number(value[2])
        if confidence is None or confidence < minimum_confidence:
            return None
    return x, y


def _distance(first: tuple[float, float] | None, second: tuple[float, float] | None) -> float | None:
    if first is None or second is None:
        return None
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _midpoint(first: tuple[float, float] | None, second: tuple[float, float] | None) -> tuple[float, float] | None:
    if first is None or second is None:
        return None
    return ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)


def _angle(first: tuple[float, float] | None, vertex: tuple[float, float] | None, third: tuple[float, float] | None) -> float | None:
    if first is None or vertex is None or third is None:
        return None
    left = (first[0] - vertex[0], first[1] - vertex[1])
    right = (third[0] - vertex[0], third[1] - vertex[1])
    scale = math.hypot(*left) * math.hypot(*right)
    if scale < 1e-6:
        return None
    cosine = max(-1.0, min(1.0, (left[0] * right[0] + left[1] * right[1]) / scale))
    return math.degrees(math.acos(cosine))


def _median(values: Iterable[float]) -> float | None:
    valid = [float(value) for value in values if math.isfinite(float(value))]
    return float(median(valid)) if valid else None


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _bst_family_for_label(*labels: object) -> str | None:
    """Map only unambiguous BST labels to an existing coach family."""
    values = [str(raw or "").strip() for raw in labels]
    # ``model_label`` is the already-decoded public label.  If it is present
    # but explicitly unknown, do not let a contradictory raw label bypass the
    # safety gate in a hand-edited manifest.
    if values and values[0]:
        values = values[:1]
    for value in values:
        if not value:
            continue
        for prefix in ("Top_", "Bottom_", "Top-", "Bottom-"):
            if value.startswith(prefix):
                value = value[len(prefix):]
                break
        if value in _BST_UNKNOWN_LABELS:
            continue
        key = value.casefold()
        # Keep Chinese aliases exact while accepting the English labels used
        # by the BadmintonDB checkpoint.
        for alias, family in _BST_FAMILY_ALIASES.items():
            if key == alias.casefold():
                return family
    return None


def _family_area(family: str) -> str:
    if family == "serve":
        return "serve"
    if family in _FRONT_FAMILIES:
        return "front"
    if family in _OVERHEAD_FAMILIES or family == "defensive_drive":
        return "back"
    return "unknown"


def _bst_confidence_margin(bst_hit: Mapping[str, Any]) -> float | None:
    """Return the Top-1/Top-2 gap in percentage points when auditable."""
    explicit = _number(bst_hit.get("top1_top2_margin"))
    if explicit is not None:
        # Adapters may expose either a probability gap or percentage points.
        percentage_points = explicit * 100.0 if 0.0 <= explicit <= 1.0 else explicit
        return round(_clamp(percentage_points, 0.0, 100.0), 1)
    top = bst_hit.get("top3", [])
    if not isinstance(top, (list, tuple)) or len(top) < 2:
        return None

    def confidence(item: object) -> float | None:
        if not isinstance(item, Mapping):
            return None
        value = _number(item.get("confidence"))
        if value is not None:
            return value
        probability = _number(item.get("probability"))
        return probability * 100.0 if probability is not None else None

    first, second = confidence(top[0]), confidence(top[1])
    if first is None or second is None:
        return None
    return round(_clamp(first - second, 0.0, 100.0), 1)


def _coaching_gate(
    eligible: bool,
    reason_code: str,
    *,
    source: str,
    bst_confidence: float = 0.0,
    bst_margin: float | None = None,
    pose_coverage: float = 0.0,
    ball_coverage: float = 0.0,
) -> dict[str, Any]:
    """Build a stable, machine-readable gate without user-facing hedging."""
    return {
        "eligible": bool(eligible),
        "reason_code": reason_code,
        "source": source,
        "bst_confidence": round(_clamp(bst_confidence, 0.0, 100.0), 1),
        "bst_margin": round(bst_margin, 1) if bst_margin is not None else None,
        "pose_coverage": round(_clamp(pose_coverage, 0.0, 1.0), 3),
        "ball_coverage": round(_clamp(ball_coverage, 0.0, 1.0), 3),
    }


def select_bst_stroke_type(
    rule_stroke: Mapping[str, Any],
    bst_hit: Mapping[str, Any] | None,
    *,
    player: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select a safe type source and return ``(stroke, classification)``.

    BST is allowed to choose a coaching family only when its event, side,
    coverage and confidence gates pass.  A known route/rule disagreement is
    deliberately resolved in favour of the explainable route rule; the BST
    label remains in ``classification`` for audit and UI display.
    """
    stroke = dict(rule_stroke) if isinstance(rule_stroke, Mapping) else {}
    rule_family = str(stroke.get("family", "unknown"))
    rule_label = str(stroke.get("label", "") or stroke.get("base_label", ""))
    rule_confidence = int(_number(stroke.get("confidence"), 0) or 0)
    rule_advice_eligible = (
        rule_family not in _STROKE_UNKNOWN_FAMILIES
        and rule_confidence >= _MIN_RELIABLE_STROKE_CONFIDENCE
    )
    classification: dict[str, Any] = {
        "source": "rule",
        "status": "unavailable",
        "family": rule_family,
        "label": rule_label,
        "rule_family": rule_family,
        "rule_label": rule_label,
        "rule_confidence": rule_confidence,
        "bst_family": "",
        "bst_label": "",
        "bst_raw_label": "",
        "bst_confidence": 0.0,
        "bst_margin": None,
        "type_conflict": False,
        "advice_eligible": rule_advice_eligible,
        "coaching_gate": _coaching_gate(
            rule_advice_eligible,
            "rule_ready" if rule_advice_eligible else "rule_evidence_low",
            source="rule_only",
        ),
        "reason": "按可见球路判断",
    }

    if not isinstance(bst_hit, Mapping):
        return stroke, classification

    bst_label = str(bst_hit.get("model_label", "")).strip()
    bst_raw_label = str(bst_hit.get("model_raw_label", "")).strip()
    bst_confidence = _clamp(float(_number(bst_hit.get("confidence"), 0.0) or 0.0), 0.0, 100.0)
    bst_margin = _bst_confidence_margin(bst_hit)
    bst_family = _bst_family_for_label(bst_label, bst_raw_label)
    classification.update({
        "bst_family": bst_family or "",
        "bst_label": bst_label,
        "bst_raw_label": bst_raw_label,
        "bst_confidence": round(max(0.0, min(100.0, bst_confidence)), 1),
        "bst_margin": bst_margin,
    })

    def fallback(
        status: str,
        reason: str,
        *,
        conflict: bool = False,
        reason_code: str | None = None,
        block_rule: bool = True,
    ):
        # A weak BST card is missing evidence, not negative evidence.  Keep its
        # diagnostic state, but let an independently reliable route rule drive
        # coaching.  Side/type contradictions and a confident unsupported
        # label remain explicit blockers.
        eligible = rule_advice_eligible and not block_rule
        classification.update({
            "status": status if block_rule else "rule_fallback",
            "bst_status": status,
            "source": "rule",
            "family": rule_family,
            "label": rule_label,
            "type_conflict": bool(conflict),
            "advice_eligible": eligible,
            "coaching_gate": _coaching_gate(
                eligible,
                reason_code or status,
                source="bst_blocked" if block_rule else "rule_fallback",
                bst_confidence=bst_confidence,
                bst_margin=bst_margin,
                pose_coverage=float(_number(bst_hit.get("pose_coverage"), 0.0) or 0.0),
                ball_coverage=float(_number(bst_hit.get("ball_coverage"), 0.0) or 0.0),
            ),
            "reason": reason,
        })
        return stroke, classification

    hitter = str(bst_hit.get("hitter", ""))
    model_side = str(bst_hit.get("model_side", ""))
    # An explicitly reported opposite side is independent attribution
    # evidence.  Keep it a hard blocker even when the class probability itself
    # is weak; a weak *type* card is missing evidence, but a side contradiction
    # would otherwise let the wrong athlete vote.
    if player not in {"near", "far"} or (hitter in {"near", "far"} and hitter != player) or (model_side in {"near", "far"} and model_side != player):
        return fallback("side_mismatch", "BST 击球方与姿态归属不一致，沿用球路判断", reason_code="side_mismatch")

    if str(bst_hit.get("confidence_level", "review")) != "ready" or bst_confidence < _BST_MIN_CONFIDENCE:
        return fallback(
            "low_confidence",
            "BST 置信度不足，使用球路判断",
            reason_code="bst_type_confidence_low",
            block_rule=False,
        )
    if bst_hit.get("event_confirmed", False) is not True:
        return fallback(
            "insufficient_evidence",
            "BST 击球事件证据不足，使用球路判断",
            reason_code="event_unconfirmed",
            block_rule=False,
        )
    pose_coverage = float(_number(bst_hit.get("pose_coverage"), 0.0) or 0.0)
    ball_coverage = float(_number(bst_hit.get("ball_coverage"), 0.0) or 0.0)
    if pose_coverage < _BST_MIN_COVERAGE or ball_coverage < _BST_MIN_COVERAGE:
        return fallback(
            "insufficient_evidence",
            "BST 姿态或球点覆盖不足，使用球路判断",
            reason_code="bst_type_coverage_low",
            block_rule=False,
        )
    if not bst_family:
        return fallback("unsupported", "BST 类型不适合生成对应动作建议，沿用球路判断", reason_code="bst_family_unsupported")

    rule_area = str(stroke.get("area", "unknown"))
    bst_area = _family_area(bst_family)
    conflict = False
    if rule_family not in _STROKE_UNKNOWN_FAMILIES and rule_family != bst_family:
        conflict = True
    elif rule_family in _STROKE_UNKNOWN_FAMILIES and rule_area in {"front", "back", "serve"} and rule_area != bst_area:
        conflict = True
    if conflict:
        return fallback(
            "conflict",
            f"BST 识别为{bst_label or '其他类型'}，与球路判断不一致，沿用球路建议",
            conflict=True,
            reason_code="type_conflict",
        )

    status = "agreement" if rule_family == bst_family else "adopted"
    strict_reason = "ready"
    strict_eligible = status == "agreement"
    if status != "agreement":
        strict_reason = "route_type_unconfirmed"
    elif bst_confidence < _BST_COACH_MIN_CONFIDENCE:
        strict_eligible, strict_reason = False, "bst_coach_confidence_low"
    elif bst_margin is None:
        strict_eligible, strict_reason = False, "bst_margin_missing"
    elif bst_margin < _BST_COACH_MIN_MARGIN:
        strict_eligible, strict_reason = False, "bst_margin_low"
    elif pose_coverage < _BST_COACH_MIN_POSE_COVERAGE:
        strict_eligible, strict_reason = False, "bst_pose_coverage_low"
    elif ball_coverage < _BST_COACH_MIN_BALL_COVERAGE:
        strict_eligible, strict_reason = False, "bst_ball_coverage_low"
    elif bst_hit.get("coach_ready") is False:
        strict_eligible = False
        strict_reason = str(bst_hit.get("coach_gate_reason", "bst_adapter_gate_failed") or "bst_adapter_gate_failed")

    # When the route rule already knows the same family, a BST card that does
    # not pass the stricter Motion Coach gate must not veto or relabel it.
    if status == "agreement" and not strict_eligible:
        return fallback(
            "agreement",
            "BST 未达到动作建议门槛，使用球路判断",
            reason_code=strict_reason,
            block_rule=False,
        )

    selected = dict(stroke)
    selected["family"] = bst_family
    selected["base_label"] = bst_label
    selected["label"] = bst_label
    hand_label = str(stroke.get("hand_label", ""))
    if bst_family in _FRONT_FAMILIES and str(stroke.get("hand_status", "")) == "candidate" and hand_label in {"正手", "反手"}:
        selected["label"] = f"{hand_label}{bst_label}"
    selected["area"] = _family_area(bst_family)
    raw_reasons = stroke.get("reasons", [])
    if not isinstance(raw_reasons, (list, tuple)):
        raw_reasons = []
    selected["reasons"] = [f"BST 高置信识别：{bst_label}"] + [str(item) for item in raw_reasons if isinstance(item, str)]
    selected["reasons"] = selected["reasons"][:3]

    classification.update({
        "source": "bst",
        "status": status,
        "family": bst_family,
        "label": selected["label"],
        "advice_eligible": strict_eligible,
        "coaching_gate": _coaching_gate(
            strict_eligible,
            strict_reason,
            source="bst_agreement" if status == "agreement" else "bst_adopted",
            bst_confidence=bst_confidence,
            bst_margin=bst_margin,
            pose_coverage=pose_coverage,
            ball_coverage=ball_coverage,
        ),
        "reason": "BST 与球路类型一致" if rule_family == bst_family else "球路类型信息不足，由 BST 补充",
    })
    return selected, classification


def _confidence_level(value: int) -> str:
    if value >= 72:
        return "medium"
    return "review"


def zone_for_player_position(position: Sequence[object] | None, player: str) -> str | None:
    """Return front/mid/back from a foot point in calibrated court metres.

    The zones are measured from *that player's own net side*.  A narrow buffer
    around boundaries intentionally returns ``None`` instead of pretending a
    player standing on a line belongs to one technical family.
    """
    if player not in {"near", "far"} or not isinstance(position, (list, tuple)) or len(position) < 2:
        return None
    x, y = _number(position[0]), _number(position[1])
    if x is None or y is None or not (-0.45 <= x <= COURT_WIDTH_M + 0.45 and -0.45 <= y <= COURT_LENGTH_M + 0.45):
        return None
    distance_to_net = (y - NET_Y_M) if player == "near" else (NET_Y_M - y)
    if distance_to_net < -ZONE_BUFFER_M or distance_to_net > NET_Y_M + ZONE_BUFFER_M:
        return None
    if distance_to_net <= FRONT_COURT_M - ZONE_BUFFER_M:
        return "front"
    if distance_to_net >= FRONT_COURT_M + ZONE_BUFFER_M and distance_to_net <= MID_COURT_M - ZONE_BUFFER_M:
        return "mid"
    if distance_to_net >= MID_COURT_M + ZONE_BUFFER_M:
        return "back"
    return None


def _axis_separation_degrees(
    first_a: tuple[float, float] | None,
    first_b: tuple[float, float] | None,
    second_a: tuple[float, float] | None,
    second_b: tuple[float, float] | None,
) -> float | None:
    """Return the unsigned 2-D separation between two body axes."""
    if None in {first_a, first_b, second_a, second_b}:
        return None
    first_angle = math.degrees(math.atan2(
        first_b[1] - first_a[1], first_b[0] - first_a[0]
    ))
    second_angle = math.degrees(math.atan2(
        second_b[1] - second_a[1], second_b[0] - second_a[0]
    ))
    difference = abs((first_angle - second_angle) % 180.0)
    return min(difference, 180.0 - difference)


def _phase_pose_sample(
    raw: Mapping[str, Any],
    *,
    arm: str,
) -> dict[str, Any] | None:
    """Extract one scale-normalized temporal sample for a configured arm."""
    try:
        frame = int(raw.get("frame", -1))
    except (AttributeError, TypeError, ValueError):
        return None
    keypoints = raw.get("keypoints")
    if frame < 0 or not isinstance(keypoints, Sequence) or arm not in {"left", "right"}:
        return None
    shoulder_i, elbow_i, wrist_i = (5, 7, 9) if arm == "left" else (6, 8, 10)
    shoulder = _point(keypoints[shoulder_i] if len(keypoints) > shoulder_i else None)
    elbow = _point(keypoints[elbow_i] if len(keypoints) > elbow_i else None)
    wrist = _point(keypoints[wrist_i] if len(keypoints) > wrist_i else None)
    left_shoulder = _point(keypoints[5] if len(keypoints) > 5 else None)
    right_shoulder = _point(keypoints[6] if len(keypoints) > 6 else None)
    left_hip = _point(keypoints[11] if len(keypoints) > 11 else None)
    right_hip = _point(keypoints[12] if len(keypoints) > 12 else None)
    shoulder_center = _midpoint(left_shoulder, right_shoulder)
    hip_center = _midpoint(left_hip, right_hip)
    torso = _distance(shoulder_center, hip_center)
    if shoulder is None or elbow is None or wrist is None or torso is None or torso < 4.0:
        return None

    knees = []
    for hip_i, knee_i, ankle_i in ((11, 13, 15), (12, 14, 16)):
        knee = _angle(
            _point(keypoints[hip_i] if len(keypoints) > hip_i else None),
            _point(keypoints[knee_i] if len(keypoints) > knee_i else None),
            _point(keypoints[ankle_i] if len(keypoints) > ankle_i else None),
        )
        if knee is not None:
            knees.append(knee)
    shoulder_width = _distance(left_shoulder, right_shoulder)
    left_ankle = _point(keypoints[15] if len(keypoints) > 15 else None)
    right_ankle = _point(keypoints[16] if len(keypoints) > 16 else None)
    ankle_width = _distance(left_ankle, right_ankle)
    confidences = []
    for index in (shoulder_i, elbow_i, wrist_i, 11, 12, 13, 14, 15, 16):
        point = keypoints[index] if len(keypoints) > index else None
        confidence = _number(point[2]) if isinstance(point, (list, tuple)) and len(point) >= 3 else None
        if confidence is not None:
            confidences.append(confidence)
    return {
        "frame": frame,
        "wrist": wrist,
        "torso": torso,
        "wrist_height": (shoulder[1] - wrist[1]) / torso,
        "elbow": _angle(shoulder, elbow, wrist),
        "knee": min(knees) if knees else None,
        "stance": (
            ankle_width / shoulder_width
            if ankle_width is not None and shoulder_width is not None and shoulder_width > 4.0
            else None
        ),
        "twist": _axis_separation_degrees(
            left_shoulder, right_shoulder, left_hip, right_hip
        ),
        "confidence": _median(confidences),
    }


def _phase_metrics_for_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    hit_frame: int,
    fps: float,
    arm: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build conservative preparation/contact/recovery measurements.

    All distances are divided by the per-frame torso length and all angular
    changes use the elapsed video time.  The result is therefore comparable
    across camera scale and does not pretend that a 2-D pose is a biomechanical
    ground-truth measurement.
    """
    records: list[dict[str, Any]] = []
    if arm in {"left", "right"}:
        seen_frames: set[int] = set()
        for raw in samples:
            if not isinstance(raw, Mapping):
                continue
            item = _phase_pose_sample(raw, arm=arm)
            if item is None or item["frame"] in seen_frames:
                continue
            seen_frames.add(item["frame"])
            item["delta_seconds"] = (item["frame"] - hit_frame) / max(1.0, float(fps))
            records.append(item)
    records.sort(key=lambda item: item["frame"])

    def window(lower: float, upper: float) -> list[dict[str, Any]]:
        return [
            item for item in records
            if lower <= float(item["delta_seconds"]) <= upper
        ]

    # The production sampler places three frames before contact and four
    # frames after it.  Slightly wider contact bounds keep this helper useful
    # for lower-rate offline callers while the signal-specific checks below
    # retain the stricter phase sample counts.
    prep = window(-0.50, -0.12)
    contact = window(-0.20, 0.20)
    recovery = window(0.20, 0.90)

    def values(items: Sequence[Mapping[str, Any]], key: str) -> list[float]:
        result = []
        for item in items:
            value = _number(item.get(key))
            if value is not None:
                result.append(value)
        return result

    prep_wrist = values(prep, "wrist_height")
    prep_knee = values(prep, "knee")
    prep_stance = values(prep, "stance")
    prep_twist = values(prep, "twist")
    contact_wrist = values(contact, "wrist_height")

    intervals: list[dict[str, Any]] = []
    for before, after in zip(records, records[1:]):
        dt = float(after["delta_seconds"]) - float(before["delta_seconds"])
        if dt <= 0.0 or dt > 0.34:
            continue
        # Keep the expression explicit; operator precedence around ``or`` is
        # otherwise easy to misread and can produce a zero scale.
        torso_before = _number(before.get("torso"))
        torso_after = _number(after.get("torso"))
        torso = ((torso_before or 0.0) + (torso_after or 0.0)) / 2.0
        if torso < 4.0:
            continue
        wrist_speed = None
        wrist_distance = _distance(before.get("wrist"), after.get("wrist"))
        if wrist_distance is not None:
            wrist_speed = wrist_distance / torso / dt

        def angular_speed(key: str) -> float | None:
            first, second = _number(before.get(key)), _number(after.get(key))
            return abs(second - first) / dt if first is not None and second is not None else None

        twist_change = None
        first_twist, second_twist = _number(before.get("twist")), _number(after.get("twist"))
        if first_twist is not None and second_twist is not None:
            twist_change = abs(second_twist - first_twist)
        midpoint = (float(before["delta_seconds"]) + float(after["delta_seconds"])) / 2.0
        intervals.append({
            "start": float(before["delta_seconds"]),
            "end": float(after["delta_seconds"]),
            "midpoint": midpoint,
            "wrist_speed": wrist_speed,
            "elbow_speed": angular_speed("elbow"),
            "knee_speed": angular_speed("knee"),
            "twist_speed": (twist_change / dt) if twist_change is not None else None,
            "twist_change": twist_change,
        })

    def peak(key: str, items: Sequence[Mapping[str, Any]] = intervals) -> tuple[float | None, float | None]:
        candidates = [
            (float(item[key]), float(item["midpoint"]))
            for item in items
            if _number(item.get(key)) is not None
        ]
        return max(candidates, key=lambda pair: pair[0]) if candidates else (None, None)

    peak_wrist, peak_wrist_time = peak("wrist_speed")
    peak_elbow, peak_elbow_time = peak("elbow_speed")
    peak_knee, peak_knee_time = peak("knee_speed")
    peak_twist, peak_twist_time = peak("twist_speed")
    contact_intervals = [item for item in intervals if abs(float(item["midpoint"])) <= 0.20]
    contact_wrist_speed, contact_wrist_speed_time = peak("wrist_speed", contact_intervals)

    recovery_intervals = [
        item for item in intervals
        if float(item["start"]) >= 0.20 and float(item["end"]) <= 0.90
    ]
    recovery_valid: list[dict[str, Any]] = []
    for item in recovery_intervals:
        if all(_number(item.get(key)) is not None for key in ("wrist_speed", "elbow_speed", "knee_speed")):
            recovery_valid.append(item)
    stable_intervals = [
        item for item in recovery_valid
        if float(item["wrist_speed"]) <= 1.15
        and float(item["elbow_speed"]) <= 360.0
        and float(item["knee_speed"]) <= 300.0
    ]
    first_stable: float | None = None
    for first, second in zip(stable_intervals, stable_intervals[1:]):
        # Intervals must be adjacent in the observed timeline; two separated
        # quiet windows do not prove a continuous recovery.
        if abs(float(second["start"]) - float(first["end"])) <= 0.041:
            first_stable = float(second["end"])
            break

    def rounded(value: float | None, digits: int = 3) -> float | None:
        return round(value, digits) if value is not None and math.isfinite(value) else None

    peak_twist_change = max(
        (float(item["twist_change"]) for item in intervals if _number(item.get("twist_change")) is not None),
        default=None,
    )
    peak_twist_change_time = next(
        (float(item["midpoint"]) for item in intervals if item.get("twist_change") == peak_twist_change),
        None,
    )
    peak_times = {
        "shoulder_hip": peak_twist_change_time if (peak_twist_change or 0.0) > 1e-6 else None,
        "elbow": peak_elbow_time if (peak_elbow or 0.0) > 1e-6 else None,
        "knee": peak_knee_time if (peak_knee or 0.0) > 1e-6 else None,
        "wrist": peak_wrist_time if (peak_wrist or 0.0) > 1e-6 else None,
    }
    kinetic_order = [
        name for name, _time in sorted(
            ((name, time) for name, time in peak_times.items() if time is not None),
            key=lambda pair: pair[1],
        )
    ]
    phase_confidences = values(records, "confidence")
    phase_confidence = _median(phase_confidences)
    phase_metrics = {
        "arm": arm,
        "prep_sample_count": len(prep),
        "contact_phase_sample_count": len(contact),
        "recovery_sample_count": len(recovery),
        "prep_wrist_height_over_torso": rounded(max(prep_wrist), 3) if prep_wrist else None,
        "prep_wrist_height_max_over_torso": rounded(max(prep_wrist), 3) if prep_wrist else None,
        "prep_wrist_height_median_over_torso": rounded(_median(prep_wrist), 3),
        "prep_knee_angle_deg": rounded(_median(prep_knee), 2),
        "prep_knee_samples": len(prep_knee),
        "prep_stance_over_shoulders": rounded(_median(prep_stance), 3),
        "prep_stance_samples": len(prep_stance),
        "prep_shoulder_hip_twist_max_deg": rounded(max(prep_twist), 2) if prep_twist else None,
        "prep_twist_samples": len(prep_twist),
        "contact_wrist_height_over_torso": rounded(_median(contact_wrist), 3),
        "contact_wrist_height_samples": len(contact_wrist),
        "contact_wrist_speed_torso_s": rounded(contact_wrist_speed, 3),
        "contact_wrist_speed_time_s": rounded(contact_wrist_speed_time, 3),
        "peak_wrist_speed_torso_s": rounded(peak_wrist, 3),
        "peak_wrist_speed_time_s": rounded(peak_wrist_time, 3),
        "peak_elbow_angular_speed_deg_s": rounded(peak_elbow, 2),
        "peak_elbow_angular_speed_time_s": rounded(peak_elbow_time, 3),
        "peak_knee_angular_speed_deg_s": rounded(peak_knee, 2),
        "peak_knee_angular_speed_time_s": rounded(peak_knee_time, 3),
        "peak_shoulder_hip_change_deg": rounded(peak_twist_change, 2),
        "peak_shoulder_hip_change_time_s": rounded(peak_twist_change_time, 3),
        "peak_shoulder_hip_angular_speed_deg_s": rounded(peak_twist, 2),
        "peak_shoulder_hip_angular_speed_time_s": rounded(peak_twist_time, 3),
        "kinetic_order": kinetic_order,
        "recovery_valid_intervals": len(recovery_valid),
        "recovery_stable_intervals": len(stable_intervals),
        "recovery_observed_seconds": rounded(max((float(item["delta_seconds"]) for item in recovery), default=0.0), 3),
        "recovery_stable_detected": first_stable is not None,
        "recovery_first_stable_seconds": rounded(first_stable, 3),
        "recovery_stable_seconds": rounded(first_stable, 3),
        "phase_pose_confidence": rounded(phase_confidence, 3),
    }
    phase_reasons: list[str] = []
    if arm not in {"left", "right"}:
        phase_reasons.append("arm_unresolved")
    if len(prep) < 2:
        phase_reasons.append("prep_samples_low")
    if len(contact) < 3:
        phase_reasons.append("contact_phase_samples_low")
    if len(recovery) < 3:
        phase_reasons.append("recovery_samples_low")
    if phase_confidence is None or phase_confidence < _MIN_RELIABLE_POSE_CONFIDENCE:
        phase_reasons.append("phase_pose_confidence_low")
    phase_gate = {
        "eligible": not phase_reasons,
        "reason_code": phase_reasons[0] if phase_reasons else "ready",
        "reason_codes": phase_reasons,
        "arm": arm,
        "pose_confidence": rounded(phase_confidence, 3) or 0.0,
        "prep_samples": len(prep),
        "contact_samples": len(contact),
        "recovery_samples": len(recovery),
    }
    return phase_metrics, phase_gate


def extract_pose_metrics(
    samples: Sequence[Mapping[str, Any]],
    *,
    hit_frame: int,
    hit_point: Sequence[object] | None = None,
    fps: float = 25.0,
    dominant_hand: str = "auto",
) -> dict[str, Any]:
    """Extract only scale-normalized, visible 2-D signals around one hit.

    A sample holds COCO-17 ``keypoints`` with confidence in item 3 and may
    carry a calibrated ``court_position``.  We use a short contact window
    (roughly +/- 0.18 s, supplied by the caller) and preserve missing values
    rather than creating a synthetic pose.
    """
    ball = _point(hit_point, minimum_confidence=0.0)
    contact_samples = []
    for raw in samples:
        try:
            frame = int(raw.get("frame", -1))
        except (AttributeError, TypeError, ValueError):
            continue
        keypoints = raw.get("keypoints") if isinstance(raw, Mapping) else None
        if not isinstance(keypoints, Sequence):
            continue
        # The caller samples a sparse timeline.  The closest three samples are
        # enough to stabilize a contact-side proxy without treating an entire
        # rally as the contact pose.
        contact_samples.append((abs(frame - hit_frame), frame, keypoints, raw))
    contact_samples.sort(key=lambda item: item[0])
    contact_samples = contact_samples[:3]

    usable_contact_samples = 0
    elbows: list[float] = []
    reaches: list[float] = []
    knees: list[float] = []
    stances: list[float] = []
    arm_above: list[bool] = []
    contact_distances: list[float] = []
    pose_confidences: list[float] = []
    contact_arms: list[str] = []
    motion_arms: list[str] = []
    side_scores: list[float] = []
    knee_sample_frames = 0

    for _distance_from_hit, _frame_number, keypoints, _raw in contact_samples:
        left_shoulder, right_shoulder = _point(keypoints[5] if len(keypoints) > 5 else None), _point(keypoints[6] if len(keypoints) > 6 else None)
        left_hip, right_hip = _point(keypoints[11] if len(keypoints) > 11 else None), _point(keypoints[12] if len(keypoints) > 12 else None)
        shoulder_center, hip_center = _midpoint(left_shoulder, right_shoulder), _midpoint(left_hip, right_hip)
        torso = _distance(shoulder_center, hip_center)
        shoulder_width = _distance(left_shoulder, right_shoulder)
        if torso is None or torso < 4.0:
            continue
        usable_contact_samples += 1
        visible_confidence = []
        for index in range(5, min(17, len(keypoints))):
            point = keypoints[index]
            if isinstance(point, (list, tuple)) and len(point) >= 3:
                confidence = _number(point[2])
                if confidence is not None:
                    visible_confidence.append(confidence)
        if visible_confidence:
            pose_confidences.append(sum(visible_confidence) / len(visible_confidence))

        arm_candidates: list[tuple[float, str, float | None, float | None, bool]] = []
        for arm, shoulder_i, elbow_i, wrist_i in (("left", 5, 7, 9), ("right", 6, 8, 10)):
            shoulder = _point(keypoints[shoulder_i] if len(keypoints) > shoulder_i else None)
            elbow = _point(keypoints[elbow_i] if len(keypoints) > elbow_i else None)
            wrist = _point(keypoints[wrist_i] if len(keypoints) > wrist_i else None)
            if shoulder is None or elbow is None or wrist is None:
                continue
            tip = (wrist[0] + 0.35 * (wrist[0] - elbow[0]), wrist[1] + 0.35 * (wrist[1] - elbow[1]))
            ball_distance = _distance(tip, ball) if ball is not None else None
            # A virtual racket-tip distance is a contact *association* proxy,
            # not a claim that the racket itself was detected.
            rank = ball_distance / torso if ball_distance is not None else -_distance(shoulder, wrist) / torso
            arm_candidates.append((rank, arm, _angle(shoulder, elbow, wrist), _distance(shoulder, wrist) / torso, wrist[1] < shoulder[1] - 0.10 * torso))
        if arm_candidates:
            if ball is not None:
                chosen = min(arm_candidates, key=lambda item: item[0])
                # A distant virtual tip is not usable contact-arm evidence.
                if chosen[0] <= 1.8:
                    contact_arms.append(chosen[1])
            else:
                chosen = max(arm_candidates, key=lambda item: item[3] or 0.0)
            motion_arms.append(chosen[1])
            if chosen[2] is not None:
                elbows.append(chosen[2])
            if chosen[3] is not None:
                reaches.append(chosen[3])
            arm_above.append(chosen[4])

        frame_knees = []
        for hip_i, knee_i, ankle_i in ((11, 13, 15), (12, 14, 16)):
            knee = _angle(
                _point(keypoints[hip_i] if len(keypoints) > hip_i else None),
                _point(keypoints[knee_i] if len(keypoints) > knee_i else None),
                _point(keypoints[ankle_i] if len(keypoints) > ankle_i else None),
            )
            if knee is not None:
                frame_knees.append(knee)
        if frame_knees:
            knees.extend(frame_knees)
            knee_sample_frames += 1
        left_ankle, right_ankle = _point(keypoints[15] if len(keypoints) > 15 else None), _point(keypoints[16] if len(keypoints) > 16 else None)
        ankle_width = _distance(left_ankle, right_ankle)
        if shoulder_width is not None and shoulder_width > 4.0 and ankle_width is not None:
            stances.append(ankle_width / shoulder_width)
        if ball is not None and shoulder_center is not None:
            body_distance = _distance(ball, shoulder_center)
            if body_distance is not None:
                contact_distances.append(body_distance / torso)
            if shoulder_width is not None and shoulder_width > 4.0 and left_shoulder and right_shoulder:
                axis = (right_shoulder[0] - left_shoulder[0], right_shoulder[1] - left_shoulder[1])
                side_scores.append(((ball[0] - shoulder_center[0]) * axis[0] + (ball[1] - shoulder_center[1]) * axis[1]) / (shoulder_width * shoulder_width))

    pre_positions = []
    hit_positions = []
    post_positions = []
    for raw in samples:
        try:
            frame = int(raw.get("frame", -1))
        except (AttributeError, TypeError, ValueError):
            continue
        point = raw.get("court_position") if isinstance(raw, Mapping) else None
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        x, y = _number(point[0]), _number(point[1])
        if x is None or y is None:
            continue
        if frame < hit_frame:
            pre_positions.append((frame, (x, y)))
        elif frame == hit_frame:
            hit_positions.append((x, y))
        elif frame > hit_frame:
            post_positions.append((frame, (x, y)))
    pre_motion = None
    post_motion = None
    nearby_positions = [
        item[1]
        for item in pre_positions + post_positions
        if abs(item[0] - hit_frame) <= 2
    ]
    hit_position = (
        (_median([item[0] for item in hit_positions]), _median([item[1] for item in hit_positions]))
        if hit_positions
        else nearby_positions[-1] if nearby_positions else None
    )
    if hit_position is not None and pre_positions:
        pre_motion = _distance(hit_position, pre_positions[0][1])
    if hit_position is not None and post_positions:
        post_motion = _distance(hit_position, post_positions[-1][1])

    arm = None
    if contact_arms:
        counts = Counter(contact_arms)
        most_common, count = counts.most_common(1)[0]
        if count / len(contact_arms) >= 0.67:
            arm = most_common
    motion_arm = None
    motion_arm_share = 0.0
    if motion_arms:
        motion_counts = Counter(motion_arms)
        motion_candidate, motion_count = motion_counts.most_common(1)[0]
        motion_arm_share = motion_count / len(motion_arms)
        if motion_arm_share >= 0.67:
            motion_arm = motion_candidate
    configured_hand = str(dominant_hand or "auto").lower()
    if configured_hand not in _VALID_HANDS:
        configured_hand = "auto"
    phase_arm = arm
    phase_arm_conflict = False
    if phase_arm is None and configured_hand in {"left", "right"}:
        if motion_arm is not None and motion_arm != configured_hand:
            phase_arm_conflict = True
        else:
            phase_arm = configured_hand
    if phase_arm is None and configured_hand == "auto":
        phase_arm = motion_arm
    phase_metrics, phase_gate = _phase_metrics_for_samples(
        samples,
        hit_frame=hit_frame,
        fps=max(1.0, float(fps)),
        arm=phase_arm,
    )
    phase_gate["contact_arm"] = arm
    phase_gate["motion_arm"] = motion_arm
    phase_gate["motion_arm_share"] = round(motion_arm_share, 3)
    phase_gate["configured_hand"] = configured_hand
    if phase_arm_conflict:
        phase_gate["eligible"] = False
        phase_gate["reason_codes"] = ["configured_hand_motion_conflict"] + list(phase_gate["reason_codes"])
        phase_gate["reason_code"] = "configured_hand_motion_conflict"
    side_score = _median(side_scores)
    side_consistent = False
    if side_score is not None and side_scores:
        side_consistent = sum((value >= 0) == (side_score >= 0) and abs(value) >= 0.25 for value in side_scores) / len(side_scores) >= 0.75
    return {
        # Count only frames with a usable torso reference.  Merely having a
        # 17-slot keypoint array must not satisfy the multi-frame evidence
        # gate when those points are missing or fully occluded.
        "contact_samples": usable_contact_samples,
        "pose_confidence": round(_median(pose_confidences) or 0.0, 3),
        "contact_arm": arm,
        "motion_arm": motion_arm,
        "motion_arm_share": round(motion_arm_share, 3),
        "phase_arm": phase_arm,
        "elbow_angle": round(_median(elbows), 1) if elbows else None,
        "elbow_samples": len(elbows),
        "reach_over_torso": round(_median(reaches), 2) if reaches else None,
        "reach_samples": len(reaches),
        "knee_angle": round(_median(knees), 1) if knees else None,
        "knee_samples": knee_sample_frames,
        "stance_over_shoulders": round(_median(stances), 2) if stances else None,
        "stance_samples": len(stances),
        "arm_above_shoulder_share": round(sum(arm_above) / len(arm_above), 2) if arm_above else None,
        "arm_above_samples": len(arm_above),
        "contact_distance_over_torso": round(_median(contact_distances), 2) if contact_distances else None,
        "contact_distance_samples": len(contact_distances),
        "contact_side": round(side_score, 2) if side_score is not None else None,
        "contact_side_consistent": side_consistent,
        "pre_motion_m": round(pre_motion, 2) if pre_motion is not None else None,
        "post_motion_m": round(post_motion, 2) if post_motion is not None else None,
        "phase_metrics": phase_metrics,
        "phase_gate": phase_gate,
        "phase_pose_confidence": phase_metrics.get("phase_pose_confidence"),
    }


def hand_variant(metrics: Mapping[str, Any], dominant_hand: str, *, allow_candidate: bool) -> dict[str, Any]:
    """Return a deliberately conservative forehand/backhand candidate label."""
    hand = str(dominant_hand or "auto").lower()
    if not allow_candidate:
        return {"label": "正反手信息不足", "status": "review"}
    if hand not in {"left", "right"}:
        return {"label": "设置持拍手可区分正反手", "status": "review"}
    arm = str(metrics.get("contact_arm", ""))
    side = _number(metrics.get("contact_side"))
    if arm != hand or side is None or abs(side) < 0.25 or not bool(metrics.get("contact_side_consistent")):
        return {"label": "正反手信息不足", "status": "review"}
    forehand = (hand == "right" and side > 0) or (hand == "left" and side < 0)
    confidence = int(round(_clamp(48 + 28 * float(_number(metrics.get("pose_confidence"), 0.0) or 0.0), 45, 76)))
    return {
        "label": "正手" if forehand else "反手",
        "status": "candidate",
        "confidence": confidence,
    }


def _evidence_confidence(
    event: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    origin_zone: str | None,
    receiver_zone: str | None,
    pose: Mapping[str, Any],
) -> int:
    score = 22.0
    if bool(event.get("confirmed")):
        score += 20.0
    if int(_number(trajectory.get("real_points"), 0) or 0) >= 4:
        score += 17.0
    if bool(trajectory.get("speed_reliable")):
        score += 10.0
    if origin_zone in _VALID_ZONES:
        score += 13.0
    if receiver_zone in _VALID_ZONES:
        score += 13.0
    if int(_number(pose.get("contact_samples"), 0) or 0) >= 2 and float(_number(pose.get("pose_confidence"), 0.0) or 0.0) >= 0.45:
        score += 8.0
    # Cap the evidence-only score before attribution penalties.  Capping only
    # at return time would erase both penalties whenever the raw sum exceeds
    # 88 (the common full-evidence case).
    score = _clamp(score, 20.0, 88.0)
    # An unknown ``HitHitter`` can be associated later from a clearly closest
    # visible contact arm.  It is useful for a review card, but receives a
    # meaningful confidence penalty versus a tracking-side confirmed hitter.
    if bool(event.get("hitter_inferred")):
        score -= 12.0
    if bool(trajectory.get("receiver_side_inferred")):
        score -= 8.0
    return int(round(_clamp(score, 20.0, 88.0)))


def classify_stroke(
    event: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    *,
    origin_zone: str | None,
    receiver_zone: str | None,
    pose: Mapping[str, Any],
    dominant_hand: str = "auto",
) -> dict[str, Any]:
    """Classify a route as a *candidate*, never as an asserted technique."""
    confidence = _evidence_confidence(event, trajectory, origin_zone, receiver_zone, pose)
    real_points = int(_number(trajectory.get("real_points"), 0) or 0)
    arc = float(_number(trajectory.get("arc_signal"), 0.0) or 0.0)
    flight = float(_number(trajectory.get("flight_seconds"), 0.0) or 0.0)
    speed_percentile = _number(trajectory.get("speed_percentile"))
    speed_reliable = bool(trajectory.get("speed_reliable"))
    lateral = abs(float(_number(trajectory.get("lateral_player_m"), 0.0) or 0.0))
    arm_high = float(_number(pose.get("arm_above_shoulder_share"), 0.0) or 0.0) >= 0.67
    family = "unknown"
    reasons: list[str] = []

    route_candidate = bool(event.get("route_candidate", event.get("confirmed")))
    if not route_candidate:
        reasons.append("击球事件缺少可用的实测球点或最低击球质量")
    elif real_points < 4:
        reasons.append("出球段少于 4 个真实球点")
    elif int(_number(event.get("rally_hit_index"), 0) or 0) == 1:
        family = "serve"
        reasons.extend(["回合首拍", f"{real_points} 个真实出球球点"])
    elif origin_zone == "front":
        if receiver_zone == "back":
            if arc >= 0.12 or flight >= 1.10:
                family = "lift"
                reasons.append("前场起拍，后场接球窗口，且存在较长飞行/弧线信号")
            else:
                family = "push"
                reasons.append("前场起拍并指向对方后场，未见高弧强证据")
        elif receiver_zone == "front":
            if lateral >= 1.15:
                family = "cross_net"
                reasons.append("前场对前场且两次接触位置横向跨越明显")
            else:
                family = "net"
                reasons.append("前场对前场的短球路线")
        elif receiver_zone == "mid":
            if speed_reliable and speed_percentile is not None and speed_percentile >= 0.82 and arm_high:
                family = "net_kill"
                reasons.append("前场快速出球、可见较高接触臂，指向中场")
            else:
                family = "push"
                reasons.append("前场指向中场的较直接路线")
        else:
            family = "front_unknown"
            reasons.append("前场起拍，但下一拍接球区域不可见")
    elif origin_zone == "back":
        if receiver_zone == "front":
            if speed_reliable and speed_percentile is not None and speed_percentile >= 0.85 and arm_high:
                family = "smash"
                reasons.append("后场上手可见，视频内速度分位较高，下一拍在前场")
            else:
                family = "drop"
                reasons.append("后场至前场的控制球路线，未满足快速下压全部证据")
        elif receiver_zone == "back":
            if arc >= 0.14 and flight >= 1.10:
                family = "clear"
                reasons.append("后场至后场且存在较长飞行和弧线信号")
            elif arc < 0.14:
                family = "drive_clear"
                reasons.append("后场深球路线较平，接近平高球")
            else:
                family = "back_unknown"
                reasons.append("后场深球的弧线证据不足以区分高远和平高")
        elif receiver_zone == "mid":
            if speed_reliable and speed_percentile is not None and speed_percentile >= 0.85 and arm_high:
                family = "smash"
                reasons.append("后场快速上手路线指向中前场")
            elif arc < 0.12:
                family = "defensive_drive"
                reasons.append("后场低平路线，主动性需结合回合判断")
            else:
                family = "drive_clear"
                reasons.append("后场中距离较平路线")
        else:
            family = "back_unknown"
            reasons.append("后场起拍，但下一拍接球区域不可见")
    else:
        reasons.append("击球者脚点未能稳定映射到前/中/后场")

    # A quick net-kill candidate must meet both observable conditions.  This
    # override prevents a slow front-court ball from being dressed up as a kill.
    if family in {"push", "front_unknown"} and origin_zone == "front" and receiver_zone in {"front", "mid"}:
        if speed_reliable and speed_percentile is not None and speed_percentile >= 0.88 and arm_high:
            family = "net_kill"
            reasons.append("视频内速度进入高分位且接触臂高于肩部")

    allow_hand = family in _FRONT_FAMILIES
    hand = hand_variant(pose, dominant_hand, allow_candidate=allow_hand)
    base_label = _FAMILY_LABELS[family]
    label = base_label
    if family not in {"serve", "unknown", "front_unknown", "back_unknown"}:
        if hand["status"] == "candidate":
            label = f"{hand['label']}{base_label}"
    if family == "defensive_drive":
        label += " · 结合回合判断主动性"

    # Technical family confidence is capped below a confirmation.  Single
    # broadcast video cannot elevate a candidate into a definitive call.
    if route_candidate and not bool(event.get("confirmed")):
        confidence = min(confidence, 58)
        reasons.append("击球信号较弱，建议优先查看视频")
    if family in {"unknown", "front_unknown", "back_unknown"}:
        confidence = min(confidence, 58)
    if family in {"cross_net", "net", "defensive_drive"}:
        confidence = min(confidence, 74)
    if not speed_reliable and family in {"smash", "net_kill", "drive_clear"}:
        confidence = min(confidence, 65)
    return {
        "family": family,
        "label": label,
        "base_label": base_label,
        "area": "serve" if family == "serve" else "front" if family in _FRONT_FAMILIES | {"front_unknown"} else "back" if family in _OVERHEAD_FAMILIES | {"defensive_drive", "back_unknown"} else "unknown",
        "confidence": int(confidence),
        "confidence_level": _confidence_level(int(confidence)),
        "hand_label": hand["label"],
        "hand_status": hand["status"],
        "reasons": reasons[:3],
    }


def _evaluation_coaching_gate(
    stroke: Mapping[str, Any],
    pose: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    signals: Sequence[str],
) -> dict[str, Any]:
    """Decide whether visible issue signals may enter repeat aggregation."""
    family = str(stroke.get("family", "unknown"))
    confidence = int(_number(stroke.get("confidence"), 0) or 0)
    contact_samples = int(_number(pose.get("contact_samples"), 0) or 0)
    pose_confidence = float(_number(pose.get("pose_confidence"), 0.0) or 0.0)
    real_points = int(_number(trajectory.get("real_points"), 0) or 0)
    reason_codes: list[str] = []
    if family in _STROKE_UNKNOWN_FAMILIES or family not in _FAMILY_LABELS:
        reason_codes.append("stroke_family_unknown")
    if confidence < _MIN_RELIABLE_STROKE_CONFIDENCE:
        reason_codes.append("stroke_confidence_low")
    if contact_samples < _MIN_RELIABLE_CONTACT_SAMPLES:
        reason_codes.append("contact_samples_low")
    if pose_confidence < _MIN_RELIABLE_POSE_CONFIDENCE:
        reason_codes.append("pose_confidence_low")
    if real_points < _MIN_RELIABLE_REAL_POINTS:
        reason_codes.append("trajectory_points_low")
    if not signals:
        reason_codes.append("no_actionable_signal")
    return {
        "eligible_for_aggregation": not reason_codes,
        "reason_code": reason_codes[0] if reason_codes else "ready",
        "reason_codes": reason_codes,
        "stroke_confidence": confidence,
        "contact_samples": contact_samples,
        "pose_confidence": round(pose_confidence, 3),
        "real_points": real_points,
    }


def _phase_evaluation_gate(
    pose: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    signals: Sequence[str],
) -> dict[str, Any]:
    """Gate type-agnostic temporal signals independently of route family."""
    raw_gate = pose.get("phase_gate", {})
    raw_gate = raw_gate if isinstance(raw_gate, Mapping) else {}
    phase_metrics = pose.get("phase_metrics", {})
    phase_metrics = phase_metrics if isinstance(phase_metrics, Mapping) else {}
    pose_confidence = float(
        _number(raw_gate.get("pose_confidence"), phase_metrics.get("phase_pose_confidence", 0.0))
        or 0.0
    )
    contact_samples = int(_number(pose.get("contact_samples"), 0) or 0)
    real_points = int(_number(trajectory.get("real_points"), 0) or 0)
    reason_codes: list[str] = []
    if trajectory.get("event_confirmed") is False:
        reason_codes.append("event_unconfirmed")
    if contact_samples < _MIN_RELIABLE_CONTACT_SAMPLES:
        reason_codes.append("contact_samples_low")
    if pose_confidence < _MIN_RELIABLE_POSE_CONFIDENCE:
        reason_codes.append("phase_pose_confidence_low")
    if real_points < _MIN_RELIABLE_REAL_POINTS:
        reason_codes.append("trajectory_points_low")

    raw_reasons = raw_gate.get("reason_codes", [])
    raw_reasons = list(raw_reasons) if isinstance(raw_reasons, (list, tuple)) else []
    common_reasons = {
        "arm_unresolved",
        "configured_hand_motion_conflict",
        "phase_pose_confidence_low",
    }
    signal_reasons = set(common_reasons)
    if any(signal in {"preparation_support", "overhead_preparation"} for signal in signals):
        signal_reasons.add("prep_samples_low")
    if "overhead_preparation" in signals:
        signal_reasons.add("contact_phase_samples_low")
    if "recovery_stability" in signals:
        signal_reasons.add("recovery_samples_low")
    for code in raw_reasons:
        if isinstance(code, str) and code in signal_reasons and code not in reason_codes:
            reason_codes.append(code)
    if not signals:
        reason_codes.append("no_actionable_phase_signal")
    return {
        "eligible": not reason_codes,
        "eligible_for_aggregation": not reason_codes,
        "reason_code": reason_codes[0] if reason_codes else "ready",
        "reason_codes": reason_codes,
        "arm": raw_gate.get("arm"),
        "pose_confidence": round(pose_confidence, 3),
        "contact_samples": contact_samples,
        "real_points": real_points,
    }


def _advice_candidates(family: str, signals: Sequence[str]) -> list[dict[str, Any]]:
    candidates = []
    for signal in signals:
        template = _RECOMMENDATION_TEMPLATES.get(signal)
        if not template:
            continue
        candidates.append({
            "status": "candidate",
            "family": family,
            "signal": signal,
            "problem": template["problem"],
            "action": template["action"],
        })
    return candidates


def evaluate_stroke(
    stroke: Mapping[str, Any],
    pose: Mapping[str, Any],
    trajectory: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a visible-pose observation and a technique-specific prompt."""
    family = str(stroke.get("family", "unknown"))
    observations: list[str] = []
    suggestions: list[str] = []
    signals: list[str] = []
    phase_signals: list[str] = []
    knee = _number(pose.get("knee_angle"))
    stance = _number(pose.get("stance_over_shoulders"))
    elbow = _number(pose.get("elbow_angle"))
    reach = _number(pose.get("reach_over_torso"))
    contact_distance = _number(pose.get("contact_distance_over_torso"))
    post_motion = _number(pose.get("post_motion_m"))
    contact_arm = str(pose.get("contact_arm", ""))
    arm_above_share = float(_number(pose.get("arm_above_shoulder_share"), 0.0) or 0.0)
    elbow_samples = int(_number(pose.get("elbow_samples"), 0) or 0)
    reach_samples = int(_number(pose.get("reach_samples"), 0) or 0)
    knee_samples = int(_number(pose.get("knee_samples"), 0) or 0)
    stance_samples = int(_number(pose.get("stance_samples"), 0) or 0)
    arm_above_samples = int(_number(pose.get("arm_above_samples"), 0) or 0)
    contact_distance_samples = int(_number(pose.get("contact_distance_samples"), 0) or 0)
    visible_pose = int(_number(pose.get("contact_samples"), 0) or 0) >= 2 and float(_number(pose.get("pose_confidence"), 0.0) or 0.0) >= 0.45

    if not visible_pose:
        return {
            "grade": "review",
            "advice_status": "suppressed",
            "title": "姿态信号不足",
            "detail": "接触窗口关键点较少，仅显示球路信息。",
            "observations": ["可见关键点少于 2 帧或清晰度偏低"],
            "suggestions": [],
            "signals": [],
            "phase_signals": [],
            "eligible_signals": [],
            "phase_eligible_signals": [],
            "phase_advice_eligible": False,
            "advice_candidates": [],
            "coaching_gate": _evaluation_coaching_gate(stroke, pose, trajectory, []),
            "phase_gate": _phase_evaluation_gate(pose, trajectory, []),
            "caveat": "仅描述二维画面中可见的姿态和球路。",
        }

    phase_metrics = pose.get("phase_metrics", {})
    phase_metrics = phase_metrics if isinstance(phase_metrics, Mapping) else {}
    prep_samples = int(_number(phase_metrics.get("prep_sample_count"), 0) or 0)
    prep_knee = _number(phase_metrics.get("prep_knee_angle_deg"))
    prep_knee_samples = int(_number(phase_metrics.get("prep_knee_samples"), 0) or 0)
    prep_stance = _number(phase_metrics.get("prep_stance_over_shoulders"))
    prep_stance_samples = int(_number(phase_metrics.get("prep_stance_samples"), 0) or 0)
    prep_wrist = _number(phase_metrics.get("prep_wrist_height_max_over_torso"))
    contact_wrist = _number(phase_metrics.get("contact_wrist_height_over_torso"))
    contact_wrist_samples = int(_number(phase_metrics.get("contact_wrist_height_samples"), 0) or 0)
    peak_wrist_speed = _number(phase_metrics.get("peak_wrist_speed_torso_s"))
    recovery_intervals = int(_number(phase_metrics.get("recovery_valid_intervals"), 0) or 0)
    recovery_observed = float(_number(phase_metrics.get("recovery_observed_seconds"), 0.0) or 0.0)
    recovery_stable = bool(phase_metrics.get("recovery_stable_detected"))
    recovery_stable_seconds = _number(phase_metrics.get("recovery_first_stable_seconds"))
    if prep_samples >= 3 and (
        prep_knee is not None
        and prep_knee_samples >= 3
        and prep_knee >= 165.0
        or prep_stance is not None
        and prep_stance_samples >= 3
        and prep_stance < 0.52
    ):
        phase_signals.append("preparation_support")
        suggestions.append("来球过网前完成轻微屈膝和分腿准备，再启动最后一步。")
    if (
        prep_samples >= 3
        and prep_wrist is not None
        and prep_wrist < -0.08
        and contact_wrist is not None
        and contact_wrist >= 0.05
        and contact_wrist_samples >= 2
        and peak_wrist_speed is not None
        and peak_wrist_speed >= 1.0
    ):
        phase_signals.append("overhead_preparation")
        suggestions.append("提前完成架拍，让触球前最后一段只负责加速。")
    if (
        float(_number(trajectory.get("flight_seconds"), 0.0) or 0.0) >= 0.75
        and recovery_intervals >= 3
        and recovery_observed >= 0.70
        and (not recovery_stable or recovery_stable_seconds is None or recovery_stable_seconds > 0.65)
    ):
        phase_signals.append("recovery_stability")
        suggestions.append("随挥结束立即收拍回到身前，并衔接回位第一步。")

    if family == "serve":
        if knee is not None:
            observations.append(f"准备窗口可见膝角约 {knee:.0f}°")
        if stance is not None:
            observations.append(f"双脚支撑面约为肩宽的 {stance:.2f} 倍")
        if (
            knee is not None and knee_samples >= _MIN_RELIABLE_CONTACT_SAMPLES and knee >= 165.0
            or stance is not None and stance_samples >= _MIN_RELIABLE_CONTACT_SAMPLES and stance < 0.52
        ):
            suggestions.append("发球前先让双脚落稳、保持轻微可控屈膝；出手后尽快回到接下一拍的准备姿态。")
            signals.append("serve_preparation")
        if post_motion is not None and post_motion < 0.18:
            suggestions.append("出手后立即启动第一步，尽快回到准备姿态。")
            signals.append("serve_first_step")
        title = "发球重点"
    elif family in _FRONT_FAMILIES | {"front_unknown"}:
        if knee is not None:
            observations.append(f"接触附近膝角约 {knee:.0f}°")
        if stance is not None:
            observations.append(f"接触附近支撑面约为肩宽的 {stance:.2f} 倍")
        if contact_distance is not None:
            observations.append(f"球点到肩部中心的二维距离约为躯干长 {contact_distance:.2f} 倍")
        if (
            knee is not None and knee_samples >= _MIN_RELIABLE_CONTACT_SAMPLES and knee >= 168.0
            or stance is not None and stance_samples >= _MIN_RELIABLE_CONTACT_SAMPLES and stance < 0.48
        ):
            suggestions.append("前场最后一步先落稳，保持轻微屈膝和可用的支撑面，再完成短促出拍。")
            signals.append("front_support")
        if (
            contact_distance is not None
            and contact_distance_samples >= _MIN_RELIABLE_CONTACT_SAMPLES
            and contact_distance < 0.78
        ):
            suggestions.append("本球触球代理量偏贴近躯干；回看能否更早到球，把可控触球点留在身体前方。")
            signals.append("front_contact_space")
        if family == "net_kill":
            suggestions.append("扑球时争取更高的触球点，并保持落地平衡。")
        elif family == "lift":
            suggestions.append("挑球时先稳住低位支撑，再把球送向后场。")
        elif family == "cross_net":
            suggestions.append("勾对角时以前脚支撑，保持短促出拍。")
        elif family == "net":
            suggestions.append("放网或搓球时保持动作紧凑，击球后及时回位。")
        else:
            suggestions.append("推球时让肘和拍头保持在身体前方，避免触球点落到身后。")
        title = "前场重点"
    elif family in _OVERHEAD_FAMILIES | {"defensive_drive", "back_unknown"}:
        if elbow is not None and reach is not None:
            observations.append(f"可见持拍臂肘角约 {elbow:.0f}°，腕肩距离约为躯干长 {reach:.2f} 倍")
        if knee is not None:
            observations.append(f"接触附近膝角约 {knee:.0f}°")
        if (
            family in _OVERHEAD_FAMILIES
            and contact_arm in {"left", "right"}
            and arm_above_share >= 0.67
            and elbow_samples >= _MIN_RELIABLE_CONTACT_SAMPLES
            and reach_samples >= _MIN_RELIABLE_CONTACT_SAMPLES
            and arm_above_samples >= _MIN_RELIABLE_CONTACT_SAMPLES
            and elbow is not None
            and reach is not None
            and elbow < 145.0
            and reach < 1.05
        ):
            suggestions.append("上手击球时拉开引拍、触球点和随挥空间，肩膀保持放松。")
            signals.append("overhead_extension")
        if knee is not None and knee_samples >= _MIN_RELIABLE_CONTACT_SAMPLES and knee >= 170.0:
            suggestions.append("支撑腿偏直；练习更早到位和可控蹬地，给落地回位留出空间。")
            signals.append("back_support")
        if family == "smash":
            suggestions.append("杀球时在身体前上方触球，落地后立即衔接下一步。")
        elif family == "clear":
            suggestions.append("高远球在身体前上方触球，落地后尽早回到准备区域。")
        elif family == "drop":
            suggestions.append("吊球准备尽量接近高远球，在触球前再收力。")
        elif family == "defensive_drive":
            suggestions.append("后场平抽提前架拍、缩短引拍，并借助来球力量。")
        else:
            suggestions.append("优先关注转身到位、身体前方触球和击球后的第一步。")
        title = "后场重点"
    else:
        # Exact route family is deliberately optional for the three temporal
        # mechanics above.  Their own phase/event/pose gates still apply, so
        # an unknown family can contribute only a directly visible generic
        # signal and never a family-specific prompt.
        title = "动作阶段"
        observations.append("精确球种未用于通用阶段信号")

    if not suggestions:
        suggestions.append("可见动作信号稳定，保持当前节奏并衔接下一拍。")
    grade = "focus" if signals or phase_signals else "steady"
    type_gate = _evaluation_coaching_gate(stroke, pose, trajectory, signals)
    phase_gate = _phase_evaluation_gate(pose, trajectory, phase_signals)
    eligible_signals = list(signals) if type_gate["eligible_for_aggregation"] else []
    phase_eligible_signals = list(phase_signals) if phase_gate["eligible_for_aggregation"] else []
    any_eligible = bool(eligible_signals or phase_eligible_signals)
    if any_eligible:
        combined_reason_codes: list[str] = []
    else:
        combined_reason_codes = []
        for code in list(type_gate["reason_codes"]) + list(phase_gate["reason_codes"]):
            if code not in combined_reason_codes:
                combined_reason_codes.append(code)
    coaching_gate = {
        "eligible_for_aggregation": any_eligible,
        "reason_code": combined_reason_codes[0] if combined_reason_codes else "ready",
        "reason_codes": combined_reason_codes,
        "stroke_confidence": type_gate["stroke_confidence"],
        "contact_samples": type_gate["contact_samples"],
        "pose_confidence": type_gate["pose_confidence"],
        "real_points": type_gate["real_points"],
    }
    return {
        "grade": grade,
        # A single hit is never promoted to a product recommendation.  It is
        # either an aggregation candidate or suppressed; ``summarize_reviews``
        # is the only function that emits ``status=\"reliable\"`` advice.
        "advice_status": "candidate" if any_eligible else "suppressed",
        "title": title,
        "detail": "；".join(observations[:3]) if observations else "接触窗口关键点可见，未发现需要优先调整的信号。",
        "observations": observations[:4],
        "suggestions": suggestions[:3],
        "signals": signals,
        "phase_signals": phase_signals,
        "eligible_signals": eligible_signals,
        "phase_eligible_signals": phase_eligible_signals,
        "phase_advice_eligible": bool(phase_eligible_signals),
        "advice_candidates": _advice_candidates(
            family,
            list(eligible_signals) + list(phase_eligible_signals),
        ),
        "coaching_gate": coaching_gate,
        "type_gate": type_gate,
        "phase_gate": phase_gate,
        "caveat": "仅描述二维画面中可见的姿态和球路。",
    }


def _review_identity(review: Mapping[str, Any]) -> tuple[object, ...] | None:
    """Return a stable strike identity so duplicate records never add votes."""
    evidence = review.get("evidence", {})
    if not isinstance(evidence, Mapping):
        return None
    frame = _number(evidence.get("frame"))
    if frame is None or frame < 0:
        return None
    rally = int(_number(review.get("rally_id"), 0) or 0)
    hit = int(_number(review.get("hit_index"), 0) or 0)
    # Event IDs remain stable when a refined detector shifts the associated
    # contact frame by one or two images.  Legacy records without those IDs
    # fall back to the globally unique decoded frame.
    if rally > 0 and hit > 0:
        return ("event", rally, hit)
    return ("frame", int(round(frame)))


def _review_frame(review: Mapping[str, Any]) -> int | None:
    evidence = review.get("evidence", {})
    if not isinstance(evidence, Mapping):
        return None
    frame = _number(evidence.get("frame"))
    return int(round(frame)) if frame is not None and frame >= 0 else None


def _review_type_is_reliable(review: Mapping[str, Any]) -> bool:
    """Apply explicit selector gates before a review can cast any vote."""
    stroke = review.get("stroke", {})
    if not isinstance(stroke, Mapping):
        return False
    family = str(stroke.get("family", "unknown"))
    confidence = int(_number(stroke.get("confidence"), 0) or 0)
    if family not in _FAMILY_LABELS or family in _STROKE_UNKNOWN_FAMILIES:
        return False
    if confidence < _MIN_RELIABLE_STROKE_CONFIDENCE:
        return False

    # New reports carry an explicit top-level gate.  ``False`` always wins;
    # absence is accepted only for rule-only legacy inputs covered by tests.
    if review.get("advice_eligible") is False:
        return False
    attribution_gate = review.get("attribution_gate", {})
    if isinstance(attribution_gate, Mapping) and attribution_gate.get("eligible") is False:
        return False
    classification = review.get("classification", {})
    if not isinstance(classification, Mapping) or not classification:
        return True
    if classification.get("advice_eligible") is False:
        return False
    gate = classification.get("coaching_gate", {})
    if isinstance(gate, Mapping) and gate.get("eligible") is False:
        return False
    status = str(classification.get("status", "unavailable"))
    source = str(classification.get("source", "rule"))
    if status in _EXPLICITLY_BLOCKED_CLASSIFICATION_STATUSES:
        return False
    # A model-derived family must agree with an independently known route.
    # Old BST reports without the strict explicit gate cannot be promoted.
    if source == "bst":
        return status == "agreement" and classification.get("advice_eligible") is True
    return True


def _review_phase_signal_is_reliable(review: Mapping[str, Any], signal: str) -> bool:
    """Validate a temporal signal without requiring an exact stroke family."""
    if signal not in _TYPE_AGNOSTIC_SIGNALS:
        return False
    attribution_gate = review.get("attribution_gate", {})
    if isinstance(attribution_gate, Mapping) and attribution_gate.get("eligible") is False:
        return False
    classification = review.get("classification", {})
    if isinstance(classification, Mapping) and classification:
        if classification.get("type_conflict") is True:
            return False
        if str(classification.get("status", "")) in _EXPLICITLY_BLOCKED_CLASSIFICATION_STATUSES:
            return False
    if review.get("phase_advice_eligible") is False:
        return False
    evaluation = review.get("evaluation", {})
    if not isinstance(evaluation, Mapping):
        return False
    if evaluation.get("phase_advice_eligible") is not True:
        return False
    gate = evaluation.get("phase_gate", {})
    if isinstance(gate, Mapping) and gate:
        eligible = gate.get("eligible_for_aggregation", gate.get("eligible"))
        if eligible is not True:
            return False
    return True


def _review_signal_is_reliable(review: Mapping[str, Any], signal: str) -> bool:
    if signal in _TYPE_AGNOSTIC_SIGNALS:
        return _review_phase_signal_is_reliable(review, signal)
    if not _review_type_is_reliable(review):
        return False
    evaluation = review.get("evaluation", {})
    if not isinstance(evaluation, Mapping):
        return False
    gate = evaluation.get("type_gate", evaluation.get("coaching_gate", {}))
    if isinstance(gate, Mapping) and gate and gate.get("eligible_for_aggregation") is not True:
        return False
    return True


def _review_signals(evaluation: Mapping[str, Any]) -> list[str]:
    # New evaluations expose the post-evidence-gate list.  Fall back to the
    # historical field only when reading a legacy review with no such key.
    key = "eligible_signals" if "eligible_signals" in evaluation else "signals"
    raw = evaluation.get(key, [])
    if not isinstance(raw, (list, tuple)):
        return []
    return [signal for signal in raw if isinstance(signal, str) and signal in _RECOMMENDATION_TEMPLATES]


def _review_phase_signals(evaluation: Mapping[str, Any]) -> list[str]:
    key = "phase_eligible_signals" if "phase_eligible_signals" in evaluation else "phase_signals"
    raw = evaluation.get(key, [])
    if not isinstance(raw, (list, tuple)):
        return []
    return [
        signal for signal in raw
        if isinstance(signal, str) and signal in _TYPE_AGNOSTIC_SIGNALS
    ]


def _signal_matches_family(signal: str, family: str) -> bool:
    if signal in _TYPE_AGNOSTIC_SIGNALS:
        return True
    if signal.startswith("serve_"):
        return family == "serve"
    if signal.startswith("front_"):
        return family in _FRONT_FAMILIES
    if signal == "overhead_extension":
        return family in _OVERHEAD_FAMILIES
    if signal == "back_support":
        return family in _OVERHEAD_FAMILIES | {"defensive_drive"}
    return False


def summarize_reviews(reviews: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Emit advice only for three independent hits in a compatible group."""
    family_reviews: defaultdict[str, dict[tuple[object, ...], Mapping[str, Any]]] = defaultdict(dict)
    signal_reviews: defaultdict[tuple[str, str], dict[tuple[object, ...], Mapping[str, Any]]] = defaultdict(dict)
    reliable_confidences: list[int] = []
    identity_reviews: defaultdict[tuple[object, ...], list[Mapping[str, Any]]] = defaultdict(list)
    frame_identities: defaultdict[int, set[tuple[object, ...]]] = defaultdict(set)
    for review in reviews:
        if not isinstance(review, Mapping):
            continue
        identity = _review_identity(review)
        frame = _review_frame(review)
        if identity is None or frame is None:
            continue
        identity_reviews[identity].append(review)
        frame_identities[frame].add(identity)

    # The same decoded frame cannot be two independent shuttle contacts.  If
    # conflicting editable IDs claim otherwise, discard every linked record
    # instead of letting one frame vote in multiple family/signal buckets.
    ambiguous_identities = {
        identity
        for identities in frame_identities.values()
        if len(identities) > 1
        for identity in identities
    }

    for identity, grouped_reviews in identity_reviews.items():
        if identity in ambiguous_identities or not grouped_reviews:
            continue
        families = {
            str(dict(review.get("stroke", {})).get("family", "unknown"))
            for review in grouped_reviews
        }
        if len(families) != 1:
            continue
        family = next(iter(families))
        representative = min(
            grouped_reviews,
            key=lambda item: _review_frame(item) if _review_frame(item) is not None else 10**18,
        )
        type_reliable = all(_review_type_is_reliable(review) for review in grouped_reviews)
        if type_reliable:
            family_reviews[family][identity] = representative
            confidences = [
                int(_number(dict(review.get("stroke", {})).get("confidence"), 0) or 0)
                for review in grouped_reviews
            ]
            reliable_confidences.append(min(confidences))

        signal_sets = []
        for review in grouped_reviews:
            evaluation = review.get("evaluation", {})
            if not isinstance(evaluation, Mapping):
                signal_sets = []
                break
            signal_sets.append(set(_review_signals(evaluation)) | set(_review_phase_signals(evaluation)))
        common_signals = set.intersection(*signal_sets) if signal_sets else set()
        for signal in common_signals:
            if not _signal_matches_family(signal, family):
                continue
            if not all(_review_signal_is_reliable(review, signal) for review in grouped_reviews):
                continue
            coach_group, _group_label = _SIGNAL_COACH_GROUPS[signal]
            signal_reviews[(coach_group, signal)][identity] = representative

        if common_signals and not type_reliable:
            phase_qualities = []
            for review in grouped_reviews:
                evaluation = review.get("evaluation", {})
                gate = evaluation.get("phase_gate", {}) if isinstance(evaluation, Mapping) else {}
                if isinstance(gate, Mapping):
                    confidence = _number(gate.get("pose_confidence"))
                    if confidence is not None:
                        phase_qualities.append(int(round(_clamp(confidence, 0.0, 1.0) * 100)))
            if phase_qualities:
                reliable_confidences.append(min(phase_qualities))

    family_counts = Counter({family: len(grouped) for family, grouped in family_reviews.items()})
    summaries = []
    for family, count in family_counts.most_common():
        confidences = [
            int(_number(dict(review.get("stroke", {})).get("confidence"), 0) or 0)
            for review in family_reviews[family].values()
        ]
        summaries.append({
            "family": family,
            "label": _FAMILY_LABELS[family],
            "count": count,
            "average_confidence": int(round(sum(confidences) / max(1, len(confidences)))),
        })

    ordered_groups = sorted(
        signal_reviews.items(),
        key=lambda item: (-len(item[1]), item[0][0], item[0][1]),
    )
    recommendations = []
    for (coach_group, signal), unique_reviews in ordered_groups:
        repeat_count = len(unique_reviews)
        template = _RECOMMENDATION_TEMPLATES.get(signal)
        if repeat_count < _MIN_REPEATED_HITS or not template:
            continue
        grouped = sorted(
            unique_reviews.values(),
            key=lambda item: int(_number(dict(item.get("evidence", {})).get("frame"), 0) or 0),
        )
        _mapped_group, stroke_label = _SIGNAL_COACH_GROUPS[signal]
        # Every new product recommendation is exposed under its compatible
        # coaching group.  The HTTP boundary still accepts legacy exact-family
        # cards from older manifests, but fresh aggregates are never split by
        # clear/drive/smash or by front-court shot subtype.
        output_family = coach_group
        problem = template["problem"]
        action = template["action"]
        recommendations.append({
            "status": "reliable",
            "advice_status": "reliable",
            "priority": "high" if repeat_count >= 5 else "medium",
            "family": output_family,
            "stroke_label": stroke_label,
            "signal": signal,
            "repeat_count": repeat_count,
            "title": f"{stroke_label} · 依据 {repeat_count} 拍",
            "problem": problem,
            "action": action,
            # Keep the old text fields populated for stored-report and client
            # compatibility; new clients should render problem/action.
            "detail": f"问题：{problem}训练：{action}",
            "metric": f"{repeat_count} 次独立击球 · {template['metric']}",
            "evidence": [
                dict(item.get("evidence", {}))
                for item in grouped[:3]
                if isinstance(item.get("evidence"), Mapping)
            ],
        })
    quality = int(round(sum(reliable_confidences) / len(reliable_confidences))) if reliable_confidences else 0
    return summaries[:12], recommendations[:3], quality


__all__ = [
    "COURT_LENGTH_M",
    "COURT_WIDTH_M",
    "classify_stroke",
    "evaluate_stroke",
    "extract_pose_metrics",
    "select_bst_stroke_type",
    "summarize_reviews",
    "zone_for_player_position",
]
