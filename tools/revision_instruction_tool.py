# tools/revision_instruction_tool.py
# -*- coding: utf-8 -*-
"""Generate layout-revision instructions under engineering constraints.

This tool is intentionally not a deterministic optimizer. It computes a small
set of interpretable engineering indicators and converts them into a structured
instruction context, so that the downstream LLM can perform engineer-like
trade-off reasoning between obstacle avoidance, span standardization, bridge-type
selection, and economic rationality.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import tool


Number = float


class DirectFeedbackGenerator:
    """Generate revision instructions from collision report and layout result."""

    STANDARD_SPANS = {20.0, 25.0, 30.0, 35.0, 40.0}
    T_GIRDER_MAX = 40.0
    STANDARD_TOL = 0.25

    def __init__(self, report_data: Any, layout_result: Optional[Dict[str, Any]] = None):
        self.report_data = report_data
        self.report = self._normalize_report(report_data)
        self.layout_result = self._extract_layout_payload(layout_result or {})
        self.collision_metrics_from_report = self._extract_metrics(report_data)
        self.collisions = self._extract_collisions(self.report)
        self.global_collision_metrics = self._calculate_collision_metrics()
        self.span_metrics = self._calculate_span_metrics()

    # ============================================================
    # Data normalization
    # ============================================================
    @staticmethod
    def _normalize_report(report_data: Any) -> List[Dict[str, Any]]:
        """Support collision_report as list or nested dict."""
        if isinstance(report_data, list):
            return report_data

        if isinstance(report_data, dict):
            if isinstance(report_data.get("collision_items"), list):
                return report_data["collision_items"]
            if isinstance(report_data.get("verification_result"), dict):
                vr = report_data["verification_result"]
                if isinstance(vr.get("collision_items"), list):
                    return vr["collision_items"]
            if isinstance(report_data.get("collision_result"), dict):
                cr = report_data["collision_result"]
                if isinstance(cr.get("collision_items"), list):
                    return cr["collision_items"]
        return []

    @staticmethod
    def _extract_metrics(report_data: Any) -> Dict[str, Any]:
        if not isinstance(report_data, dict):
            return {}
        for key in ["metrics", "collision_metrics"]:
            if isinstance(report_data.get(key), dict):
                return report_data[key]
        if isinstance(report_data.get("verification_result"), dict):
            vr = report_data["verification_result"]
            if isinstance(vr.get("metrics"), dict):
                return vr["metrics"]
        if isinstance(report_data.get("collision_result"), dict):
            cr = report_data["collision_result"]
            if isinstance(cr.get("metrics"), dict):
                return cr["metrics"]
        return {}

    @staticmethod
    def _extract_layout_payload(layout_result: Any) -> Dict[str, Any]:
        if not isinstance(layout_result, dict):
            return {}
        # Compatible with tool wrappers.
        for key in ["layout_result", "design_result", "result", "output"]:
            if isinstance(layout_result.get(key), dict):
                inner = layout_result[key]
                if "设桥总览" in inner or "桥梁列表" in inner:
                    return inner
        return layout_result

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                if math.isnan(float(value)):
                    return default
            except Exception:
                pass
            return float(value)
        text = str(value).strip()
        if not text or text.lower() in {"none", "null", "nan", "-"}:
            return default
        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        return float(nums[0]) if nums else default

    def _extract_collisions(self, report: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        collisions: List[Dict[str, Any]] = []
        for item in report:
            if not isinstance(item, dict):
                continue
            res = item.get("res", {}) or {}
            if res.get("status") != "COLLISION":
                continue

            raw_id = str(item.get("pier_id", ""))
            digits = "".join(filter(str.isdigit, raw_id))
            if not digits:
                continue

            k_value = self._safe_float(item.get("k_val"), default=0.0)
            station_label = item.get("station_label") or item.get("station") or self._format_station(k_value)
            is_abutment = "桥台" in raw_id or raw_id == "0" or raw_id.startswith("0 ")
            collisions.append(
                {
                    "id": int(digits),
                    "raw_id": raw_id,
                    "is_abutment": is_abutment,
                    "k": k_value,
                    "depth": self._safe_float(res.get("intrusion_depth_m"), default=0.0),
                    "overlap": self._safe_float(res.get("overlap_ratio"), default=0.0),
                    "bridge_id": item.get("bridge_id"),
                    "side_label": item.get("side_label"),
                    "col_name": item.get("col_name"),
                    "station_label": station_label,
                }
            )
        return collisions

    # ============================================================
    # Collision metrics and clusters
    # ============================================================
    def _calculate_collision_metrics(self) -> Dict[str, Any]:
        """Keep only the original five collision indicators."""
        if self.collision_metrics_from_report:
            metrics = self.collision_metrics_from_report
            # Preserve names used by the existing collision tool.
            conflict_count = int(self._safe_float(metrics.get("conflict_column_count"), default=len(self.collisions)))
            return {
                "conflict_column_count": conflict_count,
                "conflict_column_rate": self._safe_float(metrics.get("conflict_column_rate"), default=0.0),
                "max_intrusion_depth": self._safe_float(
                    metrics.get("max_intrusion_depth") or metrics.get("max_intrusion_depth_columns"),
                    default=max([c["depth"] for c in self.collisions], default=0.0),
                ),
                "avg_intrusion_depth": self._safe_float(
                    metrics.get("avg_intrusion_depth") or metrics.get("avg_intrusion_depth_columns"),
                    default=0.0,
                ),
                "avg_overlap_ratio": self._safe_float(metrics.get("avg_overlap_ratio") or metrics.get("avg_overlap_ratio_columns"), default=0.0),
            }

        depths = [c["depth"] for c in self.collisions]
        overlaps = [c["overlap"] for c in self.collisions]
        return {
            "conflict_column_count": len(self.collisions),
            "conflict_column_rate": 0.0,
            "max_intrusion_depth": max(depths) if depths else 0.0,
            "avg_intrusion_depth": sum(depths) / len(depths) if depths else 0.0,
            "avg_overlap_ratio": sum(overlaps) / len(overlaps) if overlaps else 0.0,
        }

    def _cluster_collisions(self) -> List[List[Dict[str, Any]]]:
        if not self.collisions:
            return []
        sorted_cols = sorted(
            self.collisions,
            key=lambda x: (str(x.get("bridge_id")), str(x.get("side_label")), x["k"]),
        )
        clusters: List[List[Dict[str, Any]]] = []
        current = [sorted_cols[0]]
        gap_threshold = 45.0
        for curr in sorted_cols[1:]:
            prev = current[-1]
            same_bridge = str(curr.get("bridge_id")) == str(prev.get("bridge_id"))
            same_side = str(curr.get("side_label")) == str(prev.get("side_label"))
            close_enough = (curr["k"] - prev["k"]) < gap_threshold
            if same_bridge and same_side and close_enough:
                current.append(curr)
            else:
                clusters.append(current)
                current = [curr]
        clusters.append(current)
        clusters.sort(
            key=lambda c: (
                len(set(p["raw_id"] for p in c)),
                max(p["depth"] for p in c),
                max(p["overlap"] for p in c),
            ),
            reverse=True,
        )
        return clusters

    # ============================================================
    # Span parsing and standard-span indicator
    # ============================================================
    @staticmethod
    def _format_station(k: float) -> str:
        km = int(k // 1000)
        m = k - km * 1000
        return f"K{km}+{m:05.1f}"

    @classmethod
    def _is_standard_span(cls, value: float) -> bool:
        return any(abs(value - s) <= cls.STANDARD_TOL for s in cls.STANDARD_SPANS)

    @staticmethod
    def _parse_span_combo(span_expr: Any) -> List[float]:
        """Parse span expressions such as '5×30 + (40+30) + 9×30'."""
        if span_expr is None:
            return []
        text = str(span_expr)
        text = text.replace("×", "x").replace("X", "x").replace("*", "x")
        text = text.replace("＋", "+").replace("（", "(").replace("）", ")")
        text = text.replace(" ", "")

        spans: List[float] = []

        def expand_multiplier(match: re.Match) -> str:
            count = int(match.group(1))
            value = float(match.group(2))
            spans.extend([value] * count)
            return "+"

        # Expand n x L first and remove those terms from the expression.
        text_without_multi = re.sub(r"(\d+)x(\d+(?:\.\d+)?)", expand_multiplier, text)
        # Remaining numeric terms are treated as single spans.
        for num in re.findall(r"\d+(?:\.\d+)?", text_without_multi):
            spans.append(float(num))
        return spans

    def _walk_bridge_items(self, obj: Any, out: List[Dict[str, Any]]) -> None:
        if isinstance(obj, dict):
            has_span = "跨径组合" in obj or "span_combo" in obj or "span_combination" in obj
            has_type = "桥型" in obj or "bridge_type" in obj
            if has_span or has_type:
                out.append(obj)
            for v in obj.values():
                self._walk_bridge_items(v, out)
        elif isinstance(obj, list):
            for v in obj:
                self._walk_bridge_items(v, out)

    def _extract_bridge_items(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        self._walk_bridge_items(self.layout_result, items)
        # De-duplicate by object id / span/type signature.
        dedup: List[Dict[str, Any]] = []
        seen = set()
        for item in items:
            sig = (str(item.get("桥梁编号") or item.get("bridge_id") or ""), str(item.get("跨径组合") or item.get("span_combo") or ""), str(item.get("桥型") or item.get("bridge_type") or ""))
            if sig in seen:
                continue
            seen.add(sig)
            dedup.append(item)
        return dedup

    def _calculate_span_metrics(self) -> Dict[str, Any]:
        bridge_items = self._extract_bridge_items()
        span_values: List[float] = []
        t_girder_span_values: List[float] = []
        bridge_summaries: List[Dict[str, Any]] = []

        for item in bridge_items:
            span_expr = item.get("跨径组合") or item.get("span_combo") or item.get("span_combination")
            bridge_type = str(item.get("桥型") or item.get("bridge_type") or "")
            spans = self._parse_span_combo(span_expr)
            if not spans:
                continue
            span_values.extend(spans)
            if "T梁" in bridge_type or "T 梁" in bridge_type or "T-girder" in bridge_type:
                t_girder_span_values.extend(spans)
            bridge_summaries.append(
                {
                    "bridge_id": item.get("桥梁编号") or item.get("bridge_id"),
                    "bridge_type": bridge_type,
                    "span_combo": span_expr,
                    "span_count": len(spans),
                    "total_length": round(sum(spans), 3),
                }
            )

        # For mixed bridge types, standard_span_rate is still reported as an overall
        # layout standardization reference. The prompt explains that special bridge
        # segments are allowed when they are structurally necessary.
        total_length = sum(span_values)
        standard_length = sum(s for s in span_values if self._is_standard_span(s))
        standard_span_rate = standard_length / total_length if total_length > 0 else None

        t_total = sum(t_girder_span_values)
        t_standard = sum(s for s in t_girder_span_values if self._is_standard_span(s))
        t_standard_rate = t_standard / t_total if t_total > 0 else None

        return {
            "standard_span_rate": standard_span_rate,
            "standard_length": round(standard_length, 3),
            "total_length": round(total_length, 3),
            "t_girder_standard_span_rate": t_standard_rate,
            "t_girder_standard_length": round(t_standard, 3),
            "t_girder_total_length": round(t_total, 3),
            "standard_spans": sorted(self.STANDARD_SPANS),
            "bridge_summaries": bridge_summaries,
        }

    # ============================================================
    # Strategy recommendation
    # ============================================================
    def _standard_rate_value(self) -> float:
        rate = self.span_metrics.get("t_girder_standard_span_rate")
        if rate is None:
            rate = self.span_metrics.get("standard_span_rate")
        return float(rate) if rate is not None else 1.0

    def _classify_collision_severity(self, cluster: Optional[List[Dict[str, Any]]] = None) -> str:
        metrics = self.global_collision_metrics
        count = int(metrics["conflict_column_count"])
        rate = float(metrics["conflict_column_rate"])
        max_depth = float(metrics["max_intrusion_depth"])
        avg_depth = float(metrics["avg_intrusion_depth"])
        avg_overlap = float(metrics["avg_overlap_ratio"])

        if cluster:
            cluster_count = len(set(c["raw_id"] for c in cluster))
            cluster_span = max(c["k"] for c in cluster) - min(c["k"] for c in cluster)
            cluster_max_depth = max(c["depth"] for c in cluster)
            cluster_avg_overlap = sum(c["overlap"] for c in cluster) / len(cluster)
            if cluster_count <= 1 and cluster_span < 20 and cluster_max_depth < 1.5 and cluster_avg_overlap < 0.5:
                return "minor"
            if cluster_count >= 3 or cluster_span > 40 or cluster_max_depth > 2.5 or cluster_avg_overlap > 0.65:
                return "major"

        if count <= 1 and max_depth < 1.5 and avg_overlap < 0.5:
            return "minor"
        if rate > 0.10 or count >= 6 or avg_depth > 2.0 or avg_overlap > 0.65:
            return "major"
        return "moderate"

    def _recommend_strategy(self, cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
        unique_piers = sorted(set(str(p["raw_id"]) for p in cluster))
        is_abutment_involved = any(p["is_abutment"] for p in cluster)

        if is_abutment_involved:
            return {
                "necessity_level": "manual_review_or_boundary_adjustment",
                "recommended_strategy": "策略D：人工复核/桥梁边界调整",
                "reason": "冲突涉及桥台，调整可能改变桥梁起终点和桥跨范围，应优先复核桥梁边界；若边界调整影响路线或桥长，应转人工复核。",
                "required_clear_span": None,
                "allowed_spans": sorted(self.STANDARD_SPANS),
            }

        if len(unique_piers) == 1:
            return {
                "necessity_level": "minor",
                "recommended_strategy": "策略A：标准跨径局部重组",
                "reason": "仅有单个普通墩位冲突，可在保持原桥型的前提下重组相邻标准跨径并移动墩位。碰撞点不能作为连续障碍物边界。",
                "required_clear_span": None,
                "allowed_spans": sorted(self.STANDARD_SPANS),
            }

        return {
            "necessity_level": "manual_review",
            "recommended_strategy": "策略D：人工复核",
            "reason": "同一桥位同一幅存在多个连续墩位碰撞，但报告没有提供可验证的连续障碍物边界。碰撞点不能作为连续障碍物边界，也不能据其纵向包络推导净跨径，应转人工复核。",
            "required_clear_span": None,
            "allowed_spans": sorted(self.STANDARD_SPANS),
        }

    # ============================================================
    # Prompt generation
    # ============================================================
    @staticmethod
    def _pct(value: Optional[float]) -> str:
        if value is None:
            return "未知"
        return f"{value * 100:.1f}%"

    def _format_global_metrics_block(self) -> str:
        m = self.global_collision_metrics
        s = self.span_metrics
        return (
            "### 1. 指标摘要\n"
            "#### 1.1 碰撞指标\n"
            f"- conflict_column_count: {m['conflict_column_count']}\n"
            f"- conflict_column_rate: {m['conflict_column_rate']:.4f}\n"
            f"- max_intrusion_depth: {m['max_intrusion_depth']:.3f} m\n"
            f"- avg_intrusion_depth: {m['avg_intrusion_depth']:.3f} m\n"
            f"- avg_overlap_ratio: {m['avg_overlap_ratio']:.3f}\n\n"
            "#### 1.2 跨径结构合理性指标\n"
            f"- standard_span_rate: {self._pct(s.get('standard_span_rate'))}\n"
            f"- t_girder_standard_span_rate: {self._pct(s.get('t_girder_standard_span_rate'))}\n"
            f"- 标准T梁跨径库: {', '.join(f'{x:g}m' for x in s.get('standard_spans', []))}\n"
            "- 指标解释: standard_span_rate用于约束修正方案的跨径标准化程度。仅凭墩柱碰撞点不得新增大跨结构体系；已有且具备明确设计依据的特殊跨径应保持原结构体系。\n"
        )

    def _format_strategy_principles(self) -> str:
        return (
            "### 2. 工程权衡原则\n"
            "以下原则的目的不是机械限制LLM，而是为LLM提供接近工程师决策时所需的利弊权衡依据。"
            "请在修正时综合判断避障必要性、跨径标准化、桥型经济性和局部结构适用性。\n"
            "1. 小范围、轻微碰撞不宜引入大跨或改变桥型，应优先采用局部墩位微调。\n"
            "2. 当标准T梁跨径组合能够解决冲突时，应优先在20m、25m、30m、35m、40m标准跨内重组，避免31.5m、48.5m、50m、75m等非标准T梁跨径。\n"
            "3. 碰撞墩柱的纵向分布不等同于连续障碍物区间，不得据此计算净跨需求或新增连续梁、刚构等大跨结构。\n"
            "4. 多个墩位连续冲突或报告信息不足时，应输出人工复核建议，不得为了通过碰撞检测而生成工程上明显不合理的跨径组合。\n"
            "5. 修正结果必须实际更新跨径组合、总跨数、桥梁总长和墩台桩号表，不得只修改说明文字。\n"
        )

    def _format_cluster_directive(self, idx: int, cluster: List[Dict[str, Any]]) -> str:
        start_k = min(p["k"] for p in cluster)
        end_k = max(p["k"] for p in cluster)
        unique_piers = sorted(set(str(p["raw_id"]) for p in cluster))
        bridge_ids = sorted(set(str(p.get("bridge_id")) for p in cluster))
        side_labels = sorted(set(str(p.get("side_label") or "未知幅别") for p in cluster))
        max_depth = max(p["depth"] for p in cluster)
        avg_overlap = sum(p["overlap"] for p in cluster) / len(cluster)
        cluster_span = end_k - start_k
        strategy = self._recommend_strategy(cluster)

        lines = [
            f"### 冲突区域 {idx}: {self._format_station(start_k)} ~ {self._format_station(end_k)}",
            f"- 涉及桥梁: {', '.join(bridge_ids) if bridge_ids else '未知'}。",
            f"- 涉及幅别: {', '.join(side_labels)}。",
            f"- 涉及墩台: {', '.join(unique_piers)}。",
            f"- 局部冲突范围: {cluster_span:.1f}m；最大侵入深度: {max_depth:.2f}m；平均重叠比例: {avg_overlap:.2f}。",
            f"- 修正必要程度: {strategy['necessity_level']}。",
            f"- 推荐策略: {strategy['recommended_strategy']}。",
            f"- 策略理由: {strategy['reason']}",
            "- 具体修正要求:",
        ]

        rec = strategy["recommended_strategy"]
        if rec.startswith("策略A"):
            lines += [
                "  1. 保持桥梁总体范围、桥型和主要标准跨径体系基本不变。",
                "  2. 通过调整冲突墩前后相邻跨径，将冲突墩位沿纵向移动至障碍物安全侧。",
                "  3. 相邻跨径应限定在20m、25m、30m、35m、40m标准跨内，不得为单墩轻微冲突引入大跨桥型。",
            ]
        else:
            lines += [
                "  1. 不建议继续强行通过零散跨径调整消除冲突。",
                "  2. 应输出需要人工复核的原因，重点说明避障要求、跨径标准化和桥型经济性之间的矛盾。",
                "  3. 在缺少连续障碍物边界和结构论证时，不得新增超过40m跨径或特殊桥型。",
            ]

        return "\n".join(lines) + "\n"

    def generate_prompt(self) -> str:
        clusters = self._cluster_collisions()
        if not clusters:
            return "[OK] 校验通过：当前设桥布跨方案未检测到墩柱与障碍物碰撞。"

        parts = [
            "## 工程约束下的设桥布跨修正指令",
            "检测到当前布跨方案存在墩柱/桥台与障碍物碰撞。请注意，本轮修正不应只追求消除碰撞，而应充分利用LLM的工程推理能力，像桥梁工程师一样在避障必要性、跨径标准化、桥型经济性之间进行利弊权衡。",
            "",
            self._format_global_metrics_block(),
            self._format_strategy_principles(),
            "### 3. 分区修正策略与指令",
        ]
        for i, cluster in enumerate(clusters, start=1):
            parts.append(self._format_cluster_directive(i, cluster))

        parts.append(
            "---\n"
            "### 4. 输出约束\n"
            "1. 修正结果必须输出完整、可解析、可复检的设桥布跨JSON。\n"
            "2. 跨径组合、总跨数、桥梁总长和墩台桩号必须逐跨一致。\n"
            "3. 本轮不得仅凭碰撞点新增超过40m跨径、连续梁或刚构；已有特殊跨径只允许在原设计依据下保留。\n"
            "4. 未受碰撞影响的区间应保持原桥型和标准跨径组合。\n"
            "5. 若判断继续自动修正会导致方案明显不经济或不合理，应明确输出人工复核建议。\n"
        )
        return "\n".join(parts)

    def to_metrics_payload(self) -> Dict[str, Any]:
        clusters = self._cluster_collisions()
        return {
            "collision_metrics": self.global_collision_metrics,
            "span_metrics": self.span_metrics,
            "cluster_strategy_summary": [
                {
                    "cluster_index": i + 1,
                    "start_station": self._format_station(min(p["k"] for p in c)),
                    "end_station": self._format_station(max(p["k"] for p in c)),
                    "pier_count": len(set(p["raw_id"] for p in c)),
                    "bridge_id": c[0].get("bridge_id"),
                    "side_label": c[0].get("side_label"),
                    "strategy": self._recommend_strategy(c),
                }
                for i, c in enumerate(clusters)
            ],
        }


@tool
def generate_revision_instruction(
    collision_report_path: Optional[str] = None,
    collision_report: Optional[Dict[str, Any]] = None,
    layout_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """根据碰撞检测报告与当前布跨结果生成设桥布跨修正指令。"""
    try:
        if collision_report_path:
            path = Path(collision_report_path)
            if not path.exists():
                raise FileNotFoundError(f"碰撞检测报告不存在: {path}")
            with open(path, "r", encoding="utf-8") as f:
                report_data = json.load(f)
        elif collision_report is not None:
            report_data = collision_report
        else:
            raise ValueError("必须提供 collision_report_path 或 collision_report。")

        generator = DirectFeedbackGenerator(report_data, layout_result=layout_result)
        instruction_text = generator.generate_prompt()

        return {
            "success": True,
            "tool_name": "generate_revision_instruction",
            "revision_instruction": instruction_text,
            "engineering_metrics": generator.to_metrics_payload(),
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "tool_name": "generate_revision_instruction",
            "revision_instruction": None,
            "engineering_metrics": None,
            "error": str(e),
        }
