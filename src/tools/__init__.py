"""Tool registry: maps tool names to callables and wraps their JSON schemas
into the function-tool envelope the chat-completions API expects.

Each tool module declares a provider-neutral `TOOL_DEFINITIONS` entry
(`{"name", "description", "parameters"}`) -- plain JSON Schema, nothing
wire-specific. This module is the single place that knows the envelope the
API wants, so swapping providers means editing `_as_function_tool` rather
than every tool file.
"""
from __future__ import annotations

from .anomaly_check_tool import TOOL_DEFINITIONS as _ANOMALY_DEFINITIONS
from .anomaly_check_tool import check_maintenance_anomalies
from .credit_risk_tool import TOOL_DEFINITIONS as _CREDIT_DEFINITIONS
from .credit_risk_tool import score_credit_risk
from .warehouse_query_tool import TOOL_DEFINITIONS as _WAREHOUSE_DEFINITIONS
from .warehouse_query_tool import (
    get_flotation_summary,
    get_maintenance_alerts,
    get_procurement_summary,
)

TOOL_REGISTRY = {
    "get_flotation_summary": get_flotation_summary,
    "get_maintenance_alerts": get_maintenance_alerts,
    "get_procurement_summary": get_procurement_summary,
    "score_credit_risk": score_credit_risk,
    "check_maintenance_anomalies": check_maintenance_anomalies,
}

TOOL_DEFINITIONS = [
    *_WAREHOUSE_DEFINITIONS,
    *_CREDIT_DEFINITIONS,
    *_ANOMALY_DEFINITIONS,
]


def _as_function_tool(definition: dict) -> dict:
    return {"type": "function", "function": definition}


TOOL_SCHEMAS = [_as_function_tool(d) for d in TOOL_DEFINITIONS]

__all__ = ["TOOL_REGISTRY", "TOOL_DEFINITIONS", "TOOL_SCHEMAS"]
