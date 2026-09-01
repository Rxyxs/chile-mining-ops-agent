"""Tool registry: maps tool names to callables and exposes their Anthropic
tool-use JSON schemas."""
from __future__ import annotations

from .anomaly_check_tool import TOOL_SCHEMAS as _ANOMALY_SCHEMAS
from .anomaly_check_tool import check_maintenance_anomalies
from .credit_risk_tool import TOOL_SCHEMAS as _CREDIT_SCHEMAS
from .credit_risk_tool import score_credit_risk
from .warehouse_query_tool import TOOL_SCHEMAS as _WAREHOUSE_SCHEMAS
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

TOOL_SCHEMAS = [*_WAREHOUSE_SCHEMAS, *_CREDIT_SCHEMAS, *_ANOMALY_SCHEMAS]

__all__ = ["TOOL_REGISTRY", "TOOL_SCHEMAS"]
