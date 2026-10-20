"""100 evaluation queries for the agent, each with the tools and arguments a correct
run must use. Spanish, the way an operator in Chile would ask.

Why 100 and not a handful: with 7 queries a "6/7 correct" is compatible with anything
from about 50% to 97%; 100 queries bring the 95% interval down to roughly +-10 points.

| Group | n | Expected |
|---|---:|---|
| flotation | 25 | `get_flotation_summary` with the month asked (5 ask for a month with no data) |
| alerts | 12 | `get_maintenance_alerts` with the window asked |
| procurement | 25 | `get_procurement_summary` with the filters asked |
| anomalies | 10 | `check_maintenance_anomalies` |
| multi-tool | 16 | two tools, both with the right arguments |
| out of scope | 12 | no tool at all |
"""
from __future__ import annotations

from dataclasses import dataclass, field

MONTH_NAMES = {
    "2017-03": "marzo de 2017", "2017-04": "abril de 2017", "2017-05": "mayo de 2017",
    "2017-06": "junio de 2017", "2017-07": "julio de 2017", "2017-08": "agosto de 2017",
    "2017-09": "septiembre de 2017",
}
EMPTY_MONTHS = {"2016-12": "diciembre de 2016", "2018-01": "enero de 2018", "2025-09": "septiembre de 2025",
                "2017-11": "noviembre de 2017", "2017-02": "febrero de 2017"}
STATUS_ES = {"delivered": "entregadas", "accepted": "aceptadas", "pending": "pendientes", "cancelled": "canceladas"}
CATEGORY_ES = {"mining_machinery": "maquinaria minera", "mining_services": "servicios de perforación minera",
               "industrial_chemicals": "productos químicos industriales",
               "fuels_lubricants": "combustibles y lubricantes"}


@dataclass(frozen=True)
class EvalQuery:
    query: str
    expected_tools: tuple[str, ...]
    note: str = ""
    # tool -> arguments a correct call must carry (other arguments are free).
    expected_args: dict = field(default_factory=dict)


def _flotation() -> list[EvalQuery]:
    templates = [
        "¿Cuál fue el resumen de la planta de flotación en {m}?",
        "¿Cómo estuvo la sílice en el concentrado durante {m}?",
        "Dame la ley de hierro del concentrado de {m}.",
        "¿Cuántas horas con sílice alta tuvo la planta en {m}?",
    ]
    out = []
    months = list(MONTH_NAMES.items())
    for i in range(20):
        key, name = months[i % len(months)]
        out.append(EvalQuery(templates[i % len(templates)].format(m=name), ("get_flotation_summary",),
                             expected_args={"get_flotation_summary": {"month": key}}))
    for key, name in EMPTY_MONTHS.items():
        out.append(EvalQuery(f"¿Cuál fue el resumen de flotación de {name}?", ("get_flotation_summary",),
                             note="month with no plant data", expected_args={"get_flotation_summary": {"month": key}}))
    return out


def _alerts() -> list[EvalQuery]:
    phrasing = [
        "¿Hubo alertas de equipos en los últimos {d} días?",
        "Muéstrame los eventos de instrumentos de los últimos {d} días, los más graves primero.",
        "¿Qué sensores fallaron en los últimos {d} días?",
    ]
    days = [7, 14, 30, 60]
    return [EvalQuery(phrasing[i % 3].format(d=days[i % 4]), ("get_maintenance_alerts",),
                      expected_args={"get_maintenance_alerts": {"days": days[i % 4]}}) for i in range(12)]


def _procurement() -> list[EvalQuery]:
    out = [EvalQuery("Dame el resumen de todas las compras públicas mineras.", ("get_procurement_summary",),
                     expected_args={"get_procurement_summary": {}}),
           EvalQuery("¿Cuál es el monto mediano de una orden de compra minera en junio de 2017?",
                     ("get_procurement_summary",), expected_args={"get_procurement_summary": {"month": "2017-06"}})]
    for status, es in STATUS_ES.items():
        out.append(EvalQuery(f"¿Cuántas órdenes de compra están {es} y por cuánto monto?",
                             ("get_procurement_summary",), expected_args={"get_procurement_summary": {"status": status}}))
    for category, es in CATEGORY_ES.items():
        out.append(EvalQuery(f"¿Cuánto se compró en {es}?", ("get_procurement_summary",),
                             expected_args={"get_procurement_summary": {"category": category}}))
    combos = [(s, c) for s in STATUS_ES for c in CATEGORY_ES][:8]
    for status, category in combos:
        out.append(EvalQuery(f"Resumen de órdenes {STATUS_ES[status]} de {CATEGORY_ES[category]}.",
                             ("get_procurement_summary",),
                             expected_args={"get_procurement_summary": {"status": status, "category": category}}))
    for i, (month, name) in enumerate(list(MONTH_NAMES.items())[:8]):
        category = list(CATEGORY_ES)[i % 4]
        out.append(EvalQuery(f"¿Cuánto se gastó en {CATEGORY_ES[category]} en {name}?", ("get_procurement_summary",),
                             expected_args={"get_procurement_summary": {"category": category, "month": month}}))
    return out


def _anomalies() -> list[EvalQuery]:
    phrasing = [
        "¿Algún equipo muestra un patrón de eventos anómalo en los últimos {d} días?",
        "Corre la detección de anomalías sobre los equipos de los últimos {d} días.",
    ]
    days = [30, 60, 90, 120, 200]
    return [EvalQuery(phrasing[i % 2].format(d=days[i % 5]), ("check_maintenance_anomalies",),
                      expected_args={"check_maintenance_anomalies": {"days": days[i % 5]}}) for i in range(10)]


def _multi() -> list[EvalQuery]:
    out = []
    months = list(MONTH_NAMES.items())
    for i in range(8):
        key, name = months[i % len(months)]
        d = [7, 30][i % 2]
        out.append(EvalQuery(f"Dame el resumen de flotación de {name} y dime si hubo alertas de equipos en los últimos {d} días.",
                             ("get_flotation_summary", "get_maintenance_alerts"), note="multi-tool",
                             expected_args={"get_flotation_summary": {"month": key}, "get_maintenance_alerts": {"days": d}}))
    for i in range(8):
        key, name = months[i % len(months)]
        category = list(CATEGORY_ES)[i % 4]
        out.append(EvalQuery(f"Compara la sílice del concentrado de {name} con lo comprado en {CATEGORY_ES[category]} ese mes.",
                             ("get_flotation_summary", "get_procurement_summary"), note="multi-tool",
                             expected_args={"get_flotation_summary": {"month": key},
                                            "get_procurement_summary": {"category": category, "month": key}}))
    return out


def _out_of_scope() -> list[EvalQuery]:
    questions = [
        "¿Cuál es la capital de Perú?", "¿Cuánto es 17 por 23?", "Escribe un haiku sobre el cobre.",
        "¿Quién ganó el mundial de 2010?", "Traduce 'flotation cell' al español.",
        "¿Qué es la flotación inversa catiónica?", "¿Qué día de la semana es hoy?",
        "Recomiéndame un libro sobre minería chilena.", "¿Cuál es el precio actual del cobre?",
        "Explica qué es un Isolation Forest.", "¿Cuántos habitantes tiene Antofagasta?",
        "Dame una receta de empanadas.",
    ]
    return [EvalQuery(q, (), note="out of scope") for q in questions]


def build_eval_queries() -> tuple[EvalQuery, ...]:
    queries = _flotation() + _alerts() + _procurement() + _anomalies() + _multi() + _out_of_scope()
    assert len(queries) == 100, len(queries)
    assert len({q.query for q in queries}) == 100, "duplicate queries"
    return tuple(queries)


EVAL_QUERIES = build_eval_queries()
