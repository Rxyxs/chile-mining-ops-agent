# Chile Mining Ops Agent

*[English version: README.md](README.md)*

## Resumen

Las operaciones mineras y de riesgo en Chile suelen consultarse a través de dashboards fijos: una pantalla para KPIs de flotación, otra para alertas de mantenimiento, otra para un modelo de riesgo crediticio. Este proyecto explora una interfaz distinta: un agente en lenguaje natural que responde preguntas operacionales y de riesgo **llamando herramientas reales** (funciones Python que consultan una base de datos real y un modelo real ya entrenado), sin inventar números.

Es el proyecto #37 de un portafolio construido mayormente sobre modelos tabulares/series de tiempo (baseline + ensamble + PyTorch, comparados entre sí). Ninguno de esos proyectos ejercita tool-calling real con un LLM — este cierra ese hueco.

Todo lo que las herramientas tocan — los datos operacionales y el modelo de riesgo crediticio — es **sintético y generado dentro de este mismo repo**, con semilla fija. No se copia ni se importa nada de otros repositorios del portafolio; la arquitectura se inspira en patrones usados en otros lados (ej. un warehouse DuckDB, un modelo de regresión logística de riesgo) pero los artefactos son autocontenidos y se construyen desde cero aquí.

## Arquitectura

```mermaid
flowchart LR
    U[Pregunta del usuario] --> A[Loop de MiningOpsAgent]
    A -->|solicita tool_use| D{Dispatcher}
    D --> T1[warehouse_query_tool<br/>DuckDB]
    D --> T2[credit_risk_tool<br/>LogisticRegression]
    D --> T3[anomaly_check_tool<br/>IsolationForest]
    T1 --> R[tool_result]
    T2 --> R
    T3 --> R
    R --> A
    A -->|end_turn| F[Respuesta final en texto]
```

El loop vive en `src/agent.py` (`MiningOpsAgent`). Envía el mensaje del usuario y los schemas de las tools al modelo; si la respuesta pide `tool_use`, despacha a la función Python correspondiente, empaqueta el resultado (o la excepción, si la tool falló) como `tool_result`, y lo devuelve — con un tope de iteraciones (5 por defecto) para que un modelo que se comporte mal no entre en loop infinito.

## Herramientas expuestas (`src/tools/`)

Las tres son funciones Python planas y síncronas, con un schema JSON asociado (`TOOL_SCHEMAS`) en el formato que espera la API de Anthropic para tool-use (`{"name", "description", "input_schema"}`). Cada una corre, y está testeada, **sin necesitar ningún LLM ni API key**.

| Tool | Archivo | Qué hace |
|---|---|---|
| `get_flotation_summary`, `get_maintenance_alerts`, `get_procurement_summary` | `warehouse_query_tool.py` | Consultas fijas, nombradas y parametrizadas contra un DuckDB local (`data/ops.duckdb`) — **no** SQL arbitrario del usuario, por diseño. |
| `score_credit_risk` | `credit_risk_tool.py` | Puntúa el perfil de un solicitante de crédito con una `LogisticRegression` pequeña y autocontenida, entrenada sobre solicitantes sintéticos (`data/credit_risk_model.joblib`). |
| `check_maintenance_anomalies` | `anomaly_check_tool.py` | Corre un `IsolationForest` sobre eventos de mantenimiento recientes para marcar equipos con patrones anómalos de downtime/severidad. |

El warehouse DuckDB tiene tres tablas sintéticas: `flotation_batches`, `maintenance_events`, `procurement_orders`, todas generadas por `src/setup_data.py` con `numpy.random.default_rng(42)`.

## Resultados

Generados por `python -m src.generate_report` (`src/visualization/plots.py`), que llama a las mismas funciones tool que despacha el agente en tiempo de ejecución y grafica sus resultados reales — nada aquí está hardcodeado ni re-simulado por separado de las tools.

| Tool | Métrica | Valor |
|---|---|---|
| `score_credit_risk` | ROC-AUC (test held-out, n=400) | 0,586 |
| `score_credit_risk` | PR-AUC (tasa base 0,245) | 0,335 |
| `score_credit_risk` | Accuracy de test | 0,755 |
| `check_maintenance_anomalies` | Equipos marcados (ventana 60d) | 3 / 24 |
| `get_flotation_summary` | Meses de datos | 12 |

**Lectura honesta sobre el número de riesgo crediticio**: ROC-AUC 0,586 es débil — apenas por encima del baseline de 0,5 sin habilidad predictiva. Esto no se disimula: el generador de solicitantes sintéticos en `setup_data.py` mezcla una probabilidad de default latente bastante ruidosa con solo seis features y una `LogisticRegression` simple, así que un puntaje mediocre aquí es un reflejo preciso de un modelo de ejemplo deliberadamente simple, no un bug. El punto de este repo es la arquitectura de tool-calling, no exprimir AUC de un dataset de juguete.

![Evaluación de riesgo crediticio](reports/figures/credit_risk_evaluation.png)
![Scores de anomalía por equipo](reports/figures/anomaly_scores.png)
![Resumen del warehouse](reports/figures/warehouse_overview.png)

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Uso

1. **Generar los datos sintéticos y entrenar el modelo de riesgo:**

   ```bash
   python -m src.setup_data
   ```

   Escribe `data/ops.duckdb` y `data/credit_risk_model.joblib`.

2. **Generar las figuras de resultados y el resumen de métricas:**

   ```bash
   python -m src.generate_report
   ```

   Escribe `reports/figures/*.png` y `reports/metrics.json`.

3. **Correr la suite de tests** (funciona completamente offline, sin API key):

   ```bash
   pytest
   ```

4. **Hablar con el agente** (requiere una `ANTHROPIC_API_KEY` real):

   ```bash
   python -m src.cli "¿Cuál fue la recuperación de flotación en septiembre de 2025?"
   ```

   Sin una key configurada, el CLI termina con un mensaje claro en vez de un traceback.

## Nota honesta sobre lo verificado

En el entorno de build de este proyecto **no hay `ANTHROPIC_API_KEY` disponible**. Eso limita lo que efectivamente se pudo correr y confirmar aquí, y esta sección reporta solo lo que se ejecutó en esta sesión — nada se describe como funcionando si no se corrió de verdad.

**Verificado en esta sesión:**
- `python -m src.setup_data` corre exitosamente y escribe ambos artefactos (la precisión de holdout del modelo de riesgo se imprimió y se observó, no se asumió).
- `python -m src.generate_report` corre exitosamente y escribe las tres figuras más `reports/metrics.json` de arriba, a partir de resultados reales de las tools.
- `pytest` — **22/22 tests pasando**. Incluye ejecuciones reales de las cinco funciones tool contra el DuckDB y el modelo generados (sin mockear las tools mismas), las tres funciones de graficado (verificando que las figuras se escriben de verdad a disco con contenido real), el script de reporte, más tests del loop del agente que mockean el cliente de Anthropic (`unittest.mock`) para verificar que el dispatcher llama a la tool correcta con los argumentos correctos, arma bien los bloques `tool_result`, maneja excepciones de las tools sin crashear, y respeta el tope de iteraciones.
- Cada función tool también se invocó una vez manualmente fuera de pytest y devolvió un resultado real, inspeccionado.

**No verificado:**
- Una conversación real end-to-end contra la API real de Anthropic. `src/cli.py` está escrito para hacer esa llamada de verdad (`anthropic.Anthropic()`, modelo `claude-sonnet-5` por defecto), pero nunca se corrió en este entorno porque no hay API key disponible. Cualquiera que clone este repo con su propia `ANTHROPIC_API_KEY` puede correr `python -m src.cli "..."` para probarlo en vivo — ese camino está implementado y testeado a nivel de dispatch, solo que no se ejercitó contra la API real aquí.

## Decisión de diseño: versionar los datos generados

`data/ops.duckdb`, `data/credit_risk_model.joblib`, y `reports/figures/*.png` + `reports/metrics.json` se versionan en el repo en vez de ignorarse. Son pequeños, sintéticos/derivados, regenerables de forma determinística (`python -m src.setup_data && python -m src.generate_report`), y versionarlos permite que las tools (y los resultados de arriba) sean visibles inmediatamente después de clonar el repo sin un paso de setup obligatorio. El `.gitignore` sigue excluyendo el entorno virtual y las cachés.

## Autor

Pablo Reyes — Data Scientist, Santiago, Chile.
