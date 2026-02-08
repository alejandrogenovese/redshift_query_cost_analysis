# ⚡ Redshift Query Cost Analyzer

Herramienta de estimación **pre-ejecución** de costos monetarios de queries de AWS Redshift a partir del EXPLAIN plan, con sistema de calibración por regresión adaptado al cluster del usuario.

---

## ¿Por qué?

Las system tables de Redshift (`stl_query`, `svl_query_summary`) solo tienen datos **post-ejecución**. Cuando necesitás saber cuánto va a costar una query **antes** de correrla — para reviews de SQL, queries ad-hoc sobre tablas grandes, o planificación de presupuesto — el `EXPLAIN` plan es la única fuente de información disponible.

Esta herramienta toma el output de `EXPLAIN` y lo convierte en una estimación de tiempo y costo monetario, calibrable al hardware específico de tu cluster.

---

## Funcionalidades

- **Analizador**: Pegás un EXPLAIN plan y obtenés tiempo estimado + costo en USD
- **Calibración**: Wizard de 3 pasos para entrenar el modelo con datos reales de tu cluster
- **Comparación**: Compará múltiples queries y rankeá por costo
- **Historial**: Registro persistente de todas las estimaciones
- **Soporte dual**: Provisioned (nodos) y Serverless (RPU)
- **Indicador de confianza**: Badge que muestra qué tan confiable es la estimación

---

## Quick Start (Local)

```bash
# 1. Clonar
git clone https://github.com/TU_USER/redshift-cost-analyzer.git
cd redshift-cost-analyzer

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Iniciar
chmod +x start.sh
./start.sh

# 4. Abrir
# → http://localhost:5000
```

---

## Deploy en Render

### Opción A: Blueprint (automático)

1. Subir el proyecto a un repo de GitHub
2. En [render.com](https://render.com) → **New** → **Blueprint**
3. Conectar el repo — Render lee `render.yaml` y configura todo

### Opción B: Manual

1. En Render → **New** → **Web Service** → conectar repo
2. Configurar:

| Setting | Valor |
|---------|-------|
| **Runtime** | Python |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `cd backend && gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120` |

3. Environment variable (opcional):

| Variable | Valor |
|----------|-------|
| `CALIBRATION_DB` | `/opt/render/project/src/backend/calibration.db` |

### Nota sobre persistencia

El free tier de Render no tiene disco persistente — la base de calibración se resetea en cada deploy. Opciones:

- **Free tier**: Perfecto para demo y uso sin calibración (modelo default)
- **Starter ($7/mes)**: Disco persistente, la calibración se mantiene
- **PostgreSQL add-on ($7/mes)**: Migrando SQLite a PostgreSQL (requiere cambio en `calibration.py`)

---

## Arquitectura

```
redshift-cost-analyzer/
│
├── frontend/
│   └── index.html              ← React SPA (CDN, sin build step)
│
├── backend/
│   ├── app.py                  ← Flask API + servidor de estáticos
│   ├── parser.py               ← Parser enriquecido de EXPLAIN plan
│   ├── calibration.py          ← Sistema de calibración + regresión
│   └── calculator.py           ← Motor de cálculo de costos monetarios
│
├── tests.py                    ← 23 tests (pytest)
├── requirements.txt            ← flask, flask-cors, gunicorn
├── render.yaml                 ← Render Blueprint
├── start.sh                    ← Script de inicio local
└── .gitignore
```

### Flujo de datos

```
EXPLAIN text → Parser → ExplainPlan (nodes, cost, rows, width, distribution)
                            ↓
                    Calibration Model (α, β, γ, δ, ε)
                            ↓
                    estimated_seconds = exp(α·log(cost) + β·log(rows) + ...)
                            ↓
                    ClusterConfig (node_type, pricing, concurrency)
                            ↓
                    CostEstimate (USD, projecciones, confianza)
```

---

## Modelo de Estimación

### Fórmula de Tiempo

```
log(estimated_time) = α·log(cost) + β·log(rows) + γ·width + δ[operation] + ε[distribution] + intercept
```

Los coeficientes se calibran con datos reales del cluster. Sin calibración, se usa un modelo default conservador.

**¿Por qué logarítmico?** Evidencia empírica muestra que la relación entre cost units y tiempo real no es lineal. En un caso documentado, una reducción del 99.8% en cost units produjo una reducción del 61% en tiempo de ejecución. El modelo log-lineal captura mejor esta relación.

### Fórmula Monetaria

| Billing Model | Fórmula |
|--------------|---------|
| **Provisioned** | `costo = (est_hours × hourly_rate × nodes) / concurrency` |
| **Serverless** | `costo = est_hours × base_RPU × $0.375/RPU-hr` |
| **Spectrum** | `costo += (data_scanned_TB) × $5/TB` |

El factor `/ concurrency` es clave: si el cluster ejecuta en promedio 5 queries simultáneas, cada query consume ~1/5 del costo del cluster, no el 100%.

### Parser Enriquecido

Extrae del EXPLAIN:

| Campo | Ejemplo |
|-------|---------|
| Cost units | `cost=0.00..1652544172278.50` |
| Rows estimados | `rows=6938483` |
| Width | `width=179` |
| Operación | `Hash Join`, `Seq Scan`, `Sort`, `Nested Loop` |
| Distribución | `DS_DIST_NONE`, `DS_BCAST_INNER`, `DS_DIST_BOTH` |
| Tabla | `orders`, `customers` |
| Condiciones | `Hash Cond`, `Filter`, `Sort Key`, `Merge Cond` |

### Paralelización por Distribución

En vez de un factor fijo, la eficiencia de paralelización varía según la estrategia de distribución que reporta el EXPLAIN:

| Distribución | Eficiencia | Significado |
|-------------|-----------|-------------|
| `DS_DIST_NONE` | ~90% | No requiere redistribución (óptimo) |
| `DS_DIST_ALL_NONE` | ~85% | Tabla ALL, sin redistribución |
| `DS_BCAST_INNER` | ~65% | Broadcast de tabla interna |
| `DS_DIST_OUTER` | ~55% | Redistribución de tabla externa |
| `DS_DIST_BOTH` | ~45% | Ambas tablas redistribuidas (peor caso) |

---

## Sistema de Calibración

### ¿Cómo funciona?

El modelo default da estimaciones genéricas. Para obtener predicciones precisas para **tu** cluster, el wizard de calibración ajusta los coeficientes con datos reales.

### Wizard (3 pasos)

**Paso 1 — Recolectar datos**

Ejecutá 5-10 queries representativas de tu workload. Para cada una:

```sql
-- 1. Obtener EXPLAIN (sin ejecutar)
EXPLAIN SELECT ... FROM orders JOIN customers ON ...;

-- 2. Ejecutar y medir tiempo
SELECT ... FROM orders JOIN customers ON ...;
-- → Anotar tiempo de ejecución (o sacarlo de stl_query)
```

**Paso 2 — Cargar en el wizard**

En la pestaña "Calibración", pegá el EXPLAIN + el tiempo real para cada query.

**Paso 3 — Entrenar**

Click en "Entrenar Modelo". El sistema ajusta los coeficientes por regresión de mínimos cuadrados y muestra R², MAE, y los offsets por operación.

### Tips para buena calibración

- Incluí queries con distintas operaciones: Seq Scan, Hash Join, Sort, Aggregate
- Variá el tamaño: queries chicas (< 1s) y grandes (> 30s)
- Incluí queries con diferentes distribuciones si es posible
- Re-calibrá si cambiás el tipo de nodo o el número de nodos

### Indicador de Confianza

| Nivel | Condición | Significado |
|-------|-----------|-------------|
| 🟢 **Alta** | ≥5 puntos, R² > 0.7 | Estimaciones confiables |
| 🟡 **Media** | 3-4 puntos, o R² 0.4-0.7 | Estimaciones aproximadas |
| 🔴 **Baja** | Sin calibrar | Modelo default, usar como referencia relativa |

---

## Uso por Pestaña

### 🔍 Analizar

1. Configurar cluster (tipo de nodo, nodos, concurrencia, billing model)
2. Pegar EXPLAIN plan en el textarea
3. Click "Analizar Costo"
4. Ver: tiempo estimado, costo por ejecución, proyecciones diarias/mensuales/anuales, árbol del plan parseado

### 🎯 Calibración

1. Agregar puntos de benchmark (EXPLAIN + tiempo real)
2. Entrenar modelo (mínimo 3 puntos, recomendado 5+)
3. Verificar R² y coeficientes

### ⚖️ Comparar

1. Pegar 2 o más EXPLAIN plans
2. Click "Comparar"
3. Ver ranking de costos con barras de costo relativo

### 📜 Historial

Registro automático de todas las estimaciones realizadas.

---

## API Endpoints

### Análisis

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `POST` | `/api/analyze` | Analizar EXPLAIN → costo estimado |
| `POST` | `/api/compare` | Comparar múltiples EXPLAIN plans |
| `POST` | `/api/parse` | Solo parsear EXPLAIN (sin cálculo monetario) |

### Calibración

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/api/calibration/points` | Listar puntos de calibración |
| `POST` | `/api/calibration/add` | Agregar punto manualmente |
| `POST` | `/api/calibration/add-from-explain` | Agregar desde EXPLAIN + tiempo |
| `POST` | `/api/calibration/fit` | Entrenar modelo con puntos actuales |
| `GET` | `/api/calibration/model` | Obtener modelo actual |
| `POST` | `/api/calibration/clear` | Borrar toda la calibración |

### Referencia

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/api/pricing` | Precios de nodos, RPU, Spectrum |
| `GET` | `/api/reference-data` | Tipos de operación y distribución |
| `GET` | `/api/history` | Historial de estimaciones |
| `GET` | `/api/health` | Estado del servicio |

### Ejemplo: Analizar un EXPLAIN

```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "explain_text": "XN Seq Scan on orders (cost=0.00..69384.83 rows=6938483 width=108)",
    "node_type": "ra3.4xlarge",
    "num_nodes": 2,
    "avg_concurrency": 5,
    "billing_model": "provisioned",
    "executions_per_day": 10
  }'
```

Respuesta:

```json
{
  "estimate": {
    "estimated_seconds": 34532.40,
    "formatted_time": "9.59 horas",
    "formatted_cost": "$12.51",
    "total_cost": 12.51,
    "confidence": "low",
    "confidence_detail": "Modelo por defecto sin calibrar...",
    "monthly_cost": 3753.00,
    "primary_operation": "Seq Scan",
    "primary_distribution": null,
    "complexity_score": 1.0
  },
  "plan": {
    "nodes": [...],
    "root_cost": 69384.83,
    "root_rows": 6938483
  }
}
```

### Ejemplo: Calibración desde EXPLAIN

```bash
# 1. Agregar punto
curl -X POST http://localhost:5000/api/calibration/add-from-explain \
  -H "Content-Type: application/json" \
  -d '{
    "explain_text": "XN Seq Scan on orders (cost=0.00..69384.83 rows=6938483 width=108)",
    "actual_time_seconds": 4.32,
    "query_label": "Full scan de orders",
    "num_nodes": 2,
    "node_type": "ra3.4xlarge"
  }'

# 2. Entrenar modelo (después de agregar 3+ puntos)
curl -X POST http://localhost:5000/api/calibration/fit

# 3. Verificar modelo
curl http://localhost:5000/api/calibration/model
```

---

## Tests

```bash
# Correr toda la suite
python -m pytest tests.py -v

# 23 tests:
#   8 Parser       → multiline, distribución, complejidad, edge cases
#   7 Calibración  → modelo default, regresión, R², confianza
#   8 Calculator   → billing models, concurrencia, reservas, comparación
```

---

## Stack Técnico

| Componente | Tecnología |
|-----------|-----------|
| **Frontend** | React 18 (CDN, sin build) + Tailwind CSS + Babel |
| **Backend** | Python 3.12 + Flask + Gunicorn |
| **DB** | SQLite (calibración + historial) |
| **Deploy** | Render (render.yaml blueprint) |
| **Tests** | pytest |

### Sin dependencias pesadas

El modelo de regresión está implementado con ecuaciones normales (Gaussian elimination) sin necesidad de numpy/scipy. El frontend usa React desde CDN sin webpack/vite. Esto mantiene el deploy liviano y rápido.

---

## Precios Soportados

### Nodos Provisioned (On-Demand USD/hr)

| Tipo | Precio/hr |
|------|----------|
| dc2.large | $0.25 |
| dc2.8xlarge | $4.80 |
| ra3.xlplus | $1.086 |
| ra3.4xlarge | $3.26 |
| ra3.16xlarge | $13.04 |

### Reserved Instance Discounts

| Tipo | Descuento |
|------|----------|
| 1yr no upfront | 20% |
| 1yr partial upfront | 33% |
| 1yr all upfront | 42% |
| 3yr no upfront | 36% |
| 3yr partial upfront | 53% |
| 3yr all upfront | 63% |

### Serverless

- **RPU**: $0.375/RPU-hour
- **Base capacity**: 8-512 RPUs (configurable)

### Spectrum

- **Scanning**: $5/TB de datos escaneados en S3

---

## Mejoras respecto a la versión anterior

- ✅ Parser enriquecido: detecta operación, distribución, condiciones
- ✅ Modelo logarítmico en vez de lineal (/1000 arbitrario eliminado)
- ✅ Sistema de calibración con regresión multi-variable
- ✅ Factor de concurrencia (cluster compartido)
- ✅ Soporte Serverless (RPU-based billing)
- ✅ Estimación de Spectrum ($5/TB)
- ✅ Factores de paralelización variables por distribución
- ✅ Indicador de confianza
- ✅ Persistencia SQLite
- ✅ 23 tests automatizados
- ✅ Herramienta de comparación de queries
- ✅ Deploy a Render con blueprint

---

## Licencia

MIT
