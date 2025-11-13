import dash
from dash import html, dcc

dash.register_page(__name__, path="/about", name="Metodología")

MD_TEXT = r'''
# Metodología

## 1) Fuentes de datos
- **Airbnb (listings)** → 'data/processed/madrid_listings.parquet'.
- **Policía Municipal (mensual)** → ficheros 'data/raw/police/pm_YYYY_MM.xlsx'.
- **Padrón municipal** → 'data/raw/padron/poblacion_distrito.csv'.
- **Mapa oficial Barrios↔Distritos** → 'geo/Barrios.csv'.

---

## 2) ETL de seguridad ('data/make_security_districts.py')
1. **Lectura robusta de Excel mensual**
   - Detecta la fila de cabecera buscando 'DISTRICT' en las primeras filas.
   - Identifica la columna de distrito comparando nombres con el listado oficial.
   - Convierte números con formato europeo (puntos = miles, comas = decimales).

2. **Áreas y agregaciones**
   - Suma mensual por distrito y **área**:
     - 'seg_ciud_total' (Seguridad ciudadana)
     - 'seg_vial_total' (Seguridad vial)
     - 'conviv_total' (Convivencia/prevención)
     - 'otras_total'
   - 'actions_total' = suma de todas las áreas.

3. **Anualización + tasas**
   - Suma los 12 meses por distrito y año.
   - Une población del padrón y calcula tasas por 1.000 hab:
     - '*_rate_1000' y 'rate_per_1000' (total).
   - Calcula 'shares' por área: 'seg_ciud_share = seg_ciud_total / actions_total'.

4. **Salidas**
   - 'data/processed/madrid_security_district_monthly.parquet'
   - 'data/processed/madrid_security_district_annual.parquet'
   - 'data/processed/madrid_barrio_to_district.parquet'

---

## 3) Limpieza y claves de unión
- Normalizamos textos (minúsculas, sin acentos, guiones→espacio) y generamos una clave canónica 'district_key'.
- **Listings → Distrito**
  1) Si el barrio ya es un distrito, se usa directamente.  
  2) Si no, se mapea con 'madrid_barrio_to_district.parquet'.  
  3) Se obtiene 'price_median' por distrito (mediana €/noche).
- **Join final**: 'district_key' entre precios y seguridad por año.

---

## 4) Modelo del gráfico 'Relaciones'
- **Objetivo (y)**: 'price_median' (mediana €/noche por distrito).
- **Regresor principal (x)**: 'actions_total'.
- **Variables adicionales** (usadas si existen y con suficiente cobertura):
  - 'rate_per_1000', 'seg_ciud_total', 'seg_ciud_rate_1000', 'seg_ciud_share'.
- **Entrenamiento**:  
  - **OLS multivariable** (train 75% / test 25%).  
  - Si faltan variables suficientes, *fallback* a **OLS simple** 'price ~ actions_total'.
- **Línea del gráfico**: **dependencia parcial** sobre 'actions_total' (el resto de variables se fijan en su media anual).
- **Métricas**: **R²** y **MAE** sobre el conjunto de test.

---

## 5) Interpretación y limitaciones
- 'Actuaciones' ≠ 'delitos': son intervenciones policiales (proxy de actividad).
- Geografía agregada a **distritos** (no barrios), por disponibilidad de datos públicos.
- El modelo es **explicativo aproximado**, no causal.
- Años distintos pueden tener **coberturas diferentes** (si falta población, se usan totales y 'shares').

---

## 6) Reproducibilidad

1. **Generar datasets procesados:**
python data/make_security_districts.py

2. **Lanzar la app:**
python app/app.py
'''

layout = html.Div(
    [
        dcc.Markdown(MD_TEXT, link_target="_blank"),
    ],
    style={"maxWidth": "960px", "margin": "0 auto", "padding": "12px 10px", "lineHeight": "1.55"}
)
