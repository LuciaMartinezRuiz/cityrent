# CityRent – Precios Airbnb y seguridad urbana

## Título del proyecto
CityRent

## Breve descripción del problema a resolver
El acceso a la vivienda y la variabilidad de precios del alquiler turístico (Airbnb) entre barrios y ciudades generan incertidumbre para residentes, visitantes y administraciones. Actualmente, la información se encuentra dispersa en múltiples fuentes y rara vez permite analizar de forma conjunta el precio del alojamiento con indicadores de seguridad o contexto socioeconómico. **CityRent** propone una aplicación web interactiva que unifica estas fuentes para explorar patrones, comparar zonas y entender qué factores se relacionan con el precio por noche.

## Objetivos principales
1. **Visualizar** precios medios y distribución del alquiler turístico por **ciudad y barrio** sobre mapas interactivos y gráficos.
2. **Integrar** datos de vivienda (Airbnb) con **delincuencia** y variables **socioeconómicas** (p. ej., renta, densidad) para análisis conjunto.
3. **Modelar** relaciones mediante **regresión múltiple** y ofrecer explicabilidad básica de variables.
4. **Desplegar** una **URL pública** estable con CI/CD y mantener mejoras continuas durante el semestre.

## Plan inicial de trabajo (fases, tareas previstas)
- **Fase 1 – Repo & Despliegue Base (Semana 0-1)**  
  Crear repositorio `cityrent`, esqueleto Dash multipágina, estilos básicos y despliegue “Hello CityRent”.
- **Fase 2 – Datos & Geometrías (Semana 1-3)**  
  Ingesta y limpieza de **Inside Airbnb** (listings) + índices de **delincuencia** y variables socioeconómicas (INE/ayuntamientos). Preparación de **GeoJSON** de barrios/ciudades.
- **Fase 3 – Visualizaciones base (Semana 3-4)**  
  Mapa choropleth por barrio/ciudad, histogramas/boxplots de precios y tabla filtrable de propiedades.
- **Fase 4 – Modelo & Relaciones (Semana 4-6)**  
  Regresión múltiple (scikit-learn) con métricas (R², MAE) y visualización de importancia de variables. Vista “Relaciones” (scatter + línea de ajuste).
- **Fase 5 – UX & Utilidad (Semana 6-7)**  
  Filtros avanzados (rango de precio, tipo de alojamiento, nº habitaciones, rating), descargas CSV y página “Metodología & Datos”.
- **Fase 6 – Calidad & Rendimiento (Semana 7-8)**  
  Tests básicos, caché de consultas/preagregados y automatización de despliegue (GitHub Actions).

## Fuentes de datos previstas (no exhaustivo)
- Inside Airbnb (listings y coordenadas por ciudad).  
- Portales de datos abiertos municipales (delincuencia por distrito/barrio).  
- INE / Eurostat (variables socioeconómicas).

## Estructura del repositorio
```
cityrent/
  app/
    pages/
      home.py            # mapa ciudades
      neighborhoods.py   # detalle barrios + lista propiedades
      relations.py       # regresión y correlaciones
      about.py           # metodología y fuentes
    app.py               # punto de entrada
    __init__.py
  assets/
    styles.css
  data/
    raw/                 # CSV originales
    processed/           # parquet/CSV limpios
  models/
    train_regression.py
    artifacts/
  geo/
    cities.geojson
    neighborhoods.geojson
  tests/
    test_smoke.py
  requirements.txt
  Dockerfile
  .gitignore
  LICENSE
  README.md
```

## Cómo ejecutar en local
```bash
pip install -r requirements.txt
python app/app.py
# Abrir http://127.0.0.1:8050
```

## Despliegue (sugerido)
- Render / Railway / Fly.io con Docker.  
- Variables de entorno para claves de APIs si aplican.  
- CI/CD con GitHub Actions (build & deploy automático en main).

---
> Nota: Cambia `tuusuario` cuando crees el repo en GitHub: `https://github.com/tuusuario/cityrent`.
