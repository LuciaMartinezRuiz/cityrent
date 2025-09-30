# CityRent – Airbnb Prices and Urban Safety

## Project Title
CityRent

## Short Description of the Problem
Access to housing and the variability of short-term rental (Airbnb) prices across neighborhoods and cities create uncertainty for residents, visitors, and local authorities. Today, information is scattered across multiple sources and rarely allows joint analysis of lodging price with safety or socioeconomic indicators. CityRent proposes an interactive web application that unifies these sources to explore patterns, compare areas, and understand which factors are associated with nightly price.

## Main Objectives
1. Visualize average prices and distribution of short-term rentals by city and neighborhood with interactive maps and charts.  
2. Integrate housing (Airbnb) data with crime and socioeconomic variables (e.g., income, density) for combined analysis.  
3. Model relationships using multiple regression and provide basic explainability of variables.  
4. Deploy a stable public URL with CI/CD and maintain continuous improvements throughout the semester.

## Initial Work Plan (phases & tasks)
### Phase 1 – Repo & Base Deployment (Week 0–1)
Create the `cityrent` repository, multi-page Dash skeleton, basic styling, and “Hello CityRent” deployment.

### Phase 2 – Data & Geometries (Week 1–3)
Ingest and clean Inside Airbnb (listings) plus crime indices and socioeconomic variables (INE/municipal portals). Prepare GeoJSON for neighborhoods/cities.

### Phase 3 – Core Visualizations (Week 3–4)
Neighborhood/city choropleth map, price histograms/boxplots, and a filterable property table.

### Phase 4 – Modeling & Relations (Week 4–6)
Multiple regression (scikit-learn) with metrics (R², MAE) and variable-importance visualization. “Relations” view (scatter plus fitted line).

### Phase 5 – UX & Utility (Week 6–7)
Advanced filters (price range, listing type, bedrooms, rating), CSV downloads, and a “Methodology & Data” page.

### Phase 6 – Quality & Performance (Week 7–8)
Basic tests, cached queries/pre-aggregations, and CI/CD automation (GitHub Actions).

## Planned Data Sources (non-exhaustive)
- Inside Airbnb (listings and coordinates by city)  
- Municipal open data portals (crime by district/neighborhood)  
- INE / Eurostat (socioeconomic variables)

## Repository Structure
```
cityrent/
  app/
    pages/
      home.py            # city map
      neighborhoods.py   # neighborhood details + property list
      relations.py       # regression & correlations
      about.py           # methodology & sources
    app.py               # entry point
    __init__.py
  assets/
    styles.css
  data/
    raw/                 # original CSVs
    processed/           # cleaned parquet/CSV
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

## How to Run Locally
```bash
pip install -r requirements.txt
python app/app.py
# Open http://127.0.0.1:8050
```

## Deployment (suggested)
- Render / Railway / Fly.io with Docker.  
- Environment variables for API keys if applicable.  
- CI/CD via GitHub Actions (auto build and deploy on main).
