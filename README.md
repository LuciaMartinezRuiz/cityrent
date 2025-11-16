# CityRent — Airbnb Prices & Urban Safety (Madrid)

CityRent is an interactive **Dash** web app to explore short-term rental prices (Airbnb) together with urban safety indicators by **district** and **neighbourhood (barrio)** in Madrid. It unifies listings data with municipal security stats so you can visualize spatial patterns, compare areas, and see how safety relates to nightly price.

---

## Live Demo
> Deploy URL (Render): `https://cityrent.onrender.com`  

---

## Key Features

- **Barrios explorer** (`/neighbourhoods`)
  - Map of listings filtered by barrio, room type, and price range.
  - Room-type composition donut.
  - **Safety score bar** (0–100) for the barrio’s **district**, with a red→yellow→green gradient and moving marker.
  - “Similar barrios” chart based on cosine similarity of price stats.

- **Relations** (`/relations`)
  - **Scatter**: median nightly price (Y) vs **mean annual** security actions (X) by district.
  - **OLS regression line**:
    - Multivariate if data coverage allows (rates per 1,000, area shares, and listing features).
    - Falls back to simple OLS on `actions_total` when needed.
  - R² and MAE metrics.

- **Methodology** (`/about`)
  - Data sources and modeling notes.

- **Multipage app** with a top navbar (Dash + `dash-bootstrap-components`).

---

## Data Inputs (required files)

Place these files in **`data/processed/`** at the **repo root**:

- `madrid_listings.parquet`
- `madrid_security_district_annual.parquet`
- `madrid_barrio_to_district.parquet`

### Expected columns (minimum)

**Listings (`madrid_listings.parquet`)**
- `id`, `price`, `latitude`, `longitude`
- `room_type` (e.g., *Entire home/apt*, *Private room*…)
- Recommended (if available): `minimum_nights`, `number_of_reviews`, `availability_365`
- Neighbourhood field: one of `neigh`, `neighbourhood_cleansed`, or `neighbourhood`

**Security (`madrid_security_district_annual.parquet`)**
- `district_name`, `year`, *(optional)* `population`
- Either:
  - **Rates per 1,000**: `seg_ciud_rate_1000`, `seg_vial_rate_1000`, `conviv_rate_1000`, `otras_rate_1000`, `rate_per_1000`
  - **Totals**: `seg_ciud_total`, `seg_vial_total`, `conviv_total`, `otras_total`, `actions_total`  
  *(If `actions_total` is missing, the app sums available numeric columns as a fallback.)*

**Barrio→District mapping (`madrid_barrio_to_district.parquet`)**
- A barrio name column (e.g., `neigh_n`, `neigh_name_official`, or `NOMBRE`)
- `district_name`

> The app auto-normalizes names (accents, hyphens, case) for robust joins.

---

## How the Safety Score Works (0–100)

For the barrio’s **district**:
1. Aggregate **security by district across all years** (mean).
2. Prefer **`rate_per_1000`** if ≥60% of districts have it; otherwise use **`actions_total`**.
3. Rank the district among all districts (lower rate/total ⇒ **safer**).  
   Score = `100 * (1 - rank_position / (N-1))`.  
   - `0` (worst), `50` (median), `100` (best/lowest rate or actions).

The colored bar shows the score with a **red → yellow → green** gradient and a marker at the score value.

---

## Project Structure (key files)
```
cityrent/
  app/
  pages/
  home.py
  neighbourhoods.py
  relations.py
  about.py
  app.py
  data/
  processed/
  madrid_listings.parquet
  madrid_security_district_annual.parquet
  madrid_barrio_to_district.parquet
  requirements.txt
  Dockerfile
  README.md
```

---

## Run Locally

**Prerequisites**: Python 3.11

```bash
# 1) Create and activate a venv (optional but recommended)
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Ensure data files exist:
#    data/processed/madrid_listings.parquet
#    data/processed/madrid_security_district_annual.parquet
#    data/processed/madrid_barrio_to_district.parquet

# 4) Run
python app/app.py
# Open http://127.0.0.1:8050
