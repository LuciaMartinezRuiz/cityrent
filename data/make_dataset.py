import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

CITY = "madrid"

# localizar listings (csv / csv.gz)
listings_path = None
for patt in ("listings.csv", "listings.csv.gz"):
    p = RAW / patt
    if p.exists():
        listings_path = p
        break
if listings_path is None:
    raise FileNotFoundError("Pon listings.csv(.gz) dentro de data/raw/")

# leer (el CSV ya viene con price numérico o NaN)
df = pd.read_csv(listings_path, low_memory=False)

# columnas presentes en el fichero
expected = [
    "id","name","host_id",
    "neighbourhood_group","neighbourhood",
    "latitude","longitude","room_type","price",
    "minimum_nights","number_of_reviews",
    "availability_365","number_of_reviews_ltm","license"
]
keep = [c for c in expected if c in df.columns]
df = df[keep].copy()


# barrio y distrito
df["neigh"] = df["neighbourhood"]
df["district"] = df["neighbourhood_group"] if "neighbourhood_group" in df.columns else pd.NA


# --- LIMPIEZA PRECIO ROBUSTA ---
# "$123.00", "€99", "  " → 123.0, 99.0, NaN
price_str = (
    df["price"]
    .astype(str)
    .str.replace(r"[^0-9.,]", "", regex=True)  # quita símbolos
    .str.replace(",", ".", regex=False)        # coma → punto
    .str.strip()
    .replace({"": None})                       # vacío → None
)

df["price"] = pd.to_numeric(price_str, errors="coerce")  # convierte y deja NaN si no se puede

# --- FILTRO RAZONABLE DE PRECIOS---
MIN_PRICE, MAX_PRICE = 15, 500
df = df[(df["price"] >= MIN_PRICE) & (df["price"] <= MAX_PRICE)].copy()
print("DEBUG price range =>", float(df["price"].min()), float(df["price"].max()))



# quitar anuncios sin coordenadas válidas
if "latitude" in df.columns and "longitude" in df.columns:
    df = df[df["latitude"].between(39.8, 41.2) & df["longitude"].between(-4.5, -3.0)]


# agregados por barrio
agg = (
    df.groupby("neigh", dropna=False)
      .agg(
          price_mean=("price","mean"),
          price_median=("price","median"),
          n_listings=("id","count")
      )
      .reset_index()
      .rename(columns={"neigh":"neighbourhood"})
)

# guardar resultados
df.to_parquet(PROC / f"{CITY}_listings.parquet", index=False)
agg.to_parquet(PROC / f"{CITY}_agg.parquet", index=False)

print("OK ->", PROC / f"{CITY}_listings.parquet", "|", PROC / f"{CITY}_agg.parquet")
