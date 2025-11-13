from pathlib import Path
import re
import pandas as pd
import numpy as np
import unicodedata

# rutas
ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
GEO  = ROOT / "geo"

POLICE_DIR   = RAW / "police"
PADRON_CSV   = RAW / "padron" / "poblacion_distrito.csv"   # CSV/XLSX población (distrito o barrio)
BARRIOS_CSV  = GEO / "Barrios.csv"                         # CSV oficial (barrio->distrito)

PROC.mkdir(parents=True, exist_ok=True)

# -------------------- funciones --------------------
def norm(s):
    """Mayúsculas, sin acentos, para matching exacto (distritos)."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    s = str(s).strip().upper()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s

def norm_loose(s: str) -> str:
    """Minúsculas, sin acentos, guiones, espacio, para matching flexible."""
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[\s\-_]+", " ", s)
    return s

def month_from_fname(p: Path):
    """Extrae (año, mes) del nombre de fichero tipo pm_YYYY_MM.xlsx."""
    m = re.search(r"(\d{4})[_\-](\d{1,2})", p.name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))

def to_number_series(s: pd.Series) -> pd.Series:
    """Convierte serie a numérica, manejando formato europeo."""
    return (
        s.astype(str)
         .str.replace("\u00A0", "", regex=False)  # NBSP
         .str.replace(".", "", regex=False)       # miles
         .str.replace(",", ".", regex=False)      # decimal
         .replace({"": np.nan, "nan": np.nan})
         .pipe(pd.to_numeric, errors="coerce")
    )

# -------------------- Distritos (norm) --------------------
MADRID_DISTRICTS = [
    "Centro","Arganzuela","Retiro","Salamanca","Chamartín","Tetuán","Chamberí",
    "Fuencarral - El Pardo","Moncloa - Aravaca","Latina","Carabanchel","Usera",
    "Puente de Vallecas","Moratalaz","Ciudad Lineal","Hortaleza","Villaverde",
    "Villa de Vallecas","Vicálvaro","San Blas - Canillejas","Barajas"
]
DISTRICT_SET = {norm(x) for x in MADRID_DISTRICTS}

# -------------------- Padrón --------------------
def sniff_read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".xlsx":
        return pd.read_excel(path)
    # CSV con separador desconocido
    try:
        df = pd.read_csv(path, engine="python", sep=None)
        if df.shape[1] > 1:
            return df
    except Exception:
        pass

    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, sep=";", encoding=enc)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue

    return pd.read_csv(path, sep="\t", encoding="utf-8")

def load_population_by_district(csv_path: Path) -> pd.DataFrame:
    """ Carga padrón municipal por distrito (o barrio) y año. """
    if not csv_path.exists():
        cand = list((csv_path.parent).glob("*.*"))
        if not cand:
            raise FileNotFoundError(f"No hay ficheros en {csv_path.parent}")
        csv_path = cand[0]
        print(f"[INFO] Usando {csv_path.name} (detectado en padron/)")
    df = sniff_read_table(csv_path)
    cols = {c.lower(): c for c in df.columns}

    # Columna distrito (nombre)
    dist_col = None
    for key in cols:
        if "distri" in key and ("nombre" in key or "nom" in key or key == "distrito"):
            dist_col = cols[key]; break
    if dist_col is None:
        for key in cols:
            if any(k in key for k in ["distrito","nom distrito","nombre distrito","unidad"]):
                dist_col = cols[key]; break
    if dist_col is None:
        obj_cols = df.select_dtypes(include="object").columns.tolist()
        dist_col = obj_cols[0] if obj_cols else df.columns[0]

    # Columnas año (numéricas)
    year_like = []
    for c in df.columns:
        cc = str(c).strip()
        if cc.isdigit() and 1990 <= int(cc) <= 2100:
            year_like.append(c)

    # Columna población (numérica)
    pop_col = None
    for key in cols:
        if any(k in key for k in ["pobl", "total"]) and key not in ("total_hogares","total_viviendas"):
            pop_col = cols[key]; break

    if year_like:
        out = df.melt(id_vars=[dist_col], value_vars=year_like, var_name="year", value_name="population")
    elif pop_col is not None and any(k in cols for k in ["ano","año","year"]):
        year_col = cols.get("ano") or cols.get("año") or cols.get("year")
        out = df[[dist_col, year_col, pop_col]].copy()
        out.columns = ["district_name", "year", "population"]
    else:
        cand_year = None
        for key in cols:
            if key in ("ano","año","year"):
                cand_year = cols[key]; break
        if cand_year:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not num_cols:
                for c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="ignore")
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                pop_col = max(num_cols, key=lambda c: df[c].notna().sum())
                out = df[[dist_col, cand_year, pop_col]].copy()
                out.columns = ["district_name", "year", "population"]
            else:
                raise ValueError("No encuentro columna numérica de población.")
        else:
            # último recurso
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not num_cols:
                raise ValueError("No se identifican columnas de población/año.")
            out = df[[dist_col, num_cols[0]]].copy()
            out["year"] = pd.Timestamp.today().year
            out.columns = ["district_name", "population", "year"]

    out["district_name"]   = out["district_name"].astype(str).str.strip()
    out["district_name_n"] = out["district_name"].map(norm)
    out["population"]      = to_number_series(out["population"])
    out["year"]            = pd.to_numeric(out["year"], errors="coerce")

    out = (out
           .dropna(subset=["district_name_n","year","population"])
           .groupby(["district_name_n","district_name","year"], as_index=False)["population"].sum())
    out = (out.sort_values(["district_name_n","year"])
             .drop_duplicates(["district_name_n","year"], keep="last"))
    return out

# -------------------- Policía Municipal --------------------
def classify_area(sheet_name: str) -> str:
    """Clasifica área según nombre de hoja."""
    s = norm_loose(sheet_name)
    if "seguridad" in s and "ciudadan" in s:
        return "seg_ciud"
    if "seguridad" in s and "vial" in s:
        return "seg_vial"
    if "conviv" in s or "prevenc" in s:
        return "conviv"
    # fallback
    return "otras"

def read_police_xlsx_month(p: Path) -> pd.DataFrame:
    """Lectura robusta de fichero mensual."""
    def _ncol(s: str) -> str:
        s = str(s or "").strip().lower()
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        s = re.sub(r"[\s\-_]+", " ", s)
        return s

    def classify_area(sheet_name: str) -> str:
        """Clasifica área según nombre de hoja."""
        s = _ncol(sheet_name)
        if "seguridad" in s and "ciudadan" in s:
            return "seg_ciud"
        if "seguridad" in s and "vial" in s:
            return "seg_vial"
        if "conviv" in s or "prevenc" in s:
            return "conviv"
        return "otras"

    xls = pd.ExcelFile(p)
    results = []

    for sheet in xls.sheet_names:
        try:
            raw = xls.parse(sheet_name=sheet, header=None, dtype=str)
        except Exception:
            continue
        if raw.empty:
            continue

        # Localizar fila de cabecera (buscamos 'distrit' en las 30 primeras filas)
        header_idx = None
        for h in range(min(30, len(raw))):
            row_vals = [str(v).strip().upper() for v in raw.iloc[h].tolist()]
            if any("DISTRIT" in v for v in row_vals):
                header_idx = h
                break
        if header_idx is None:
            continue

        df = xls.parse(sheet_name=sheet, header=header_idx, dtype=str)
        df = df.loc[:, ~df.columns.astype(str).str.fullmatch(r"\s*|nan", na=False)]
        if df.empty or df.shape[1] < 2:
            continue

        # Elegir columna DISTRITO por nº de coincidencias con nombres oficiales
        match_counts = {}
        for c in df.columns:
            vals = df[c].dropna().astype(str).head(200).tolist()
            norms = [norm(v) for v in vals]
            matches = sum((v in DISTRICT_SET) for v in norms)
            match_counts[c] = matches
        if not match_counts:
            continue
        dist_col = max(match_counts, key=match_counts.get)
        if match_counts[dist_col] < 8:
            continue

        s = df[[dist_col] + [c for c in df.columns if c != dist_col]].copy()
        s["_DIST_N"] = s[dist_col].map(norm)
        s = s[s["_DIST_N"].isin(DISTRICT_SET)].copy()
        if s.empty:
            continue

        # Convertir resto a número
        for c in s.columns:
            if c in (dist_col, "_DIST_N"):
                continue
            s[c] = to_number_series(s[c])

        area = classify_area(sheet)

        # Suma total área
        total = s.drop(columns=[dist_col, "_DIST_N"]).sum(axis=1, numeric_only=True, min_count=1)
        base = pd.DataFrame({
            "district_name": s[dist_col],
            "district_name_n": s["_DIST_N"],
            "area": area,
            "value_total": total
        })
        results.append(base)

        # Subáreas específicas (solo para seguridad ciudadana)
        if area == "seg_ciud":
            # Mapeo flexible por nombre de columna
            cols_norm = {_ncol(c): c for c in s.columns if c not in (dist_col, "_DIST_N")}
            pick = lambda *keys: next((cols_norm[k] for k in cols_norm if all(w in k for w in keys)), None)

            c_personas = pick("relacionadas", "personas")
            c_patrim   = pick("relacionadas", "patrimonio")
            c_armas    = pick("tenencia", "armas")
            c_ten_dr   = pick("tenencia", "drog")
            c_con_dr   = pick("consumo", "drog")

            sub_rows = []
            if c_personas:
                sub_rows.append(("segciud_personas", s[c_personas]))
            if c_patrim:
                sub_rows.append(("segciud_patrimonio", s[c_patrim]))
            if c_armas:
                sub_rows.append(("segciud_armas", s[c_armas]))
            if c_ten_dr:
                sub_rows.append(("segciud_tenencia_drogas", s[c_ten_dr]))
            if c_con_dr:
                sub_rows.append(("segciud_consumo_drogas", s[c_con_dr]))

            for sub_area, series in sub_rows:
                df_sub = pd.DataFrame({
                    "district_name": s[dist_col],
                    "district_name_n": s["_DIST_N"],
                    "area": sub_area,
                    "value_total": pd.to_numeric(series, errors="coerce")
                })
                results.append(df_sub)

    if not results:
        raise ValueError(f"No encuentro hoja válida con distritos en {p.name}")

    out = pd.concat(results, ignore_index=True)
    out = out.groupby(["district_name","district_name_n","area"], as_index=False)["value_total"].sum()

    y, mo = month_from_fname(p)
    out["year"], out["month"] = y, mo
    return out


def load_police_all_months(police_dir: Path) -> pd.DataFrame:
    """Carga todos los ficheros pm_YYYY_MM.xlsx en un DataFrame largo."""
    files = sorted(police_dir.glob("pm_*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No hay XLSX en {police_dir}. Renombra tus descargas a pm_YYYY_MM.xlsx")
    parts = []
    for f in files:
        try:
            parts.append(read_police_xlsx_month(f))
        except Exception as e:
            print(f"[WARN] {f.name}: {e}")
    if not parts:
        return pd.DataFrame(columns=["district_name","district_name_n","area","value_total","year","month"])
    return pd.concat(parts, ignore_index=True)

def monthly_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """Convierte DataFrame largo mensual a formato ancho por área."""
    if df_long.empty:
        return pd.DataFrame()

    pvt = (df_long
           .pivot_table(index=["district_name", "district_name_n", "year", "month"],
                        columns="area", values="value_total", aggfunc="sum")
           .reset_index()
           .fillna(0.0))

    # Aseguramos que existan las 4 áreas, aunque no hayan aparecido en algún mes
    for c in ["seg_ciud", "seg_vial", "conviv", "otras"]:
        if c not in pvt.columns:
            pvt[c] = 0.0

    # Si viene un 'actions_total' desde el origen, lo eliminamos: calcularemos el nuestro
    if "actions_total" in pvt.columns:
        pvt = pvt.drop(columns=["actions_total"])

    # Totales por área y total global
    pvt["seg_ciud_total"] = pvt["seg_ciud"]
    pvt["seg_vial_total"] = pvt["seg_vial"]
    pvt["conviv_total"]   = pvt["conviv"]
    pvt["otras_total"]    = pvt["otras"]
    pvt["actions_total"]  = (
        pvt["seg_ciud_total"] + pvt["seg_vial_total"] + pvt["conviv_total"] + pvt["otras_total"]
    )

    # Quitamos columnas intermedias y duplicados de nombre por si quedara alguno
    pvt = pvt.drop(columns=["seg_ciud", "seg_vial", "conviv", "otras"], errors="ignore")
    pvt = pvt.loc[:, ~pvt.columns.duplicated()].copy()

    return pvt


def annualize_with_rates(mon_wide: pd.DataFrame, pop_df: pd.DataFrame) -> pd.DataFrame:
    """Agrega anual y calcula tasas por 1.000 hab."""
    if mon_wide.empty:
        return pd.DataFrame()

    # columnas de totales (mensuales) + actions_total si existe
    tot_cols = [c for c in mon_wide.columns if c.endswith("_total")]
    if "actions_total" in mon_wide.columns:
        tot_cols = list(dict.fromkeys(tot_cols + ["actions_total"]))  # dedup y preservar orden

    # Agregado anual
    ann = (mon_wide
           .groupby(["district_name", "district_name_n", "year"], as_index=False)[tot_cols]
           .sum(min_count=1))

    # Merge población "asof" por año
    ann = ann.sort_values(["district_name_n", "year"])
    pop_df = pop_df.sort_values(["district_name_n", "year"])
    out = []
    for k, g in ann.groupby("district_name_n", as_index=False):
        gp = pop_df[pop_df["district_name_n"].eq(k)].copy()
        if gp.empty:
            g["population"] = np.nan
        else:
            g = pd.merge_asof(g, gp[["year", "population"]], on="year", direction="backward")
        out.append(g)
    ann = pd.concat(out, ignore_index=True)

    # Asegura nombres de columnas únicos por si viniera algo repetido
    ann = ann.loc[:, ~ann.columns.duplicated()].copy()

    # Tasas por 1.000 hab. (excluye actions_total aquí para no tratarlo dos veces)
    bases_rate = [c for c in ann.columns if c.endswith("_total") and c != "actions_total"]
    for base in bases_rate:
        rate_col = base.replace("_total", "_rate_1000")
        ann[rate_col] = np.where(
            ann["population"].gt(0),
            ann[base] / ann["population"] * 1000.0,
            np.nan
        )

    # Y ahora la tasa global de actions_total (si existe)
    if "actions_total" in ann.columns and "actions_total_rate_1000" not in ann.columns:
        ann["actions_total_rate_1000"] = np.where(
            ann["population"].gt(0),
            ann["actions_total"] / ann["population"] * 1000.0,
            np.nan
        )

    return ann


def build_barrio_to_district_map(barrios_csv: Path) -> pd.DataFrame:
    """Construye mapa barrio a distrito desde CSV oficial."""
    import re
    def norm_local(s: str):
        if s is None or (isinstance(s, float) and pd.isna(s)): return None
        s = str(s).strip().lower()
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        s = re.sub(r"[^a-z0-9 ñ]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    if not barrios_csv.exists():
        raise FileNotFoundError(f"No existe {barrios_csv}")
    b = pd.read_csv(barrios_csv)

    if not {"NOMBRE","NOMDIS"}.issubset(b.columns):
        raise ValueError(f"CSV inesperado. Cabeceras: {b.columns.tolist()}")

    name_cols = [c for c in ["NOMBRE","BARRIO_MAY","BARRIO_MT"] if c in b.columns]
    rows = []
    for _, row in b.iterrows():
        dist_name = row["NOMDIS"]
        dist_code = row["CODDIS"] if "CODDIS" in b.columns else None
        for nc in name_cols:
            barrio = row[nc]
            key = norm_local(barrio)
            if key:
                rows.append({
                    "neigh_name_official": barrio,
                    "district_name": dist_name,
                    "district_code": dist_code,
                    "neigh_n": key,
                    "district_name_n": norm_local(dist_name),
                })
    m = pd.DataFrame(rows).drop_duplicates(subset=["neigh_n"]).reset_index(drop=True)
    return m

def main():
    """ETL completo de datos de seguridad por distrito."""
    print(">> Cargando población…")
    pop = load_population_by_district(PADRON_CSV)
    print(f"  - {pop['district_name_n'].nunique()} distritos, años {int(pop['year'].min())}–{int(pop['year'].max())}")

    print(">> Cargando actuaciones Policía (mensuales)…")
    mon_long = load_police_all_months(POLICE_DIR)
    if mon_long.empty:
        print("[ERROR] No se ha podido leer ningún XLSX mensual. Revisa nombres pm_YYYY_MM.xlsx y hojas con distritos.")
        return
    mon_wide = monthly_wide(mon_long)
    print(f"  - Meses válidos: {mon_wide[['year','month']].drop_duplicates().shape[0]}  | Registros: {len(mon_wide)}")

    print(">> Agregando a anual + tasas/1000 hab…")
    ann = annualize_with_rates(mon_wide, pop)

    # Guardar
    mon_wide.to_parquet(PROC / "madrid_security_district_monthly.parquet", index=False)
    ann.to_parquet(PROC / "madrid_security_district_annual.parquet", index=False)
    print("OK -> data/processed/madrid_security_district_{monthly,annual}.parquet")

    # Mapa barrio→distrito
    try:
        m = build_barrio_to_district_map(BARRIOS_CSV)
        m.to_parquet(PROC / "madrid_barrio_to_district.parquet", index=False)
        print("OK -> data/processed/madrid_barrio_to_district.parquet")
    except Exception as e:
        print(f"[WARN] Mapa barrio→distrito: {e}")

if __name__ == "__main__":
    main()
