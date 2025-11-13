import dash
from dash import html, dcc, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
import re, unicodedata

# ML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

dash.register_page(__name__, path="/relations", name="Relaciones")

# rutas
ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
CITY = "madrid"

# -------------------- funciones --------------------
def norm(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9 ñ\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def canon_key(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    s = str(s).strip().lower().replace("–", " ").replace("-", " ").replace("/", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def normalize_cols_lookup(cols):
    out = {}
    for c in cols:
        n = unicodedata.normalize("NFKD", str(c).strip().lower())
        n = "".join(ch for ch in n if not unicodedata.combining(ch))
        out[re.sub(r"[\s\-_]+", " ", n)] = c
    return out

def empty_figure(height=520):
    fig = go.Figure()
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=height)
    return fig

# -------------------- carga de datos --------------------
LIST_PATH    = PROC / f"{CITY}_listings.parquet"
SEC_ANN_PATH = PROC / "madrid_security_district_annual.parquet"
B2D_PATH     = PROC / "madrid_barrio_to_district.parquet"

listings = pd.read_parquet(LIST_PATH).copy()
listings["price"] = pd.to_numeric(listings["price"], errors="coerce")
if "neigh" in listings.columns:
    listings["neigh"] = listings["neigh"].astype(str).str.replace("–", "-", regex=False)

SEC_ANN = pd.read_parquet(SEC_ANN_PATH) if SEC_ANN_PATH.exists() else pd.DataFrame()

def ensure_security_columns(sec: pd.DataFrame) -> pd.DataFrame:
    if sec.empty:
        return sec
    sec = sec.copy()
    sec["year"] = pd.to_numeric(sec.get("year", np.nan), errors="coerce")
    if "district_name_n" not in sec.columns and "district_name" in sec.columns:
        sec["district_name_n"] = sec["district_name"].map(norm)

    lut = normalize_cols_lookup(sec.columns)
    pop_col = lut.get("population", None)

    known_meta = {
        "district_name", "district_name_n", "year", "population",
        "actions_total", "rate_per_1000", "delincuencia_total", "delincuencia_rate_1000"
    }
    num_cols = [c for c in sec.columns if c not in known_meta and pd.api.types.is_numeric_dtype(sec[c])]

    if "actions_total" not in sec.columns or sec["actions_total"].isna().all():
        sec["actions_total"] = sec[num_cols].sum(axis=1, numeric_only=True) if num_cols else np.nan

    if "rate_per_1000" not in sec.columns or sec["rate_per_1000"].isna().all():
        if pop_col and pop_col in sec.columns:
            sec["rate_per_1000"] = np.where(
                sec[pop_col].gt(0), sec["actions_total"] / sec[pop_col] * 1000.0, np.nan
            )
        else:
            sec["rate_per_1000"] = np.nan
    return sec

if not SEC_ANN.empty:
    SEC_ANN = ensure_security_columns(SEC_ANN)

B2D = pd.read_parquet(B2D_PATH) if B2D_PATH.exists() else pd.DataFrame()
if not B2D.empty:
    B2D = B2D.copy()
    if "neigh_n" not in B2D.columns:
        for cand in ("neigh_name_official", "NOMBRE", "neigh_n"):
            if cand in B2D.columns:
                B2D["neigh_n"] = B2D[cand].map(norm)
                break
    if "district_name" in B2D.columns:
        B2D["district_name_n"] = B2D["district_name"].map(norm)
    else:
        B2D = pd.DataFrame()

neigh_col = next((c for c in ["neigh","neighbourhood_cleansed","neighbourhood"] if c in listings.columns), None)
listings["neigh"] = listings[neigh_col] if neigh_col else pd.NA
listings["neigh_n"] = listings["neigh"].map(norm)

DIST_CANON = {}
if not SEC_ANN.empty and "district_name" in SEC_ANN.columns:
    DIST_CANON = {norm(n): n for n in SEC_ANN["district_name"].dropna().unique()}

# -------------------- mapping --------------------
def map_listings_to_district_rows(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["district_direct"] = d["neigh_n"].map(DIST_CANON)
    if not B2D.empty:
        mapped = d[d["district_direct"].isna()].merge(
            B2D[["neigh_n", "district_name"]], on="neigh_n", how="left"
        )
        d.loc[d["district_direct"].isna(), "district_mapped"] = mapped["district_name"].values
    else:
        d["district_mapped"] = np.nan
    d["district_name"] = d["district_direct"].fillna(d["district_mapped"])
    d = d[~d["district_name"].isna()].copy()
    d["district_name_n"] = d["district_name"].map(norm)
    d["district_key"] = d["district_name_n"].map(canon_key)
    return d

def price_by_district_after_mapping(df: pd.DataFrame) -> pd.DataFrame:
    d = map_listings_to_district_rows(df)
    g = d.groupby("district_name_n", dropna=False)["price"].median().reset_index(name="price_median")

    if not SEC_ANN.empty and "district_name" in SEC_ANN.columns:
        canon = SEC_ANN[["district_name_n", "district_name"]].drop_duplicates()
        g = g.merge(canon, on="district_name_n", how="left")
        g["district_name"] = g["district_name"].fillna(g["district_name_n"])
    else:
        g["district_name"] = g["district_name_n"]

    g["district_key"] = g["district_name_n"].map(canon_key)
    return g[["district_name_n", "district_name", "district_key", "price_median"]]

def listing_features_by_district(df: pd.DataFrame) -> pd.DataFrame:
    d = map_listings_to_district_rows(df)

    agg_spec = {"n_listings": ("price", "size")}
    if "room_type" in d.columns:
        agg_spec["pct_entire"] = ("room_type", lambda s: (s == "Entire home/apt").mean())
    if "minimum_nights" in d.columns:
        agg_spec["min_nights_median"] = ("minimum_nights", "median")
    if "number_of_reviews" in d.columns:
        agg_spec["reviews_mean"] = ("number_of_reviews", "mean")
    if "availability_365" in d.columns:
        agg_spec["avail_mean"] = ("availability_365", "mean")

    if len(agg_spec) == 1:
        return d.groupby("district_key").agg(**agg_spec).reset_index()

    agg = d.groupby("district_key").agg(**agg_spec).reset_index()
    return agg

# -------------------- builder (TODOS LOS AÑOS JUNTOS) --------------------
def build_relations_all() -> pd.DataFrame:
    """
    Une todos los años del dataset de seguridad en una sola tabla por distrito.
    - Para 'actions_total' y rates usa la MEDIA anual.
    - Calcula shares a partir de medias (o de totales si están).
    """
    if listings.empty or SEC_ANN.empty:
        return pd.DataFrame()

    prices = price_by_district_after_mapping(listings)

    sec = SEC_ANN.copy()
    sec["district_key"] = sec["district_name_n"].map(canon_key)

    # columnas candidatas
    rate_cols  = [c for c in ["seg_ciud_rate_1000","seg_vial_rate_1000","conviv_rate_1000","otras_rate_1000","rate_per_1000"] if c in sec.columns]
    total_cols = [c for c in ["seg_ciud_total","seg_vial_total","conviv_total","otras_total","actions_total"] if c in sec.columns]

    # agregación por distrito (medias de numéricas + primer nombre de distrito)
    num_cols = [c for c in sec.columns if c not in {"district_name","district_name_n","year","district_key"} and pd.api.types.is_numeric_dtype(sec[c])]
    agg_dict = {c:"mean" for c in num_cols}
    agg_dict["district_name"] = lambda s: s.dropna().iloc[0] if s.dropna().any() else np.nan

    sec_agg = sec.groupby("district_key").agg(agg_dict).reset_index()

    # shares (si hay datos)
    if "actions_total" in sec_agg.columns:
        for area in ["seg_ciud","seg_vial","conviv","otras"]:
            tcol = f"{area}_total"
            if tcol in sec_agg.columns:
                sec_agg[f"{area}_share"] = np.where(
                    sec_agg["actions_total"].gt(0), sec_agg[tcol] / sec_agg["actions_total"], np.nan
                )

    # merge con precios y features de listings
    rel = prices.merge(sec_agg, on="district_key", how="inner")

    add_feats = listing_features_by_district(listings)
    rel = rel.merge(add_feats, on="district_key", how="left")

    # coerción numérica
    for c in rel.columns:
        if c not in {"district_name", "district_key", "district_name_n"}:
            rel[c] = pd.to_numeric(rel[c], errors="coerce")

    rel = rel.dropna(subset=["price_median", "actions_total"])
    return rel

# -------------------- figuras --------------------
def _hover_col(df: pd.DataFrame) -> str:
    return "district_name" if "district_name" in df.columns else ("district_name_n" if "district_name_n" in df.columns else df.columns[0])

def _robust_limits(series, pad=0.05):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return (0.0, 1.0)
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo_f, hi_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    lo = max(float(s.min()), float(lo_f))
    hi = min(float(s.max()), float(hi_f))
    if not np.isfinite(hi) or hi <= lo:
        lo, hi = float(s.min()), float(s.max())
    span = hi - lo
    padv = span * pad if span > 0 else 0
    return lo - padv, hi + padv

def make_scatter(df_all: pd.DataFrame):
    hover = _hover_col(df_all)
    fig = px.scatter(
        df_all, x="actions_total", y="price_median", hover_name=hover,
        labels={"actions_total": "Actuaciones (media anual)", "price_median": "Precio mediano (€)"}
    )
    fig.update_traces(marker=dict(size=9, opacity=0.85))
    xr = _robust_limits(df_all["actions_total"])
    yr = _robust_limits(df_all["price_median"])
    fig.update_xaxes(range=list(xr), automargin=True)
    fig.update_yaxes(range=list(yr), automargin=True)
    fig.update_layout(
        autosize=False, height=520,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        uirevision="rel-all", transition={'duration': 0},
        legend_itemclick=False, legend_itemdoubleclick=False,
    )
    return fig

# -------------------- Regresión (mismo criterio) --------------------
RATE_COLS  = ["seg_ciud_rate_1000","seg_vial_rate_1000","conviv_rate_1000","otras_rate_1000","rate_per_1000"]
TOTAL_COLS = ["seg_ciud_total","seg_vial_total","conviv_total","otras_total"]
SHARE_COLS = ["seg_ciud_share","seg_vial_share","conviv_share","otras_share"]
LISTING_COLS = ["n_listings","pct_entire","min_nights_median","reviews_mean","avail_mean"]

def pick_features(df: pd.DataFrame, coverage=0.6):
    feats = ["actions_total"]
    rates_ok  = [c for c in RATE_COLS  if c in df.columns and df[c].notna().mean() >= coverage]
    totals_ok = [c for c in TOTAL_COLS if c in df.columns and df[c].notna().mean() >= coverage]
    feats += (rates_ok or totals_ok)
    feats += [c for c in SHARE_COLS   if c in df.columns and df[c].notna().mean() >= coverage]
    feats += [c for c in LISTING_COLS if c in df.columns and df[c].notna().sum() >= max(5, int(0.3*len(df)))]
    return list(dict.fromkeys(feats))

def add_regression_line(fig, df_all: pd.DataFrame):
    if df_all is None or df_all.empty:
        return fig, np.nan, np.nan

    xr = fig.layout.xaxis.range or _robust_limits(df_all["actions_total"])
    x_min, x_max = float(xr[0]), float(xr[1])

    features = pick_features(df_all, coverage=0.6)
    work = df_all.dropna(subset=["price_median", "actions_total"]).copy()

    if len(features) > 1 and len(work.dropna(subset=features)) >= 6:
        work = work.dropna(subset=features)
        X = work[features].astype(float).values
        y = work["price_median"].astype(float).values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
        model = LinearRegression().fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        r2, mae = r2_score(y_te, y_pred), mean_absolute_error(y_te, y_pred)

        x_line = np.linspace(x_min, x_max, 200)
        means = work[features].mean()
        X_line = np.tile(means.values, (len(x_line), 1))
        X_line[:, features.index("actions_total")] = x_line
        y_line = model.predict(X_line)

        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines",
                                 name="Regresión (multivar OLS)", hoverinfo="skip", showlegend=True))
        return fig, r2, mae

    if len(work) >= 4:
        X = work[["actions_total"]].astype(float).values
        y = work["price_median"].astype(float).values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
        model = LinearRegression().fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        r2, mae = r2_score(y_te, y_pred), mean_absolute_error(y_te, y_pred)

        x_line = np.linspace(x_min, x_max, 200).reshape(-1, 1)
        y_line = model.predict(x_line)
        fig.add_trace(go.Scatter(x=x_line.ravel(), y=y_line, mode="lines",
                                 name="Regresión (OLS)", hoverinfo="skip", showlegend=True))
        return fig, r2, mae

    return fig, np.nan, np.nan

# -------------------- Layout --------------------
layout = html.Div(
    [
        html.H2("Relación de Precio y Seguridad por distrito en Madrid"),
        html.Div(
            "Cada punto es un distrito. En el eje X se muestran las actuaciones (media anual) y en el Y el precio mediano de los anuncios. "
            "La línea de tendencia proviene de un modelo OLS que, cuando hay datos, ajusta por tasas por 1.000 habitantes, "
            "reparto por áreas y rasgos de los anuncios. Se usa toda la serie histórica disponible, agregada por distrito.",
            style={"fontSize":"14px","color":"#555","margin":"6px 0"}
        ),
        dcc.Graph(id="rel-graph", style={"height":"520px"}, config={"displayModeBar": False}),
        html.Div(id="rel-metrics", style={"margin":"6px 0", "color":"#555"}),

        html.Div(
            [
                html.Div([html.Span("Variables candidatas del modelo:", style={"fontWeight":600})],
                         style={"display":"flex","alignItems":"center","gap":"8px","color":"#374151"}),
                html.Div(
                    id="rel-explain",
                    style={"fontSize":"13px","color":"#4B5563","marginTop":"6px",
                           "lineHeight":"1.5","whiteSpace":"pre-line"}
                ),
            ],
            id="rel-explain-card",
            role="note",
            style={"marginTop":"10px","padding":"12px 14px","backgroundColor":"#F9FAFB",
                   "border":"1px solid #E5E7EB","borderRadius":"10px",
                   "boxShadow":"0 1px 2px rgba(16,24,40,0.05)"}
        ),
    ]
)

# -------------------- Callback (sin selector de año) --------------------
@dash.callback(
    Output("rel-graph", "figure"),
    Output("rel-metrics", "children"),
    Output("rel-explain", "children"),
    Input("rel-graph", "id"),   # dispara al cargar la página
)
def update_graph(_):
    if SEC_ANN.empty or listings.empty:
        return empty_figure(), html.Span("Faltan datasets. Revisa data/processed/…"), ""

    df_all = build_relations_all()
    if df_all is None or df_all.empty:
        return empty_figure(), html.Span("Sin datos para entrenar el modelo."), ""

    fig = make_scatter(df_all)
    fig, r2, mae = add_regression_line(fig, df_all)

    metrics = html.Span(f"R²={('–' if pd.isna(r2) else f'{r2:.3f}')}  |  MAE={('–' if pd.isna(mae) else f'{mae:.0f}')}€")

    explain = html.Ul(
        [
            html.Li("Actuaciones totales (media anual) como variable principal (X)."),
            html.Li(["Tasas por 1.000 hab. y/o totales por área: ", html.B("Seg. ciudadana, Vial, Convivencia, Otras"), "."]),
            html.Li("“Shares” por área (proporción de cada área dentro del total de actuaciones)."),
            html.Li(["Rasgos de los anuncios por distrito: ",
                     "nº de anuncios, % Entire home/apt, mediana de minimum_nights, media de reviews y availability."]),
            html.Li("Se agregan todos los años disponibles por distrito; la línea se dibuja variando solo Actuaciones y manteniendo el resto en su media."),
        ],
        style={"margin":"6px 0 0 0", "paddingLeft":"18px", "listStyleType":"disc"}
    )

    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), uirevision="rel-all")
    return fig, metrics, explain
