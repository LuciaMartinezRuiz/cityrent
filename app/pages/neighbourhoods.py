import dash
from dash import html, dcc, Input, Output, State, no_update
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
import re, unicodedata

# ML para similitud (no para el modelo de relaciones)
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import cosine_similarity
    HAVE_SK = True
except Exception:
    HAVE_SK = False

dash.register_page(__name__, path="/neighbourhoods", name="Barrios")

# -------------------- rutas (idénticas a relations.py) --------------------
ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
CITY = "madrid"

LIST_PATH    = PROC / f"{CITY}_listings.parquet"
SEC_ANN_PATH = PROC / "madrid_security_district_annual.parquet"
B2D_PATH     = PROC / "madrid_barrio_to_district.parquet"

# -------------------- helpers (como en relations.py) --------------------
def norm(s):
    """Normalizar string para matching."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9 ñ\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def canon_key(s):
    """Normalización para join keys."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    s = str(s).strip().lower().replace("–", " ").replace("-", " ").replace("/", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def normalize_cols_lookup(cols):
    """Crear lookup de columnas normalizadas a originales."""
    out = {}
    for c in cols:
        n = unicodedata.normalize("NFKD", str(c).strip().lower())
        n = "".join(ch for ch in n if not unicodedata.combining(ch))
        out[re.sub(r"[\s\-_]+", " ", n)] = c
    return out

def ensure_security_columns(sec: pd.DataFrame) -> pd.DataFrame:
    """Asegurar columnas clave en security dataset."""
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

# -------------------- carga datos --------------------
listings = pd.read_parquet(LIST_PATH).copy()
listings["price"] = pd.to_numeric(listings["price"], errors="coerce")
if "neigh" in listings.columns:
    listings["neigh"] = listings["neigh"].astype(str).str.replace("–", "-", regex=False)

neigh_col = next((c for c in ["neigh","neighbourhood_cleansed","neighbourhood"] if c in listings.columns), None)
listings["neigh"] = listings[neigh_col] if neigh_col else pd.NA
listings["neigh_n"] = listings["neigh"].map(norm)

SEC_ANN = pd.read_parquet(SEC_ANN_PATH) if SEC_ANN_PATH.exists() else pd.DataFrame()
if not SEC_ANN.empty:
    SEC_ANN = ensure_security_columns(SEC_ANN)
    if "district_name_n" not in SEC_ANN.columns and "district_name" in SEC_ANN.columns:
        SEC_ANN["district_name_n"] = SEC_ANN["district_name"].map(norm)
    SEC_ANN["district_key"] = SEC_ANN["district_name_n"].map(canon_key)

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
        B2D["district_key"] = B2D["district_name_n"].map(canon_key)
    else:
        B2D = pd.DataFrame()

DIST_CANON = {}
if not SEC_ANN.empty and "district_name" in SEC_ANN.columns:
    DIST_CANON = {norm(n): n for n in SEC_ANN["district_name"].dropna().unique()}

# -------------------- mapping --------------------
def map_listings_to_district_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Mapear listings a filas con distrito."""
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

def listing_features_by_district(df: pd.DataFrame) -> pd.DataFrame:
    """Características agregadas de listings por distrito."""
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
    return d.groupby("district_key").agg(**agg_spec).reset_index()

# -------------------- features por barrio (similitud) --------------------
def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla de características agregadas por barrio."""
    g = df.groupby("neigh", dropna=False)
    out = pd.DataFrame({
        "price_median": g["price"].median(),
        "price_p25": g["price"].quantile(0.25),
        "price_p75": g["price"].quantile(0.75),
        "n_listings": g["id"].count()
    })
    out["iqr"] = (out["price_p75"] - out["price_p25"]).astype(float)
    try:
        out["pct_entire"] = g["room_type"].apply(lambda s: (s == "Entire home/apt").mean())
    except Exception:
        out["pct_entire"] = pd.NA
    return out.reset_index()

FEAT = build_feature_table(listings)

SIM = None
SIM_FEATURES = ["price_median", "iqr", "pct_entire"]
if HAVE_SK:
    base = FEAT[["neigh"] + [c for c in SIM_FEATURES if c in FEAT.columns]].dropna()
    if len(base) >= 2:
        Xs = StandardScaler().fit_transform(base.drop(columns=["neigh"]).astype(float).values)
        SIM = pd.DataFrame(cosine_similarity(Xs), index=base["neigh"], columns=base["neigh"])

# -------------------- resolver distrito de un barrio --------------------
def resolve_district_from_neigh(neigh_value: str):
    """Dado un barrio, devolver (nombre distrito, clave distrito)"""
    if not neigh_value:
        return None, None
    n_n = norm(neigh_value)
    if n_n in DIST_CANON:
        dname = DIST_CANON[n_n]
        return dname, canon_key(norm(dname))
    if not B2D.empty and "neigh_n" in B2D.columns:
        hit = B2D[B2D["neigh_n"] == n_n]
        if not hit.empty:
            if "district_key" in hit.columns and pd.notna(hit.iloc[0]["district_key"]):
                dkey = hit.iloc[0]["district_key"]
                if "district_name" in SEC_ANN.columns:
                    cand = SEC_ANN[SEC_ANN["district_key"] == dkey]["district_name"]
                    if not cand.empty:
                        return cand.iloc[0], dkey
                dname = hit.iloc[0].get("district_name")
                return (dname if pd.notna(dname) else None), dkey
            dname = hit.iloc[0].get("district_name")
            if pd.notna(dname):
                return dname, canon_key(norm(dname))
    return None, None

# -------------------- tabla de referencia para puntuación --------------------
def _reference_security_table() -> pd.DataFrame:
    """Tabla de referencia con medias por distrito."""
    if SEC_ANN.empty:
        return pd.DataFrame()
    ref = (
        SEC_ANN.groupby("district_key")
        .agg(
            rate_mean=("rate_per_1000", "mean"),
            actions_mean=("actions_total", "mean"),
            district_name=("district_name", lambda s: s.dropna().iloc[0] if s.dropna().any() else np.nan),
        )
        .reset_index()
    )
    ref["rate_mean"] = pd.to_numeric(ref["rate_mean"], errors="coerce")
    ref["actions_mean"] = pd.to_numeric(ref["actions_mean"], errors="coerce")
    return ref.set_index("district_key")

def _safety_score_for_district(dkey: str, ref: pd.DataFrame):
    """Puntuación de seguridad [0,100] para un distrito dado la tabla de referencia."""
    if ref.empty or dkey not in ref.index:
        return np.nan, "none", 0
    n = len(ref)
    non_na_rate = ref["rate_mean"].notna().sum()
    use_rate = (non_na_rate >= max(3, int(0.6 * n))) and pd.notna(ref.loc[dkey, "rate_mean"])
    metric = "rate_mean" if use_rate else "actions_mean"
    series = ref[metric].dropna()
    n_eff = len(series)
    if n_eff <= 1 or dkey not in series.index:
        return np.nan, metric, n_eff
    sorted_keys = series.sort_values(ascending=True).index.tolist()
    pos = sorted_keys.index(dkey)
    if n_eff == 1:
        return 100.0, metric, n_eff
    score = 100.0 * (1.0 - (pos / (n_eff - 1)))
    return float(score), metric, n_eff

# -------------------- Barra fina rojo, amarillo, verde con punto --------------------
def _hex_to_rgb(h):
    """Convertir color hex a tupla RGB."""
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _rgb_to_hex(rgb):
    """Convertir tupla RGB a color hex."""
    return "#%02x%02x%02x" % rgb

def _lerp(a, b, t):
    """Interpolación lineal entre a y b con t en [0,1]."""
    return a + (b - a) * t

def _interp_color(c1, c2, t):
    """Interpolar entre dos colores hex c1 y c2 con t en [0,1]."""
    r1,g1,b1 = _hex_to_rgb(c1)
    r2,g2,b2 = _hex_to_rgb(c2)
    r = int(round(_lerp(r1, r2, t)))
    g = int(round(_lerp(g1, g2, t)))
    b = int(round(_lerp(b1, b2, t)))
    return _rgb_to_hex((r,g,b))

def _gradient_color(x):
    """
    x en [0,100].
      0  → rojo   (#f80c0c)
      50 → amarillo (#facc15)
      100→ verde  (#22c55e)
    """
    stops = [(0, "#f80c0c"), (50, "#facc15"), (100, "#22c55e")]
    if x <= 0: return stops[0][1]
    if x >= 100: return stops[-1][1]
    for i in range(len(stops)-1):
        x0, c0 = stops[i]
        x1, c1 = stops[i+1]
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0) if x1 > x0 else 0
            return _interp_color(c0, c1, t)
    return stops[-1][1]

def make_security_slimbar(score: float) -> go.Figure:
    """Barra horizontal FINA (rojo, amarillo, verde) y punto marcador en el score."""
    fig = go.Figure()
    y0, y1 = 0.46, 0.54  # dimensiones barra

    if pd.isna(score):
        fig.add_shape(type="rect", x0=0, x1=100, y0=y0, y1=y1, line=dict(width=0), fillcolor="#E5E7EB")
        fig.add_annotation(text="Sin datos", x=50, y=0.8, showarrow=False, font=dict(size=13, color="#6B7280"))
    else:
        steps = 100
        xs = np.linspace(0, 100, steps+1)
        for i in range(steps):
            x0 = float(xs[i]); x1 = float(xs[i+1])
            c  = _gradient_color((x0 + x1) / 2.0)
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, line=dict(width=0), fillcolor=c)

        s = max(0, min(100, float(score)))
        marker_color = _gradient_color(s)
        fig.add_trace(go.Scatter(
            x=[s], y=[0.5],
            mode="markers",
            marker=dict(size=12, color=marker_color, line=dict(width=2, color="#111827")),
            hoverinfo="skip", showlegend=False
        ))
        fig.add_annotation(x=s, y=0.78, text=f"{s:.0f}/100", showarrow=False,
                           font=dict(size=13, color="#111827"), xanchor="center")

    fig.update_xaxes(range=[0, 100], showticklabels=False, showgrid=False, zeroline=False, fixedrange=True)
    fig.update_yaxes(range=[0, 1], showticklabels=False, showgrid=False, zeroline=False, fixedrange=True)
    fig.update_layout(height=60, margin=dict(l=8, r=8, t=8, b=8), plot_bgcolor="white", paper_bgcolor="white")
    return fig

# -------------------- Seguridad: panel --------------------
def security_card_for_neigh_all_years(neigh_value: str) -> html.Div:
    """Generar tarjeta de seguridad para un barrio dado (todos los años)."""
    if not neigh_value or SEC_ANN.empty:
        return html.Div()

    district_name, district_key = resolve_district_from_neigh(neigh_value)
    if not district_key:
        return html.Div(
            "⚠️ No se pudo determinar el distrito del barrio seleccionado.",
            style={"color":"#92400E","background":"#FEF3C7","border":"1px solid #FDE68A",
                   "padding":"8px 10px","borderRadius":8, "marginTop":8}
        )

    ref = _reference_security_table()
    if ref.empty or district_key not in ref.index:
        return html.Div(
            f"⚠️ No hay datos de seguridad para {district_name or district_key}.",
            style={"color":"#92400E","background":"#FEF3C7","border":"1px solid #FDE68A",
                   "padding":"8px 10px","borderRadius":8, "marginTop":8}
        )

    score, metric, n_eff = _safety_score_for_district(district_key, ref)

    def label_for_score(s):
        """Etiqueta cualitativa para una puntuación de seguridad."""
        if pd.isna(s): return "Sin datos"
        if s >= 80: return "Muy seguro"
        if s >= 60: return "Seguro"
        if s >= 40: return "Intermedio"
        if s >= 20: return "Bajo"
        return "Muy bajo"

    display_name = district_name or ref.loc[district_key, "district_name"]
    label = label_for_score(score)
    metric_note = "tasa por 1.000 hab." if metric == "rate_mean" else "actuaciones totales"
    bar_fig = make_security_slimbar(score)

    return html.Div(
        [
            html.Div(
                [
                    html.Div([html.Small("Barrio"), html.H5(neigh_value, style={"margin": 0})],
                             style={"flex": "0 0 180px", "minWidth": "160px"}),
                    html.Div([html.Small("Distrito"), html.H5(str(display_name or district_key), style={"margin": 0})],
                             style={"flex": "0 0 220px", "minWidth": "200px"}),
                    html.Div([html.Small("Score"), html.H3("–" if pd.isna(score) else f"{score:0.0f}", style={"margin": 0})],
                             style={"flex": "0 0 100px"}),
                    html.Div([html.Small("Clasificación"), html.H3(label, style={"margin": 0})],
                             style={"flex": "0 0 160px"}),

                    # Barra ancha
                    html.Div(
                        dcc.Graph(figure=bar_fig, config={"displayModeBar": False},
                                  style={"height": "60px", "width": "700px"}),
                        style={"flex": "0 0 auto"}
                    ),
                ],
                style={"display": "flex", "gap": 14, "alignItems":"center", "flexWrap":"wrap"},
            ),
            html.Small(
                f"Basado en medias anuales ({metric_note}) y comparación entre distritos de la ciudad. Menor valor ⇒ mayor seguridad.",
                style={"color": "#6B7280", "display": "block", "marginTop": 6}
            )
        ],
        style={"border": "1px solid #E5E7EB", "borderRadius": 12, "padding": 12, "marginTop": 8}
    )

# -------------------- UI auxiliar --------------------
has_reviews = "number_of_reviews" in listings.columns
NEIGH_OPTS = sorted(listings["neigh"].dropna().unique().tolist())
ROOM_TYPES = sorted(listings["room_type"].dropna().unique().tolist()) if "room_type" in listings.columns else []
MIN_P = int(pd.to_numeric(listings["price"], errors="coerce").min()) if "price" in listings.columns else 0
MAX_P = int(pd.to_numeric(listings["price"], errors="coerce").max()) if "price" in listings.columns else 1000

def _subset(neigh, room_types, price_range):
    """Subset de listings según filtros."""
    lo, hi = price_range
    mask = listings["neigh"].eq(neigh)
    if room_types:
        mask &= listings.get("room_type", pd.Series(index=listings.index, dtype=object)).isin(room_types)
    if "price" in listings.columns:
        mask &= pd.to_numeric(listings["price"], errors="coerce").between(lo, hi)
    return listings[mask].copy()

def make_points_map(df_subset: pd.DataFrame) -> go.Figure:
    """Mapa de puntos de listings en el barrio."""
    fig = go.Figure()
    if not df_subset.empty and {"latitude","longitude"}.issubset(df_subset.columns):
        latc = float(df_subset["latitude"].mean())
        lonc = float(df_subset["longitude"].mean())
        fig = px.scatter_mapbox(
            df_subset,
            lat="latitude", lon="longitude",
            color="price",
            hover_data={
                "id": True, "room_type": True, "price": ":.0f",
                "number_of_reviews": True if has_reviews else False,
                "latitude": False, "longitude": False
            },
            color_continuous_scale="Viridis",
            zoom=12, height=520,
        )
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox_center={"lat": latc, "lon": lonc},
            margin=dict(l=0, r=0, t=0, b=0),
        )
    else:
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox_center={"lat": 40.4168, "lon": -3.7038},
            mapbox_zoom=11.0,
            margin=dict(l=0, r=0, t=0, b=0),
            height=520,
        )
    return fig

def make_roomtype_donut(df_subset: pd.DataFrame) -> go.Figure:
    """Gráfico circular con composición por tipo de alojamiento."""
    fig = go.Figure()
    if df_subset.empty or "room_type" not in df_subset.columns:
        fig.update_layout(autosize=False, height=320, margin=dict(l=10, r=10, t=10, b=10))
        return fig
    s = (
        df_subset["room_type"].fillna("Unknown")
        .replace({"Entire home/apt":"Entire","Private room":"Private","Shared room":"Shared","Hotel room":"Hotel"})
        .value_counts(normalize=True).rename_axis("room_type").reset_index(name="pct")
    )
    if len(s) > 4:
        top = s.nlargest(3, "pct")
        other = pd.DataFrame([{"room_type": "Other", "pct": s["pct"].iloc[3:].sum()}])
        s = pd.concat([top, other], ignore_index=True)
    fig = px.pie(s, names="room_type", values="pct", hole=0.60)
    fig.update_traces(textposition="inside", texttemplate="%{label}<br>%{percent:.0%}")
    fig.update_layout(autosize=False, height=320, margin=dict(l=10, r=10, t=10, b=10),
                      showlegend=False, uirevision="donut-stable")
    return fig

def make_similarity_bar(neigh: str) -> go.Figure:
    """Gráfico de barras horizontales con barrios similares."""
    if SIM is None or not isinstance(SIM, pd.DataFrame) or neigh not in SIM.index:
        return go.Figure()
    sims = SIM.loc[neigh].drop(labels=[neigh], errors="ignore").sort_values(ascending=False).head(6)
    if sims.empty:
        return go.Figure()
    df = sims.reset_index()
    if df.shape[1] >= 2:
        first_col, second_col = df.columns[0], df.columns[1]
        df = df.rename(columns={first_col: "neighbourhood", second_col: "similarity"})
    else:
        df.columns = ["neighbourhood", "similarity"][: df.shape[1]]
    df["similarity"] = pd.to_numeric(df["similarity"], errors="coerce")
    df = df.dropna(subset=["similarity"])
    fig = px.bar(
        df.sort_values("similarity"),
        x="similarity", y="neighbourhood",
        orientation="h", text=df["similarity"].round(2),
        color_discrete_sequence=["#10B981"],
        labels={"similarity": "Similitud (cos)", "neighbourhood": "Barrio"},
    )
    fig.update_layout(margin=dict(l=0, r=10, t=6, b=6), height=360)
    fig.update_traces(customdata=df["neighbourhood"], hovertemplate="<b>%{y}</b><br>Sim: %{x:.2f}")
    return fig

# -------------------- Layout --------------------
layout = html.Div(
    [
        html.H2("Barrios"),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Barrio"),
                        dcc.Dropdown(
                            id="neigh-dd",
                            options=[{"label": n, "value": n} for n in NEIGH_OPTS],
                            value=NEIGH_OPTS[0] if NEIGH_OPTS else None,
                            style={"minWidth": 280, "zIndex": 9999},
                            placeholder="Selecciona un barrio",
                        ),
                    ],
                    style={"minWidth": "300px"},
                ),
                html.Div(
                    [
                        html.Label("Tipos de alojamiento"),
                        dcc.Dropdown(
                            id="room2-dd",
                            options=[{"label": rt, "value": rt} for rt in ROOM_TYPES],
                            value=ROOM_TYPES,
                            multi=True,
                            style={"minWidth": 320, "zIndex": 9999},
                        ),
                    ],
                ),
                html.Div(
                    [
                        html.Label("Rango de precio (€ por noche)"),
                        dcc.RangeSlider(
                            id="price2-slider",
                            min=MIN_P,
                            max=MAX_P,
                            value=[MIN_P, MAX_P],
                            tooltip={"always_visible": True},
                        ),
                    ],
                    style={"flex": "1", "paddingLeft": "12px", "minWidth": "260px"},
                ),
            ],
            style={"display": "flex","gap": "12px","alignItems": "center","flexWrap": "wrap","marginBottom": "8px"},
        ),

        dcc.Graph(id="map-neigh"),
        html.H4("Composición por tipo de alojamiento"),
        dcc.Graph(id="donut-roomtype", style={"height": "360px"}, config={"displayModeBar": False}),
        html.H4("Seguridad del barrio"),
        html.Div(id="security-panel"),
        html.H4("Barrios similares"),
        dcc.Graph(id="sim-bar"),
    ]
)

# -------------------- Callbacks --------------------
@dash.callback(
    Output("map-neigh", "figure"),
    Output("donut-roomtype", "figure"),
    Output("sim-bar", "figure"),
    Input("neigh-dd", "value"),
    Input("room2-dd", "value"),
    Input("price2-slider", "value"),
)
def update_neigh(neigh, room_types, price_range):
    """Actualizar visualizaciones según filtros de barrio, tipo de alojamiento y precio."""
    if isinstance(room_types, str):
        room_types = [room_types]
    if not neigh or not room_types or not price_range:
        empty_df = listings.iloc[0:0].copy()
        return make_points_map(empty_df), go.Figure(), go.Figure()
    d = _subset(neigh, room_types, price_range)
    fig_map = make_points_map(d)
    fig_donut = make_roomtype_donut(d)
    fig_sim = make_similarity_bar(neigh)
    return fig_map, fig_donut, fig_sim

@dash.callback(Output("security-panel", "children"), Input("neigh-dd", "value"))
def update_security_panel(neigh_value):
    return security_card_for_neigh_all_years(neigh_value)

@dash.callback(
    Output("neigh-dd", "value"),
    Input("sim-bar", "clickData"),
    State("neigh-dd", "value"),
    prevent_initial_call=True,
)
def jump_to_similar(clickData, current):
    """Al hacer clic en un barrio similar, actualizar el dropdown del barrio."""
    if not clickData:
        return no_update
    val = clickData["points"][0].get("customdata")
    if not val or val == current:
        return no_update
    return val
