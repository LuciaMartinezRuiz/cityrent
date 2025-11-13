import dash
from dash import html, dcc, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
from pathlib import Path

dash.register_page(__name__, path="/", name="Inicio")

# datos
PROC = Path("data/processed")
GEO = Path("geo")
CITY = "madrid"

listings_all = pd.read_parquet(PROC / f"{CITY}_listings.parquet")

# rango de precio disponible
MIN_P = int(listings_all["price"].min())
MAX_P = int(listings_all["price"].max())

# eliminar NaNs
ROOM_TYPES = sorted(listings_all["room_type"].dropna().unique().tolist()) 
with open(GEO / "neighbourhoods.geojson", "r", encoding="utf-8") as f:
    GJ = json.load(f)

# -------------------- funciones --------------------
def subset(room_types, price_range):
    """Subset según filtros."""
    lo, hi = price_range
    return listings_all[
        listings_all["room_type"].isin(room_types)
        & listings_all["price"].between(lo, hi)
    ].copy()

def make_map(df_subset: pd.DataFrame) -> go.Figure:
    """Mapa por barrio según media de precios."""
    fig = go.Figure()
    if not df_subset.empty:
        agg = (
            df_subset.groupby("neigh", dropna=False)
            .agg(price_mean=("price", "mean"), n_listings=("id", "count"))
            .reset_index()
            .rename(columns={"neigh": "neighbourhood"})
        )
        geo_neighs = {ft["properties"]["neighbourhood"] for ft in GJ["features"]}
        agg = agg[agg["neighbourhood"].isin(geo_neighs)]

        fig.add_trace(
            go.Choroplethmapbox(
                geojson=GJ,
                featureidkey="properties.neighbourhood",
                locations=agg["neighbourhood"],
                z=agg["price_mean"].round(0),
                colorscale="Viridis",
                colorbar_title="Avg €",
                marker_line_width=0.5,
                marker_line_color="rgba(0,0,0,0.35)",
                hovertemplate="<b>%{location}</b><br>Avg: %{z} €<br>Listings: %{customdata}<extra></extra>",
                customdata=agg["n_listings"],
            )
        )

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_center={"lat": 40.4168, "lon": -3.7038},
        mapbox_zoom=9.8,
        margin=dict(l=0, r=0, t=0, b=0),
        height=560,
    )
    return fig

# -------------------- Layout --------------------
layout = html.Div(
    [
        html.H2("Madrid"),

        # filtros
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Tipo de alojamiento"),
                        dcc.Dropdown(
                            id="room-type-dd",
                            options=[{"label": rt, "value": rt} for rt in ROOM_TYPES],
                            value=ROOM_TYPES,
                            multi=True,
                            placeholder="Selecciona tipo(s) de habitación",
                            style={"minWidth": 280, "zIndex": 9999},
                        ),
                    ],
                    style={"minWidth": "280px", "flex": "1"},
                ),
                html.Div(
                    [
                        html.Label("Rango de precio (€ por noche)"),
                        dcc.RangeSlider(
                            id="price-slider",
                            min=MIN_P, max=MAX_P, value=[MIN_P, MAX_P],
                            tooltip={"always_visible": True},
                        ),
                    ],
                    style={"flex": "2", "paddingLeft": "16px"},
                ),
            ],
            style={
                "display": "flex",
                "gap": "12px",
                "alignItems": "center",
                "flexWrap": "wrap",
                "marginBottom": "8px",
            },
        ),

        # mapa
        dcc.Graph(id="map-agg"),

        # KPIs
        html.Div(
            id="kpi-row",
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(4, 1fr)",
                "gap": "10px",
                "margin": "8px 0",
            },
        ),

        # header y controles del ranking
        html.H4("Comparativa de precios por barrio", style={"marginTop": "8px"}),
        html.Div(
            [
                html.Label("Métrica"),
                dcc.Dropdown(
                    id="metric-dd",
                    options=[
                        {"label": "Precio medio (€)", "value": "price_mean"},
                        {"label": "Número de anuncios", "value": "n_listings"},
                    ],
                    value="price_mean",
                    clearable=False,
                    style={"width": 260},
                ),
                dcc.RadioItems(
                    id="order-dd",
                    options=[
                        {"label": "Top 15", "value": "desc"},
                        {"label": "Bottom 15", "value": "asc"},
                    ],
                    value="desc",
                    inline=True,
                    style={"marginLeft": "12px"},
                ),
            ],
            style={"display": "flex", "alignItems": "center", "gap": "12px", "margin": "6px 0"},
        ),

        # ranking
        dcc.Graph(id="bar-ranking"),
    ]
)

# -------------------- Callbacks --------------------
@dash.callback(
    Output("map-agg", "figure"),
    Output("kpi-row", "children"),
    Output("bar-ranking", "figure"),
    Input("room-type-dd", "value"),
    Input("price-slider", "value"),
    Input("metric-dd", "value"),
    Input("order-dd", "value"),
)
def update_all(room_types, price_range, metric, order_dir):
    """Actualizar visualizaciones según filtros."""
    d = subset(room_types, price_range)

    # KPIs
    kpis = []
    if not d.empty:
        kpi_vals = {
            "Anuncios": f"{len(d):,}".replace(",", "."),
            "€ mediana": f"{d['price'].median():.0f}",
            "€ p25–p75": f"{d['price'].quantile(.25):.0f}–{d['price'].quantile(.75):.0f}",
            "Barrios": f"{d['neigh'].nunique()}",
        }
        for title, val in kpi_vals.items():
            kpis.append(
                html.Div(
                    [
                        html.Div(title, style={"fontSize": "12px", "color": "#666"}),
                        html.Div(val, style={"fontSize": "24px", "fontWeight": "600"}),
                    ],
                    style={
                        "padding": "10px",
                        "border": "1px solid #eee",
                        "borderRadius": "10px",
                        "background": "#fff",
                    },
                )
            )

    # ranking por barrio
    agg = (
        d.groupby("neigh", dropna=False)
        .agg(price_mean=("price", "mean"),
             n_listings=("id", "count"))
        .reset_index()
        .rename(columns={"neigh": "neighbourhood"})
    )

    # top o bottom segun métrica
    ascending = (order_dir == "asc")
    top = agg.sort_values(metric, ascending=ascending).head(15)
    top = top.sort_values(metric, ascending=True)  # para que crezca de abajo arriba

    # color
    color = "#6BB9CC"  

    # gráfico de barras
    fig_bar = px.bar(
        top, x=metric, y="neighbourhood", orientation="h",
        labels={"neighbourhood": "Barrio", metric: "Valor"},
        text=top[metric].round(0),
        color_discrete_sequence=[color],
    )
    fig_bar.update_traces(hovertemplate="<b>%{y}</b><br>%{x:.0f}", marker_line_width=0.5)
    fig_bar.update_layout(margin=dict(l=0, r=10, t=6, b=6), height=520)

    return make_map(d), kpis, fig_bar
