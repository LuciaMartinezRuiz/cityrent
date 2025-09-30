import dash
from dash import html, dcc

dash.register_page(__name__, path="/neighborhoods", name="Barrios")

layout = html.Div(
    [
        html.H2("Detalle por barrios (placeholder)"),
        html.P("Lista de propiedades filtrable por rango de precio, tipo, habitaciones, rating."),
        dcc.RangeSlider(min=0, max=500, step=10, value=[50, 200], tooltip={"always_visible": True}),
    ]
)
