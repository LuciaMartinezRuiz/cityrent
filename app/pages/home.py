import dash
from dash import html, dcc

dash.register_page(__name__, path="/", name="Inicio")

layout = html.Div(
    [
        html.H2("Mapa por ciudades (placeholder)"),
        html.P("Aquí irá el choropleth de precios medios por ciudad."),
        dcc.Slider(min=0, max=300, step=5, value=100, marks=None, tooltip={"placement": "bottom", "always_visible": True}),
    ]
)
