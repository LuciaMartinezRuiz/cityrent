import dash
from dash import html

dash.register_page(__name__, path="/relations", name="Relaciones")

layout = html.Div(
    [
        html.H2("Relación precio vs. delincuencia (placeholder)"),
        html.P("Aquí se mostrará scatter + regresión y métricas (R², MAE)."),
    ]
)
