import dash
from dash import html

dash.register_page(__name__, path="/about", name="Metodología & Datos")

layout = html.Div(
    [
        html.H2("Metodología & Datos"),
        html.Ul(
            [
                html.Li("Fuentes previstas: Inside Airbnb, datos municipales de delincuencia, INE/Eurostat."),
                html.Li("Limpieza y unión de datasets con Pandas/GeoPandas."),
                html.Li("Modelo de regresión con scikit-learn; explicabilidad básica."),
            ]
        ),
    ]
)
