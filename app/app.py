from dash import Dash, html
import dash
import dash_bootstrap_components as dbc

app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "CityRent"

navbar = dbc.NavbarSimple(
    brand="CityRent",
    brand_href="/",
    color="light",
    dark=False,
)

app.layout = dbc.Container(
    [navbar, html.Hr(), dash.page_container],
    fluid=True
)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    # Dash >= 2.16 usa app.run en lugar de app.run_server
    app.run(host="127.0.0.1", port=port, debug=True)
