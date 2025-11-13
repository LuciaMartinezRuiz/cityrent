from dash import Dash, html, dcc
import dash
import dash_bootstrap_components as dbc

# -------------------- app --------------------
app = Dash(
    __name__,
    use_pages=True,  # habilita multipágina
    external_stylesheets=[dbc.themes.SANDSTONE],
    suppress_callback_exceptions=True,  # necesario en multipágina
)
app.title = "CityRent"

server = app.server  # para render


# --- Definición de las rutas ---
# home.py -> path="/"
# neighborhoods.py -> path="/neighbourhoods"
# relations.py -> path="/relations"
# about.py -> path="/about"

# -------------------- Navbar horizontal (top) --------------------
navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand("CityRent", href="/", class_name="fw-semibold"),
            dbc.Nav(
                [
                    dbc.NavItem(dbc.NavLink("Inicio", href="/", active="exact", id="nav-home")),
                    dbc.NavItem(dbc.NavLink("Barrios", href="/neighbourhoods", active="exact", id="nav-neigh")),
                    dbc.NavItem(dbc.NavLink("Relaciones", href="/relations", active="exact", id="nav-rel")),
                    dbc.NavItem(dbc.NavLink("Metodología", href="/about", active="exact", id="nav-about")),
                ],
                pills=True, # estilo tipo “pestañas”
                justified=False,
                class_name="ms-auto" # empuja a la derecha
            ),
        ],
        fluid=True,
        class_name="py-2",
    ),
    color="pink",
    dark=False,
    sticky="top",  # fijo arriba al hacer scroll
)

# -------------------- Layout --------------------
app.layout = html.Div(
    [
        dcc.Location(id="url"), # necesario para resaltar NavLink activo
        navbar,
        html.Hr(className="mt-0"),
        dbc.Container(dash.page_container, fluid=True, class_name="pb-4"),
    ]
)

# -------------------- Run --------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run(host="127.0.0.1", port=port, debug=True)
