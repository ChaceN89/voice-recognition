from dash import html, dcc
import dash_bootstrap_components as dbc

layout = html.Div(
    children=[
        html.H2("Create Vocal Profile"),
        html.Div(
            className="side-by-side",
            children=[
                dcc.Input(id='model-name-input', type='text', placeholder='Profile Name...'),
                html.Button("Create Profile", id="create-model", className="button create-button"),
            ]
        ),
        html.Div( id="output-container", className="output-text" ),

        dcc.Interval(
            id='dropdown-refresh',
            interval=1000,  # in milliseconds
            n_intervals=0
        )


    ],
)